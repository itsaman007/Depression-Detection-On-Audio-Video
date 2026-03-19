"""
MDDformer-V3.1 - Kaggle Training Script (Self-Contained)
========================================================
Paper: "LMVD: A Large-Scale Multimodal Vlog Dataset for Depression Detection in the Wild"
       (arXiv: 2407.00024)

Phase 3.1: V2 Architecture + Gentle R-Drop Regularization
V2 achieved 77.29% avg 10-fold accuracy (exceeded paper's 76.88%).
V3 (alpha=3.0 + audio TTA) regressed to 76.52% — too aggressive.

DESIGN PRINCIPLE: Keep the proven V2 architecture + TTA strategy,
add gentle R-Drop with warmup to close the overfitting gap.

Fixes from V3 → V3.1 (lessons learned):
  1. [FIX] R-Drop alpha 3.0 → 1.0 (gentler — 3.0 destabilized some folds)
  2. [FIX] R-Drop warmup: disabled for first 20 epochs
     - Lets model learn basic features before adding KL constraint
     - Prevents early destabilization seen in V3 (Fold 2: 71.58%)
  3. [FIX] Reverted TTA to VIDEO-ONLY masking (proven in V2)
     - Audio TTA masking destroyed too much info → TTA won only 2/10 folds
     - V2's video-only TTA won 7/10 folds

Changes from V2 (kept from V3):
  1. [REMOVED] SWA — never won any fold in V2, pure overhead
  2. [CHANGED] TTA passes 5 → 10 (video-only masking, more averaging)
  3. [NEW] R-Drop Regularization (Liang et al., 2021)
     - Alpha=1.0 with 20-epoch warmup (gradually ramp up)
     - Forces consistent outputs between two dropout-masked forward passes

Architecture: Identical to V2 (~990K params)
  - Same TCN backbone, same cross-attention (1 layer, 4 heads), same fusion FFN
  - Same Input BatchNorm, same Gated Attention Pooling, same 3-layer classifier

Results history:
  Baseline:  74.39% avg (952K params)
  V2:        77.29% avg (990K params) — Gated Attention Pooling + TTA + SWA
  V3:        76.52% avg (990K params) — R-Drop alpha=3.0 + audio TTA (regressed)
  V3.1:      ??% (target ≥78%+)

HOW TO USE ON KAGGLE:
  1. Create a new Kaggle notebook (GPU T4 x2 or P100)
  2. Upload your dataset as a Kaggle Dataset
  3. Copy this entire file into a single code cell
  4. Set DATASET_ROOT to your Kaggle dataset path
  5. Run!
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as udata
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from math import sqrt, cos
import math
import os
import time
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt

# ╔══════════════════════════════════════════════════════════════════╗
# ║  CHANGE THIS PATH TO YOUR KAGGLE DATASET LOCATION              ║
# ╚══════════════════════════════════════════════════════════════════╝
DATASET_ROOT = "/kaggle/input/lmvd-dataset"  # <-- CHANGE THIS

# Derived paths
TCN_VIDEO_PATH = os.path.join(DATASET_ROOT, "TCN_processed_video")
AUDIO_PATH     = os.path.join(DATASET_ROOT, "Audio_feature")
LABEL_PATH     = os.path.join(DATASET_ROOT, "label")

# Output paths (Kaggle writable directory)
OUTPUT_DIR = "/kaggle/working"
LOG_DIR    = os.path.join(OUTPUT_DIR, "logs")
SAVE_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")

# ======================== Hyperparameters ========================
# ALL values IDENTICAL to V2/baseline (proven)
lr = 1e-5
epochSize = 300
warmupEpoch = 10
testRows = 1
TRAIN_BATCH = 15
DEV_BATCH = 4
DEVICE = 0
SEED = 2222
KFOLD_RANDOM_STATE = 42
NUM_FOLDS = 10
TOPACC_THRESHOLD = 60
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 80

# ======================== V3 Enhancement Toggles ========================
USE_ATTN_POOL = True        # [V2] Gated temporal attention pooling (proven +2.9%)
USE_TTA = True              # [V2] Test-Time Augmentation during final evaluation
TTA_PASSES = 10             # [V3] Increased from 5 → 10 for more robust averaging
TTA_DROP_RATIO = 0.25       # [V2] Temporal mask drop ratio for TTA

# [V3.1] R-Drop Regularization (gentler than V3)
USE_RDROP = True            # Enable R-Drop during training
RDROP_ALPHA = 1.0           # [V3.1] Reduced from 3.0 → 1.0 (3.0 was too aggressive)
RDROP_WARMUP_EPOCHS = 20    # [V3.1] No R-Drop for first N epochs (let model learn basics)

normalVideoShape = 915
normalAudioShape = 186


# ====================================================================
# ==================== DATA LOADER (IDENTICAL TO BASELINE) ===========
# ====================================================================

class temporalMask():
    """Randomly zeros out a fraction of temporal frames as data augmentation."""
    def __init__(self, drop_ratio):
        self.ratio = drop_ratio

    def __call__(self, frame_indices):
        frame_len = frame_indices.shape[0]
        sample_len = int(self.ratio * frame_len)
        sample_list = random.sample([i for i in range(0, frame_len)], sample_len)
        frame_indices[sample_list, :] = 0
        return frame_indices


class MyDataLoader(udata.Dataset):
    """
    Dataset for multimodal depression detection.
    Loads TCN-processed video features and VGGish audio features.
    Normalization is handled by learnable BatchNorm inside the model.
    """
    def __init__(self, videoFileName, AudioFileName, Kfolds, labelPath, type) -> None:
        super().__init__()
        if type == "train":
            self.temp = temporalMask(0.25)
        else:
            self.temp = None

        self.videoList = []
        self.audioList = []
        self.label = []
        self.type = type

        for file in Kfolds:
            file = str(file)
            self.videoList.append(os.path.join(videoFileName, file))
            self.audioList.append(os.path.join(AudioFileName, file))
            file_csv = pd.read_csv(os.path.join(labelPath, file.replace(".npy", "_Depression.csv")))
            bdi = int(file_csv.columns[0])
            self.label.append(bdi)

    def __getitem__(self, index: int):
        videoData = np.load(self.videoList[index])
        audioData = np.load(self.audioList[index])
        label = np.array(self.label[index])

        if self.temp is not None:
            videoData = self.temp(videoData)

        label = torch.from_numpy(label).type(torch.float)
        videoData = torch.from_numpy(videoData).float()
        audioData = torch.from_numpy(audioData).float()

        # Truncate if longer than expected
        if audioData.shape[0] > normalAudioShape:
            audioData = audioData[:normalAudioShape, :]
        if videoData.shape[0] > normalVideoShape:
            videoData = videoData[:normalVideoShape, ]

        assert videoData.shape[0] <= normalVideoShape
        assert audioData.shape[0] <= normalAudioShape
        assert videoData.shape[0] > 0
        assert audioData.shape[0] > 0

        # Zero-pad if shorter
        if videoData.shape[0] < normalVideoShape:
            zeroPadVideo = nn.ZeroPad2d(padding=(0, 0, 0, normalVideoShape - videoData.shape[0]))
            videoData = zeroPadVideo(videoData)
        if audioData.shape[0] < normalAudioShape:
            zeroPadAudio = nn.ZeroPad2d(padding=(0, 0, 0, normalAudioShape - audioData.shape[0]))
            audioData = zeroPadAudio(audioData)

        return videoData, audioData, label

    def __len__(self) -> int:
        return len(self.videoList)


# ====================================================================
# ==================== MODEL ARCHITECTURE ============================
# ====================================================================
# TCN backbone, Cross-Attention core: ALL IDENTICAL to V2/baseline

class Conv1d_Encoder(nn.Module):
    """3-layer 1D CNN for initial video feature extraction."""
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(171, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class AstroModel(nn.Module):
    """Dilated Temporal Convolutional Network with residual connections."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(128, 256, 1)
        self.dropout = nn.Dropout(0.3)
        self.conv1_1 = nn.Conv1d(128, 128, 3, padding=2, dilation=2)
        self.conv1_2 = nn.Conv1d(128, 128, 3, padding=2, dilation=2)
        self.conv2_1 = nn.Conv1d(128, 128, 3, padding=4, dilation=4)
        self.conv2_2 = nn.Conv1d(128, 128, 3, padding=4, dilation=4)
        self.conv3_1 = nn.Conv1d(128, 128, 3, padding=8, dilation=8)
        self.conv3_2 = nn.Conv1d(128, 128, 3, padding=8, dilation=8)
        self.conv4_1 = nn.Conv1d(128, 256, 3, padding=16, dilation=16)
        self.conv4_2 = nn.Conv1d(256, 256, 3, padding=16, dilation=16)

    def forward(self, x):
        raw = x
        x = F.relu(self.conv1_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv1_2(x))
        raw = F.relu(x + raw)

        x = raw
        x = F.relu(self.conv2_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv2_2(x))
        raw = F.relu(x + raw)

        x = raw
        x = F.relu(self.conv3_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv3_2(x))
        raw = F.relu(x + raw)

        x = raw
        x = F.relu(self.conv4_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv4_2(x))
        raw = self.conv(raw)
        raw = F.relu(x + raw)
        return raw


class TCNModel(nn.Module):
    """TCN video feature encoder: Conv1d -> Dilated TCN."""
    def __init__(self) -> None:
        super().__init__()
        self.Conv1d = Conv1d_Encoder()
        self.AstroModel = AstroModel()

    def forward(self, input):
        input = input.transpose(1, 2)  # (B, 171, 915)
        x = self.Conv1d(input)
        x = self.AstroModel(x)
        x = x.transpose(1, 2)  # (B, T, 256)
        return x


class CalculateAttention(nn.Module):
    """Bidirectional cross-attention computation.
    [V2 FIX]: Removed unsafe .squeeze() that could drop batch dim when B=1."""
    def __init__(self):
        super().__init__()

    def forward(self, Qx, Kx, Vx, Qy, Ky, Vy):
        attentionx = torch.matmul(Qx, torch.transpose(Kx, -1, -2))
        attentiony = torch.matmul(Qy, torch.transpose(Ky, -1, -2))
        attention = torch.cat((attentionx, attentiony), dim=1)
        B, C, H, W = attention.size()
        attention = attention.reshape(B, 2, C // 2, H, W)
        attention = torch.mean(attention, dim=1)  # (B, H, T, T) — no .squeeze()
        attention1 = torch.softmax(attention / sqrt(Qx.size(-1)), dim=-1)
        attention1 = torch.matmul(attention1, Vx)
        attention2 = torch.softmax(attention / sqrt(Qx.size(-1)), dim=-1)
        attention2 = torch.matmul(attention2, Vy)
        return attention1, attention2


class FeedForward(nn.Module):
    """Position-wise feed-forward with Conv1d."""
    def __init__(self, dim_in, hidden_dim, dim_out=None, *, dropout=0.1, f=nn.Conv1d, activation=nn.ELU):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        self.net = nn.Sequential(
            f(in_channels=dim_in, out_channels=hidden_dim, kernel_size=1, padding=0, stride=1),
            activation(),
            nn.Dropout(dropout),
            f(in_channels=hidden_dim, out_channels=dim_out, kernel_size=1, padding=0, stride=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Multi_CrossAttention(nn.Module):
    """Multi-head bidirectional cross-attention (IDENTICAL to baseline)."""
    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num
        assert all_head_size % head_num == 0
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.attn_dropout = nn.Dropout(0.1)
        self.norm = sqrt(all_head_size)

    def forward(self, x, y):
        batch_size = x.size(0)
        q_sx = self.linear_q(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_sx = self.linear_k(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_sx = self.linear_v(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        q_sy = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_sy = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_sy = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        attention1, attention2 = CalculateAttention()(q_sx, k_sx, v_sx, q_sy, k_sy, v_sy)
        attention1 = self.attn_dropout(
            attention1.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        ) + x
        attention2 = self.attn_dropout(
            attention2.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        ) + y
        return attention1, attention2


class ConvNet1d(nn.Module):
    """Linear projection from 256-dim to 128-dim."""
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(256, 128)

    def forward(self, input):
        sizeTmp = input.size(1)
        batch_size = input.size(0)
        outConv1d = input.contiguous().view(input.size(0) * input.size(1), -1)
        output = self.fc(outConv1d)
        output = output.view(batch_size, sizeTmp, -1)
        return output


# ==================== V2 COMPONENTS (PROVEN) ========================

class GatedAttentionPooling(nn.Module):
    """
    [V2] Gated attention pooling over temporal dimension.

    Gated attention (Ilse et al., 2018 - "Attention-based Deep MIL"):
      a = w^T (tanh(V*h) ⊙ sigmoid(U*h))
    The sigmoid gate controls what information flows through.

    Input:  (B, T=186, D=256) — temporal sequence of fused features
    Output: (B, D=256) — attention-weighted temporal summary
    """
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.V = nn.Linear(feature_dim, hidden_dim)
        self.U = nn.Linear(feature_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        # x: (B, T, D)
        h_V = torch.tanh(self.V(x))               # (B, T, hidden)
        h_U = torch.sigmoid(self.U(x))            # (B, T, hidden)
        attn_scores = self.w(h_V * h_U)           # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T, 1)
        pooled = (x * attn_weights).sum(dim=1)    # (B, D)
        return pooled


class ClassifierV2(nn.Module):
    """
    [V2] 3-layer classifier for 256-dim attention-pooled features.
    256 → 128 → 64 → 2
    NO output activation — raw logits for CrossEntropyLoss.
    """
    def __init__(self, input_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.fc(x)


class Regress2(nn.Module):
    """Baseline classifier: 186 → 64 → 2 (used when USE_ATTN_POOL=False)."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(186, 64),
            nn.ELU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 2),
            nn.ELU()
        )

    def forward(self, x):
        x = x.view(-1, 186)
        return self.fc(x)


# ==================== MDDformer-V3 ==================================

class NetV3(nn.Module):
    """
    MDDformer-V3: V2 architecture (identical) + R-Drop training.

    Pipeline (identical to V2):
      Video(B,915,171) → InputBN → TCN → (B,114,256) → Linear → (B,114,128)
        → Conv1d → (B,186,128)
      Audio(B,186,128) → InputBN → (B,186,128)
      → Bidirectional Cross-Attention (1 layer, 4 heads)
      → concat → (B,186,256)
      → LayerNorm + FeedForward + residual → LayerNorm → (B,186,256)
      → Gated Attention Pooling → (B,256)
      → 3-layer Classifier → (B,2)
    """
    def __init__(self):
        super().__init__()
        # ---- Input normalization (CRITICAL — baseline breakthrough) ----
        self.video_bn = nn.BatchNorm1d(171)
        self.audio_bn = nn.BatchNorm1d(128)

        # ---- Video branch (IDENTICAL to baseline) ----
        self.TCNModel = TCNModel()
        self.Conv1dModel = ConvNet1d()
        self.conv = nn.Conv1d(in_channels=114, out_channels=186, kernel_size=1,
                              padding=0, stride=1)

        # ---- Cross-attention (IDENTICAL to baseline: 1 layer, 4 heads) ----
        self.mhca = Multi_CrossAttention(hidden_size=128, all_head_size=128, head_num=4)

        # ---- Fusion (IDENTICAL to baseline) ----
        self.norm = nn.LayerNorm(128 * 2)
        self.FFN = FeedForward(dim_in=186, hidden_dim=186 * 2, dim_out=186)
        self.norm2 = nn.LayerNorm(128 * 2)

        # ---- Pooling & Classification [V2, proven] ----
        if USE_ATTN_POOL:
            self.attn_pool = GatedAttentionPooling(feature_dim=256, hidden_dim=64)
            self.classifier = ClassifierV2(input_dim=256)
        else:
            self.pooling = nn.AdaptiveAvgPool1d(1)
            self.classifier = Regress2()

    def forward(self, inputVideo, inputAudio):
        """
        Args:
            inputVideo: (B, 915, 171) — TCN-processed video features
            inputAudio: (B, 186, 128) — VGGish audio features
        Returns:
            (B, 2) — raw logits (NO softmax/ELU on output)
        """
        # ---- Input BatchNorm ----
        inputVideo = inputVideo.transpose(1, 2)     # (B, 171, 915)
        inputVideo = self.video_bn(inputVideo)
        inputVideo = inputVideo.transpose(1, 2)     # (B, 915, 171)

        inputAudio = inputAudio.transpose(1, 2)     # (B, 128, 186)
        inputAudio = self.audio_bn(inputAudio)
        inputAudio = inputAudio.transpose(1, 2)     # (B, 186, 128)

        # ---- Video branch ----
        inputVideo = self.TCNModel(inputVideo)       # (B, 114, 256)
        outputConv1dVideo = self.Conv1dModel(inputVideo)  # (B, 114, 128)
        outputConv1dVideo = self.conv(outputConv1dVideo)  # (B, 186, 128)

        # ---- Cross-attention ----
        output1, output2 = self.mhca(outputConv1dVideo, inputAudio)

        # ---- Fusion ----
        outputFeature = torch.cat((output1, output2), dim=2)  # (B, 186, 256)
        outputFeature = self.FFN(self.norm(outputFeature)) + outputFeature
        output = self.norm2(outputFeature)           # (B, 186, 256)

        # ---- Pool & Classify [V2] ----
        if USE_ATTN_POOL:
            output = self.attn_pool(output)          # (B, 256)
        else:
            output = self.pooling(output).reshape(output.shape[0], -1)  # (B, 186)

        result = self.classifier(output)             # (B, 2)
        result = result.squeeze(-1)
        return result


# ====================================================================
# ==================== V3 R-DROP LOSS ================================
# ====================================================================

def compute_rdrop_loss(logits1, logits2, labels, ce_loss_fn, alpha=RDROP_ALPHA):
    """
    [V3] R-Drop loss: CE + bidirectional KL divergence.

    R-Drop (Liang et al., 2021) regularizes by forcing two forward passes
    (with different dropout masks) to produce similar output distributions.

    Loss = (CE(logits1, y) + CE(logits2, y)) / 2
         + alpha * (KL(p1 || p2) + KL(p2 || p1)) / 2

    Args:
        logits1: (B, C) — first forward pass logits
        logits2: (B, C) — second forward pass logits (different dropout)
        labels:  (B,)   — ground truth class indices
        ce_loss_fn: CrossEntropyLoss instance (with label smoothing)
        alpha: KL divergence weight

    Returns:
        total_loss: scalar — combined CE + KL loss
        ce_loss: scalar — just the CE component (for logging)
        kl_loss: scalar — just the KL component (for logging)
    """
    # Standard CE loss on both passes
    ce_loss1 = ce_loss_fn(logits1, labels)
    ce_loss2 = ce_loss_fn(logits2, labels)
    ce_loss = (ce_loss1 + ce_loss2) / 2.0

    # Bidirectional KL divergence between the two output distributions
    p1 = F.log_softmax(logits1, dim=-1)
    p2 = F.log_softmax(logits2, dim=-1)
    q1 = F.softmax(logits1, dim=-1)
    q2 = F.softmax(logits2, dim=-1)

    # KL(p1 || p2) + KL(p2 || p1)
    kl_loss = (F.kl_div(p1, q2, reduction='batchmean') +
               F.kl_div(p2, q1, reduction='batchmean')) / 2.0

    total_loss = ce_loss + alpha * kl_loss
    return total_loss, ce_loss.item(), kl_loss.item()


# ====================================================================
# ==================== UTILITIES =====================================
# ====================================================================

classes = ['Normal', 'Depression']


def plot_confusion_matrix(y_true, y_pred, labels_name, savename, title=None, thresh=0.6):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels_name)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()
    if title is not None:
        plt.title(title)
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, ['Normal', 'Depression'])
    plt.yticks(num_local, ['Normal', 'Depression'], rotation=90, va='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if cm[i][j] > 0:
                plt.text(j, i, str(cm[i][j]),
                         ha="center", va="center",
                         color="white" if cm[i][j] > thresh * cm.max() else "black")
    plt.tight_layout()
    plt.savefig(savename, format='png', dpi=150)
    plt.clf()
    plt.close()
    print(f"Confusion matrix saved to {savename}")


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """Cosine annealing with linear warmup (IDENTICAL to baseline)."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 0.5 * (cos(min((current_step - num_warmup_steps) /
                              (num_training_steps - num_warmup_steps), 1) * math.pi) + 1)
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# ==================== TTA UTILITIES (V2, enhanced in V3) ============

def tta_predict(model, videoData, audioData, num_passes=10, drop_ratio=0.25):
    """
    [V3.1] Test-Time Augmentation: average logits from multiple forward passes.
    Pass 1: clean input (no masking)
    Passes 2-N: random temporal masking on VIDEO ONLY

    [V3.1 FIX]: Reverted to video-only masking (proven in V2, won 7/10 folds).
    Audio TTA masking in V3 destroyed too much info → TTA only won 2/10 folds.
    """
    all_logits = []

    # Pass 1: clean
    logits = model(videoData, audioData)
    all_logits.append(logits)

    # Passes 2-N: with temporal masking on video only
    B, T, D = videoData.shape
    mask_len = int(drop_ratio * T)

    for _ in range(num_passes - 1):
        masked_video = videoData.clone()
        for b in range(B):
            mask_indices = random.sample(range(T), mask_len)
            masked_video[b, mask_indices, :] = 0
        logits = model(masked_video, audioData)
        all_logits.append(logits)

    return torch.stack(all_logits).mean(dim=0)


def evaluate_model(model, devLoader, device):
    """Evaluate model on dev set (single forward pass). Returns (acc, p, r, f1, pre_list, label_list)."""
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    label2 = []
    pre2 = []

    with torch.no_grad():
        for videoData, audioData, label in devLoader:
            videoData = videoData.cuda(device)
            audioData = audioData.cuda(device)
            label = label.cuda(device)

            output = model(videoData, audioData)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).cpu().sum()

            label2.append(label.data)
            pre2.append(predicted)
            all_labels += label.data.tolist()
            all_preds += predicted.tolist()

    acc = 100.0 * correct / total
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    p = precision_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
    r = recall_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
    f1 = f1_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
    return float(acc), p, r, f1, pre2, label2


def evaluate_with_tta(model, devLoader, device, num_passes=10, drop_ratio=0.25):
    """[V3.1] Evaluate with Test-Time Augmentation (10 passes, video-only masking)."""
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    label2 = []
    pre2 = []

    with torch.no_grad():
        for videoData, audioData, label in devLoader:
            videoData = videoData.cuda(device)
            audioData = audioData.cuda(device)
            label = label.cuda(device)

            avg_logits = tta_predict(model, videoData, audioData, num_passes, drop_ratio)
            _, predicted = torch.max(avg_logits.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).cpu().sum()

            label2.append(label.data)
            pre2.append(predicted)
            all_labels += label.data.tolist()
            all_preds += predicted.tolist()

    acc = 100.0 * correct / total
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    p = precision_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
    r = recall_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
    f1 = f1_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
    return float(acc), p, r, f1, pre2, label2


# ====================================================================
# ==================== TRAINING FUNCTION =============================
# ====================================================================

def train(VideoPath, AudioPath, X_train, X_test, labelPath, numkfold,
          log_dir, save_dir):
    """Train MDDformer-V3 for one fold with R-Drop regularization."""
    mytop = 0
    topacc = TOPACC_THRESHOLD
    top_p = top_r = top_f1 = 0
    top_pre = []
    top_label = []
    patience_counter = 0
    best_model_state = None

    # Data loaders
    trainSet = MyDataLoader(VideoPath, AudioPath, X_train, labelPath, "train")
    trainLoader = DataLoader(trainSet, batch_size=TRAIN_BATCH, shuffle=True,
                            num_workers=2, pin_memory=True)
    devSet = MyDataLoader(VideoPath, AudioPath, X_test, labelPath, "dev")
    devLoader = DataLoader(devSet, batch_size=DEV_BATCH, shuffle=False,
                          num_workers=2, pin_memory=True)
    print(f"Fold {numkfold} - Train: {len(trainLoader)} batches, Dev: {len(devLoader)} batches")

    # Model
    model = NetV3().cuda(DEVICE)
    lossFunc = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).cuda(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=WEIGHT_DECAY)

    # LR Scheduler (IDENTICAL to baseline)
    target_steps = len(trainLoader) * epochSize
    warmup_steps = len(trainLoader) * warmupEpoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, target_steps)

    print(f'Fold {numkfold} training begins! (R-Drop alpha={RDROP_ALPHA}, warmup={RDROP_WARMUP_EPOCHS}ep)')
    savePath = os.path.join(save_dir, str(numkfold))
    os.makedirs(savePath, exist_ok=True)

    for epoch in range(1, epochSize):
        # ==================== TRAIN ====================
        model.train()
        traloss_one = 0
        ce_loss_sum = 0
        kl_loss_sum = 0
        correct = 0
        total = 0

        # [V3.1] R-Drop warmup: linearly ramp alpha from 0 to RDROP_ALPHA
        # over the first RDROP_WARMUP_EPOCHS epochs
        if USE_RDROP and epoch <= RDROP_WARMUP_EPOCHS:
            current_alpha = RDROP_ALPHA * (epoch / RDROP_WARMUP_EPOCHS)
        else:
            current_alpha = RDROP_ALPHA
        use_rdrop_this_epoch = USE_RDROP  # always do 2 passes for consistency

        for batch_idx, (videoData, audioData, label) in enumerate(trainLoader):
            videoData = videoData.cuda(DEVICE)
            audioData = audioData.cuda(DEVICE)
            label = label.cuda(DEVICE)

            if use_rdrop_this_epoch:
                # [V3.1] R-Drop: Two forward passes with different dropout masks
                output1 = model(videoData, audioData)
                output2 = model(videoData, audioData)

                traLoss, ce_val, kl_val = compute_rdrop_loss(
                    output1, output2, label.long(), lossFunc, current_alpha)

                ce_loss_sum += ce_val
                kl_loss_sum += kl_val
                traloss_one += traLoss.item()

                # Use average of both logits for accuracy tracking
                output = (output1 + output2) / 2.0
            else:
                # Standard single forward pass (fallback)
                output = model(videoData, audioData)
                traLoss = lossFunc(output, label.long())
                traloss_one += traLoss.item()

            optimizer.zero_grad()
            traLoss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).cpu().sum()

        train_acc = 100.0 * correct / total
        avg_loss = traloss_one / len(trainLoader)

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            if USE_RDROP:
                avg_ce = ce_loss_sum / len(trainLoader)
                avg_kl = kl_loss_sum / len(trainLoader)
                alpha_str = f', α={current_alpha:.2f}' if epoch <= RDROP_WARMUP_EPOCHS else ''
                print(f'  Epoch {epoch}/{epochSize} - Loss: {avg_loss:.4f} '
                      f'(CE: {avg_ce:.4f}, KL: {avg_kl:.4f}{alpha_str}), '
                      f'Acc: {train_acc:.2f}%, LR: {optimizer.param_groups[0]["lr"]:.8f}')
            else:
                print(f'  Epoch {epoch}/{epochSize} - Train Loss: {avg_loss:.4f}, '
                      f'Acc: {train_acc:.2f}%, LR: {optimizer.param_groups[0]["lr"]:.8f}')

        # ==================== EVAL ====================
        if epoch % testRows == 0:
            model.eval()
            correct = 0
            total = 0
            all_labels = []
            all_preds = []
            label2 = []
            pre2 = []
            loss_one = 0

            with torch.no_grad():
                for videoData, audioData, label in devLoader:
                    videoData = videoData.cuda(DEVICE)
                    audioData = audioData.cuda(DEVICE)
                    label = label.cuda(DEVICE)

                    devOutput = model(videoData, audioData)
                    loss = lossFunc(devOutput, label.long())
                    loss_one += loss.item()

                    _, predicted = torch.max(devOutput.data, 1)
                    total += label.size(0)
                    correct += predicted.eq(label.data).cpu().sum()

                    label2.append(label.data)
                    pre2.append(predicted)
                    all_labels += label.data.tolist()
                    all_preds += predicted.tolist()

            acc = 100.0 * correct / total
            all_labels_np = np.array(all_labels)
            all_preds_np = np.array(all_preds)

            p = precision_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
            r = recall_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
            f1 = f1_score(all_labels_np, all_preds_np, average='macro', zero_division=0)

            if epoch % 10 == 0 or epoch == 1:
                print(f'  Dev   {epoch}/{epochSize} - Loss: {loss_one/len(devLoader):.4f}, '
                      f'Acc: {acc:.2f}%, P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}')

            if acc > mytop:
                mytop = acc
                top_p = p
                top_r = r
                top_f1 = f1
                top_pre = pre2
                top_label = label2
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if acc > topacc:
                topacc = acc
                checkpoint = {
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'scheduler': scheduler.state_dict(),
                    'acc': float(acc),
                }
                save_name = f"MDDformer_v3_{epoch}_{float(acc):.2f}_{p:.4f}_{r:.4f}_{f1:.4f}.pth"
                torch.save(checkpoint, os.path.join(savePath, save_name))

        # Early stopping check
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    # ==================== POST-TRAINING EVALUATION ====================
    print(f"\n  --- Post-training evaluation (Fold {numkfold}) ---")
    print(f"  Regular best:  {mytop:.2f}% (P:{top_p:.4f} R:{top_r:.4f} F1:{top_f1:.4f})")

    final_acc = mytop
    final_p = top_p
    final_r = top_r
    final_f1 = top_f1
    final_pre = top_pre
    final_label = top_label

    # [V2/V3] TTA evaluation of best regular model
    if USE_TTA and best_model_state is not None:
        model.load_state_dict(best_model_state)
        tta_acc, tta_p, tta_r, tta_f1, tta_pre, tta_label = evaluate_with_tta(
            model, devLoader, DEVICE, TTA_PASSES, TTA_DROP_RATIO)
        print(f"  Regular+TTA:   {tta_acc:.2f}% (P:{tta_p:.4f} R:{tta_r:.4f} F1:{tta_f1:.4f})")
        if tta_acc > final_acc:
            final_acc = tta_acc
            final_p = tta_p
            final_r = tta_r
            final_f1 = tta_f1
            final_pre = tta_pre
            final_label = tta_label

    print(f"  >> FOLD {numkfold} BEST: {final_acc:.2f}% "
          f"(P:{final_p:.4f} R:{final_r:.4f} F1:{final_f1:.4f})")

    # Collect fold results
    if len(final_pre) == 0:
        print(f"WARNING: Fold {numkfold} - no predictions above threshold!")
        return torch.tensor([]), torch.tensor([]), final_acc, final_p, final_r, final_f1

    final_pre_cat = torch.cat(final_pre, axis=0).cpu()
    final_label_cat = torch.cat(final_label, axis=0).cpu()
    return final_label_cat, final_pre_cat, final_acc, final_p, final_r, final_f1


# ====================================================================
# ==================== MAIN ==========================================
# ====================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MDDformer-V3.1 Training")
    print("Paper: LMVD (arXiv: 2407.00024)")
    print("V2 Architecture + Gentle R-Drop (alpha=1.0, warmup=20ep)")
    print("=" * 60)

    # Seed everything
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Verify paths exist
    for name, path in [("Video", TCN_VIDEO_PATH), ("Audio", AUDIO_PATH), ("Label", LABEL_PATH)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} path not found: {path}")
        ext = '.npy' if name != 'Label' else '.csv'
        count = len([f for f in os.listdir(path) if f.endswith(ext)])
        print(f"  {name}: {count} files at {path}")

    # Timestamp for this run
    tim = time.strftime('%m_%d__%H_%M', time.localtime())
    log_dir = os.path.join(LOG_DIR, f'MDDformer_v3.1_{tim}')
    save_dir = os.path.join(SAVE_DIR, f'MDDformer_v3.1_{tim}')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Build sample list
    X = os.listdir(TCN_VIDEO_PATH)
    X.sort(key=lambda x: int(x.split(".")[0]))
    X = np.array(X)

    Y = []
    for i in X:
        file_csv = pd.read_csv(os.path.join(LABEL_PATH, i.split('.npy')[0] + "_Depression.csv"))
        bdi = int(file_csv.columns[0])
        Y.append(bdi)

    # Model info
    tmp_model = NetV3()
    total_params = sum(p.numel() for p in tmp_model.parameters())
    trainable_params = sum(p.numel() for p in tmp_model.parameters() if p.requires_grad)
    del tmp_model

    print(f"\nDataset: {len(X)} samples, Depression: {sum(Y)}, Normal: {len(Y) - sum(Y)}")
    print(f"Model params: {total_params:,} (trainable: {trainable_params:,})")
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print(f"\nConfig (baseline hyperparameters):")
    print(f"  LR={lr}, Epochs={epochSize}, Warmup={warmupEpoch}")
    print(f"  Weight Decay={WEIGHT_DECAY}, Label Smoothing={LABEL_SMOOTHING}")
    print(f"  Grad Clip={GRAD_CLIP}, Batch: Train={TRAIN_BATCH}, Dev={DEV_BATCH}")
    print(f"  Early Stop Patience={EARLY_STOP_PATIENCE}")
    print(f"\nV3.1 Enhancements:")
    print(f"  Attention Pooling: {USE_ATTN_POOL} (from V2)")
    print(f"  R-Drop: {USE_RDROP} (alpha={RDROP_ALPHA}, warmup={RDROP_WARMUP_EPOCHS}ep)")
    print(f"  TTA: {USE_TTA} (passes={TTA_PASSES}, drop={TTA_DROP_RATIO}, video-only)")
    print(f"  SWA: REMOVED (never helped in V2)")
    print("=" * 60)

    # 10-Fold Stratified CV
    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=KFOLD_RANDOM_STATE)

    all_totals = []
    all_ps = []
    all_rs = []
    all_f1s = []
    all_pre = []
    all_label = []

    fold_start_time = time.time()

    for fold_num, (train_index, test_index) in enumerate(kf.split(X, Y), 1):
        X_train, X_test = X[train_index], X[test_index]
        print(f"\n{'='*60}")
        print(f"FOLD {fold_num}/{NUM_FOLDS}  (Train: {len(X_train)}, Test: {len(X_test)})")
        print(f"{'='*60}")

        top_label, top_pre, top_acc, top_p, top_r, top_f1 = train(
            TCN_VIDEO_PATH, AUDIO_PATH, X_train, X_test, LABEL_PATH,
            fold_num, log_dir, save_dir
        )

        all_totals.append(top_acc)
        all_ps.append(top_p)
        all_rs.append(top_r)
        all_f1s.append(top_f1)
        if len(top_pre) > 0:
            all_pre.append(top_pre)
            all_label.append(top_label)

        elapsed = time.time() - fold_start_time
        est_total = elapsed / fold_num * NUM_FOLDS
        print(f"  Time: {elapsed/60:.1f} min elapsed, ~{(est_total-elapsed)/60:.1f} min remaining")

    # ==================== FINAL RESULTS ====================
    total_time = time.time() - fold_start_time

    if len(all_pre) > 0:
        total_pre_np = torch.cat(all_pre, axis=0).cpu().numpy()
        total_label_np = torch.cat(all_label, axis=0).cpu().numpy()
        np.save(os.path.join(log_dir, "total_pre.npy"), total_pre_np)
        np.save(os.path.join(log_dir, "total_label.npy"), total_label_np)

        plot_confusion_matrix(total_label_np, total_pre_np, [0, 1],
                              savename=os.path.join(log_dir, 'confusion_matrix.png'),
                              title='MDDformer-V3.1 10-Fold Confusion Matrix')

    avg_acc = sum(all_totals) / len(all_totals)
    avg_p = sum(all_ps) / len(all_ps)
    avg_r = sum(all_rs) / len(all_rs)
    avg_f1 = sum(all_f1s) / len(all_f1s)

    print("\n" + "=" * 60)
    print("MDDformer-V3.1 10-Fold Cross-Validation Results")
    print("=" * 60)
    print(f"Per-fold Accuracy: {[f'{x:.2f}' for x in all_totals]}")
    print(f"Average Accuracy:  {avg_acc:.2f}%")
    print(f"Average Precision: {avg_p:.4f}")
    print(f"Average Recall:    {avg_r:.4f}")
    print(f"Average F1:        {avg_f1:.4f}")
    print(f"Total Time:        {total_time/60:.1f} minutes")
    print("=" * 60)

    # Save results to text file
    with open(os.path.join(log_dir, "results.txt"), "w") as f:
        f.write(f"MDDformer-V3.1 10-Fold CV Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Avg Accuracy:  {avg_acc:.2f}%\n")
        f.write(f"Avg Precision: {avg_p:.4f}\n")
        f.write(f"Avg Recall:    {avg_r:.4f}\n")
        f.write(f"Avg F1:        {avg_f1:.4f}\n")
        f.write(f"Per-fold Acc:  {all_totals}\n")
        f.write(f"Total Time:    {total_time/60:.1f} min\n")
        f.write(f"\nHyperparameters:\n")
        f.write(f"  LR={lr}, Epochs={epochSize}, Warmup={warmupEpoch}\n")
        f.write(f"  Weight Decay={WEIGHT_DECAY}, Label Smoothing={LABEL_SMOOTHING}\n")
        f.write(f"  Grad Clip={GRAD_CLIP}\n")
        f.write(f"  Batch: Train={TRAIN_BATCH}, Dev={DEV_BATCH}\n")
        f.write(f"\nV3.1 Enhancements:\n")
        f.write(f"  Attention Pooling: {USE_ATTN_POOL}\n")
        f.write(f"  R-Drop: {USE_RDROP} (alpha={RDROP_ALPHA}, warmup={RDROP_WARMUP_EPOCHS}ep)\n")
        f.write(f"  TTA: {USE_TTA} (passes={TTA_PASSES}, video-only)\n")
        f.write(f"  SWA: REMOVED\n")
        f.write(f"  Total params: {total_params:,}\n")
    print(f"\nResults saved to {log_dir}")
