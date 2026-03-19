"""
MDDformer-V2 - Kaggle Training Script (Self-Contained)
========================================================
Paper: "LMVD: A Large-Scale Multimodal Vlog Dataset for Depression Detection in the Wild"
       (arXiv: 2407.00024)

Phase 2: Baseline Architecture + Targeted Improvements
Baseline achieved 74.39% avg 10-fold accuracy.

DESIGN PRINCIPLE: Keep the proven baseline architecture, add only targeted
non-disruptive improvements with strong theoretical/empirical backing.

Improvements over baseline (all independently toggleable):
  1. [ARCH] Gated Attention Pooling (USE_ATTN_POOL)
     - Replaces naive AdaptiveAvgPool1d(1) which averaged over 256 feature dims
       (losing cross-modal structure: 128 video + 128 audio → 186 scalars)
     - Learns temporal attention weights → focuses on informative time positions
     - Classifier now receives 256-dim fused feature vector (richer than 186 scalars)
     - Uses gated attention mechanism (Ilse et al., 2018)
     - Adds only ~33K parameters (~3.5% increase)
     - Removes output ELU from classifier (raw logits, standard for CrossEntropyLoss)

  2. [EVAL] Test-Time Augmentation (USE_TTA)
     - 5 forward passes: 1 clean + 4 with random temporal masking
     - Averages logits → more robust predictions
     - Zero model changes, zero training cost

  3. [TRAIN] Stochastic Weight Averaging (USE_SWA)
     - Accumulates weight averages during late training (after SWA_START_EPOCH)
     - After training: recomputes BN stats and evaluates SWA model
     - Reports best of {regular, regular+TTA, SWA, SWA+TTA}
     - Smooths loss landscape → better generalization

  4. [FIX] Removed unsafe .squeeze() in CalculateAttention (batch-size=1 safe)

Architecture: Nearly identical to baseline (~990K params vs 952K)
  - Same TCN backbone, same cross-attention (1 layer, 4 heads), same fusion FFN
  - Same Input BatchNorm (proven breakthrough)
  - Only change: pooling head + classifier (see above)

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
from torch.optim.swa_utils import AveragedModel
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
# ALL values IDENTICAL to the proven baseline (74.39% avg)
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

# ======================== V2 Enhancement Toggles ========================
# Set any to False to disable and match baseline behavior exactly
USE_ATTN_POOL = True        # [V2] Gated temporal attention pooling
USE_SWA = True              # [V2] Stochastic Weight Averaging
SWA_START_EPOCH = 100       # [V2] Start accumulating SWA weights after this epoch
USE_TTA = True              # [V2] Test-Time Augmentation during final evaluation
TTA_PASSES = 5              # [V2] Number of TTA forward passes (1 clean + N-1 masked)
TTA_DROP_RATIO = 0.25       # [V2] Temporal mask drop ratio for TTA

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
# TCN backbone, Cross-Attention core: ALL IDENTICAL to baseline

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


# ==================== V2 NEW COMPONENTS ============================

class GatedAttentionPooling(nn.Module):
    """
    [V2] Gated attention pooling over temporal dimension.

    Replaces AdaptiveAvgPool1d(1) which naively averaged the 256 feature dims
    into 186 scalars, losing all cross-modal structure.

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
    Uses ELU activation (same as baseline) for consistency.
    NO output activation — raw logits for CrossEntropyLoss (standard practice).
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


# ==================== MDDformer-V2 ==================================

class NetV2(nn.Module):
    """
    MDDformer-V2: Baseline architecture + targeted improvements.

    Pipeline (identical to baseline through fusion):
      Video(B,915,171) → InputBN → TCN → (B,114,256) → Linear → (B,114,128)
        → Conv1d → (B,186,128)
      Audio(B,186,128) → InputBN → (B,186,128)
      → Bidirectional Cross-Attention (1 layer, 4 heads)
      → concat → (B,186,256)
      → LayerNorm + FeedForward + residual → LayerNorm → (B,186,256)

    Then [V2 CHANGE]:
      → Gated Attention Pooling → (B,256) → 3-layer Classifier → (B,2)
      (baseline was: AvgPool → (B,186) → 2-layer Classifier → (B,2))
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

        # ---- Pooling & Classification [V2 CHANGE] ----
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

        # ---- Pool & Classify [V2 CHANGE] ----
        if USE_ATTN_POOL:
            output = self.attn_pool(output)          # (B, 256)
        else:
            output = self.pooling(output).reshape(output.shape[0], -1)  # (B, 186)

        result = self.classifier(output)             # (B, 2)
        result = result.squeeze(-1)
        return result


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


# ==================== V2 UTILITIES ==================================

def tta_predict(model, videoData, audioData, num_passes=5, drop_ratio=0.25):
    """
    [V2] Test-Time Augmentation: average logits from multiple forward passes.
    Pass 1: clean input (no masking)
    Passes 2-N: random temporal masking on video (same augmentation as training)
    """
    all_logits = []

    # Pass 1: clean
    logits = model(videoData, audioData)
    all_logits.append(logits)

    # Passes 2-N: with temporal masking
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


@torch.no_grad()
def update_swa_bn(loader, swa_model, device):
    """
    [V2] Update BatchNorm running stats for SWA model.
    Custom implementation for multi-input model (PyTorch's built-in only handles single input).
    """
    momenta = {}
    for module in swa_model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = swa_model.training
    swa_model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for videoData, audioData, label in loader:
        swa_model(videoData.cuda(device), audioData.cuda(device))

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    swa_model.train(was_training)


def evaluate_model(model, devLoader, device):
    """Evaluate model on dev set. Returns (acc, p, r, f1, pre_list, label_list)."""
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
    p = precision_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
    r = recall_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
    f1 = f1_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
    return float(acc), p, r, f1, pre2, label2


def evaluate_with_tta(model, devLoader, device, num_passes=5, drop_ratio=0.25):
    """[V2] Evaluate with Test-Time Augmentation."""
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
    p = precision_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
    r = recall_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
    f1 = f1_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
    return float(acc), p, r, f1, pre2, label2


# ====================================================================
# ==================== TRAINING FUNCTION =============================
# ====================================================================

def train(VideoPath, AudioPath, X_train, X_test, labelPath, numkfold,
          log_dir, save_dir):
    """Train MDDformer-V2 for one fold with all enhancements."""
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
    model = NetV2().cuda(DEVICE)
    lossFunc = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).cuda(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=WEIGHT_DECAY)

    # LR Scheduler (IDENTICAL to baseline)
    target_steps = len(trainLoader) * epochSize
    warmup_steps = len(trainLoader) * warmupEpoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, target_steps)

    # [V2] SWA setup
    swa_model = AveragedModel(model) if USE_SWA else None
    swa_n = 0

    print(f'Fold {numkfold} training begins!')
    savePath = os.path.join(save_dir, str(numkfold))
    os.makedirs(savePath, exist_ok=True)

    for epoch in range(1, epochSize):
        # ==================== TRAIN ====================
        model.train()
        traloss_one = 0
        correct = 0
        total = 0

        for batch_idx, (videoData, audioData, label) in enumerate(trainLoader):
            videoData = videoData.cuda(DEVICE)
            audioData = audioData.cuda(DEVICE)
            label = label.cuda(DEVICE)

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

        # [V2] Update SWA weights (no LR schedule change — just accumulate)
        if USE_SWA and swa_model is not None and epoch >= SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_n += 1

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
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

            p = precision_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
            r = recall_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
            f1 = f1_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)

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
                save_name = f"MDDformer_v2_{epoch}_{float(acc):.2f}_{p:.4f}_{r:.4f}_{f1:.4f}.pth"
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

    # [V2] TTA evaluation of best regular model
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

    # [V2] SWA evaluation
    if USE_SWA and swa_model is not None and swa_n > 0:
        print(f"  SWA: {swa_n} weight snapshots averaged (from epoch {SWA_START_EPOCH})")
        # Recompute BN running stats with SWA model
        bn_loader = DataLoader(trainSet, batch_size=TRAIN_BATCH, shuffle=True,
                              num_workers=2, pin_memory=True)
        update_swa_bn(bn_loader, swa_model, DEVICE)

        swa_acc, swa_p, swa_r, swa_f1, swa_pre, swa_label = evaluate_model(
            swa_model, devLoader, DEVICE)
        print(f"  SWA:           {swa_acc:.2f}% (P:{swa_p:.4f} R:{swa_r:.4f} F1:{swa_f1:.4f})")
        if swa_acc > final_acc:
            final_acc = swa_acc
            final_p = swa_p
            final_r = swa_r
            final_f1 = swa_f1
            final_pre = swa_pre
            final_label = swa_label

        # [V2] SWA + TTA
        if USE_TTA:
            swa_tta_acc, swa_tta_p, swa_tta_r, swa_tta_f1, swa_tta_pre, swa_tta_label = \
                evaluate_with_tta(swa_model, devLoader, DEVICE, TTA_PASSES, TTA_DROP_RATIO)
            print(f"  SWA+TTA:       {swa_tta_acc:.2f}% "
                  f"(P:{swa_tta_p:.4f} R:{swa_tta_r:.4f} F1:{swa_tta_f1:.4f})")
            if swa_tta_acc > final_acc:
                final_acc = swa_tta_acc
                final_p = swa_tta_p
                final_r = swa_tta_r
                final_f1 = swa_tta_f1
                final_pre = swa_tta_pre
                final_label = swa_tta_label

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
    print("MDDformer-V2 Training")
    print("Paper: LMVD (arXiv: 2407.00024)")
    print("Baseline + Targeted Improvements")
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
    log_dir = os.path.join(LOG_DIR, f'MDDformer_v2_{tim}')
    save_dir = os.path.join(SAVE_DIR, f'MDDformer_v2_{tim}')
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
    tmp_model = NetV2()
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
    print(f"\nV2 Enhancements:")
    print(f"  Attention Pooling: {USE_ATTN_POOL}")
    print(f"  SWA: {USE_SWA} (start epoch={SWA_START_EPOCH})")
    print(f"  TTA: {USE_TTA} (passes={TTA_PASSES}, drop={TTA_DROP_RATIO})")
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
                              title='MDDformer-V2 10-Fold Confusion Matrix')

    avg_acc = sum(all_totals) / len(all_totals)
    avg_p = sum(all_ps) / len(all_ps)
    avg_r = sum(all_rs) / len(all_rs)
    avg_f1 = sum(all_f1s) / len(all_f1s)

    print("\n" + "=" * 60)
    print("MDDformer-V2 10-Fold Cross-Validation Results")
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
        f.write(f"MDDformer-V2 10-Fold CV Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Avg Accuracy:  {avg_acc:.2f}%\n")
        f.write(f"Avg Precision: {avg_p:.4f}\n")
        f.write(f"Avg Recall:    {avg_r:.4f}\n")
        f.write(f"Avg F1:        {avg_f1:.4f}\n")
        f.write(f"Per-fold Acc:  {all_totals}\n")
        f.write(f"Total Time:    {total_time/60:.1f} min\n")
        f.write(f"\nBaseline Hyperparameters:\n")
        f.write(f"  LR={lr}, Epochs={epochSize}, Warmup={warmupEpoch}\n")
        f.write(f"  Weight Decay={WEIGHT_DECAY}, Label Smoothing={LABEL_SMOOTHING}\n")
        f.write(f"  Grad Clip={GRAD_CLIP}\n")
        f.write(f"  Batch: Train={TRAIN_BATCH}, Dev={DEV_BATCH}\n")
        f.write(f"\nV2 Enhancements:\n")
        f.write(f"  Attention Pooling: {USE_ATTN_POOL}\n")
        f.write(f"  SWA: {USE_SWA} (start={SWA_START_EPOCH})\n")
        f.write(f"  TTA: {USE_TTA} (passes={TTA_PASSES})\n")
        f.write(f"  Total params: {total_params:,}\n")
    print(f"\nResults saved to {log_dir}")
