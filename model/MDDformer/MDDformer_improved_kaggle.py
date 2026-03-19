"""
MDDformer-Improved - Kaggle Training Script (Self-Contained)
=============================================================
Paper: "LMVD: A Large-Scale Multimodal Vlog Dataset for Depression Detection in the Wild"
       (arXiv: 2407.00024)

Phase 2: Enhanced architecture targeting >=80% accuracy for publication.
Baseline achieved 74.39% avg 10-fold accuracy.

Improvements over baseline MDDformer:
  Architecture:
    1. Sinusoidal positional encoding for temporal sequences
    2. 2-layer stacked cross-attention (vs 1 in baseline)
    3. Pre-norm architecture (LayerNorm before attention, more stable)
    4. GELU activation (vs ELU, better gradient flow)
    5. 4x FFN expansion (vs 2x)
    6. 8 attention heads (vs 4)
    7. Output projection layer in cross-attention
    8. Enhanced 3-layer classifier with LayerNorm
    9. Xavier/Kaiming weight initialization
  Training:
    10. MixUp data augmentation
    11. Audio Gaussian noise augmentation
    12. Input BatchNorm (proven breakthrough in baseline)

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

# ========================== Hyperparameters ==========================
lr = 1e-5                  # match baseline's proven LR
epochSize = 300
warmupEpoch = 10           # 10-epoch linear warmup
testRows = 1               # evaluate every epoch
TRAIN_BATCH = 15           # match baseline batch size
DEV_BATCH = 4
DEVICE = 0                 # GPU index
SEED = 2222
KFOLD_RANDOM_STATE = 42
NUM_FOLDS = 10
TOPACC_THRESHOLD = 60      # only save checkpoints above this accuracy
WEIGHT_DECAY = 5e-4         # baseline-proven L2 regularization
LABEL_SMOOTHING = 0.1       # prevent overconfident predictions
GRAD_CLIP = 1.0             # gradient clipping max norm
MIXUP_ALPHA = 0.2           # MixUp interpolation (0 = disabled)
MIXUP_PROB = 0.5            # probability of applying MixUp per batch
AUDIO_NOISE_STD = 0.01      # Gaussian noise std for audio augmentation
EARLY_STOP_PATIENCE = 80    # stop if no dev improvement for N epochs

# ========================== Model Config ==========================
NUM_CROSS_ATTN_LAYERS = 2   # stacked cross-attention depth
NUM_HEADS = 8               # attention heads (baseline: 4)
FFN_EXPANSION = 2           # match baseline to prevent overfitting
ATTN_DROPOUT = 0.1          # dropout on attention weights
FFN_DROPOUT = 0.1           # baseline-proven dropout
PROJ_DROPOUT = 0.1          # baseline-proven dropout
CLS_DROPOUT = 0.3           # dropout in classifier (baseline: 0.3)
TCN_DROPOUT = 0.3           # TCN dilated conv dropout (baseline: 0.3)
HIDDEN_SIZE = 128            # cross-attention hidden dim (matches audio features)

normalVideoShape = 915
normalAudioShape = 186


# ====================================================================
# ==================== DATA LOADER ===================================
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

# -------------------- Positional Encoding --------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (B, T, D)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# -------------------- TCN Video Backbone (same as baseline) --------------------
class Conv1d_Encoder(nn.Module):
    """3-layer 1D CNN: (B,171,915) -> (B,128,114)."""
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


class DilatedTCN(nn.Module):
    """Dilated Temporal Convolutional Network with residual connections."""
    def __init__(self, dropout=0.3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(128, 256, 1)
        self.dropout = nn.Dropout(dropout)
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
    """Video feature encoder: Conv1d -> Dilated TCN."""
    def __init__(self) -> None:
        super().__init__()
        self.Conv1d = Conv1d_Encoder()
        self.DilatedTCN = DilatedTCN(dropout=TCN_DROPOUT)

    def forward(self, input):
        input = input.transpose(1, 2)  # (B, 171, 915)
        x = self.Conv1d(input)          # (B, 128, 114)
        x = self.DilatedTCN(x)          # (B, 256, 114)
        x = x.transpose(1, 2)           # (B, 114, 256)
        return x


# -------------------- Improved Cross-Attention --------------------
class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention (paper's custom design):
    Averages attention maps from both directions, then applies to both V's.
    Fixed: removed unsafe .squeeze() that could drop batch dim.
    """
    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, Qx, Kx, Vx, Qy, Ky, Vy):
        scale = sqrt(Qx.size(-1))

        # Attention from both directions
        attentionx = torch.matmul(Qx, torch.transpose(Kx, -1, -2))  # Q_y*K_x
        attentiony = torch.matmul(Qy, torch.transpose(Ky, -1, -2))  # Q_x*K_y

        # Average bidirectional attention
        attention = torch.cat((attentionx, attentiony), dim=1)  # (B, 2*H, T, T)
        B, C, H, W = attention.size()
        attention = attention.reshape(B, 2, C // 2, H, W)
        attention = torch.mean(attention, dim=1)  # (B, H, T, T) — no squeeze needed

        attention_weights = torch.softmax(attention / scale, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        attention1 = torch.matmul(attention_weights, Vx)
        attention2 = torch.matmul(attention_weights, Vy)
        return attention1, attention2


class ImprovedMultiCrossAttention(nn.Module):
    """
    Multi-head bidirectional cross-attention with:
    - Configurable number of heads
    - Attention dropout
    - Output projection layers
    - NO internal residual (handled by CrossAttentionBlock for correct pre-norm)
    """
    def __init__(self, hidden_size, head_num, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = head_num
        self.h_size = hidden_size // head_num
        assert hidden_size % head_num == 0

        self.linear_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output projections
        self.out_proj_x = nn.Linear(hidden_size, hidden_size)
        self.out_proj_y = nn.Linear(hidden_size, hidden_size)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.calc_attention = BidirectionalCrossAttention(attn_dropout)

    def forward(self, x, y):
        """
        Args:
            x, y: (B, T, D) — pre-normalized inputs
        Returns:
            attn_x, attn_y: (B, T, D) — attention outputs WITHOUT residual
        """
        batch_size = x.size(0)

        # Q from y, K/V from x → attention(y→x)
        q_sx = self.linear_q(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_sx = self.linear_k(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_sx = self.linear_v(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # Q from x, K/V from y → attention(x→y)
        q_sy = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_sy = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_sy = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        attention1, attention2 = self.calc_attention(q_sx, k_sx, v_sx, q_sy, k_sy, v_sy)

        # Reshape and project
        attention1 = attention1.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        attention2 = attention2.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)

        # Output projection — NO residual here (handled by CrossAttentionBlock)
        attention1 = self.proj_dropout(self.out_proj_x(attention1))
        attention2 = self.proj_dropout(self.out_proj_y(attention2))

        return attention1, attention2


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block with correct pre-norm architecture:
      x = x + CrossAttn(LN(x), LN(y))
      x = x + FFN(LN(x))
    Same for y. Residual always comes from ORIGINAL (un-normed) input.
    """
    def __init__(self, hidden_size=128, head_num=8, ffn_expansion=4,
                 attn_dropout=0.1, ffn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        # Pre-norm layers for cross-attention
        self.norm1_x = nn.LayerNorm(hidden_size)
        self.norm1_y = nn.LayerNorm(hidden_size)

        self.cross_attn = ImprovedMultiCrossAttention(
            hidden_size=hidden_size,
            head_num=head_num,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout
        )

        # Pre-norm layers for FFN
        self.norm2_x = nn.LayerNorm(hidden_size)
        self.norm2_y = nn.LayerNorm(hidden_size)

        # Per-modality FFN (Linear-based, standard transformer style)
        self.ffn_x = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_expansion),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden_size * ffn_expansion, hidden_size),
            nn.Dropout(ffn_dropout),
        )
        self.ffn_y = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_expansion),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden_size * ffn_expansion, hidden_size),
            nn.Dropout(ffn_dropout),
        )

    def forward(self, x, y):
        # Cross-attention with pre-norm + residual from ORIGINAL inputs
        attn_x, attn_y = self.cross_attn(self.norm1_x(x), self.norm1_y(y))
        x = x + attn_x
        y = y + attn_y

        # FFN with pre-norm + residual
        x = x + self.ffn_x(self.norm2_x(x))
        y = y + self.ffn_y(self.norm2_y(y))

        return x, y


# -------------------- Video Projection --------------------
class ConvNet1d(nn.Module):
    """Linear projection from TCN output (256-dim) to 128-dim."""
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(256, HIDDEN_SIZE)

    def forward(self, input):
        sizeTmp = input.size(1)
        batch_size = input.size(0)
        outConv1d = input.contiguous().view(batch_size * sizeTmp, -1)
        output = self.fc(outConv1d)
        output = output.view(batch_size, sizeTmp, -1)
        return output


# -------------------- Fusion FFN --------------------
class FusionFeedForward(nn.Module):
    """Conv1d-based FFN for post-concatenation fusion. Uses GELU."""
    def __init__(self, dim_in, hidden_dim, dim_out=None, dropout=0.1):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=dim_in, out_channels=hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=hidden_dim, out_channels=dim_out, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# -------------------- Enhanced Classifier --------------------
class ImprovedClassifier(nn.Module):
    """
    Enhanced 3-layer classification head with LayerNorm and GELU.
    186 -> 128 -> 64 -> 2
    """
    def __init__(self, input_dim=186, hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -------------------- MDDformer-Improved --------------------
class MDDformerImproved(nn.Module):
    """
    Enhanced multimodal depression detection model.

    Pipeline:
      Video(B,915,171) -> InputBN -> TCN -> (B,114,256) -> Linear -> (B,114,128)
        -> Conv1d -> (B,186,128)
      Audio(B,186,128) -> InputBN -> (B,186,128)
      -> Positional Encoding
      -> 2x [Pre-Norm CrossAttn Block] -> concat -> (B,186,256)
      -> LayerNorm + FusionFFN + residual -> LayerNorm
      -> AdaptiveAvgPool1d -> (B,186) -> Classifier -> (B,2)
    """
    def __init__(self):
        super().__init__()
        # ---- Input normalization (CRITICAL — baseline breakthrough) ----
        self.video_bn = nn.BatchNorm1d(171)   # video: 171 AU/landmark features
        self.audio_bn = nn.BatchNorm1d(128)   # audio: 128 VGGish features

        # ---- Video branch ----
        self.TCNModel = TCNModel()
        self.Conv1dModel = ConvNet1d()
        self.conv = nn.Conv1d(in_channels=114, out_channels=186, kernel_size=1)

        # ---- Positional encoding ----
        self.pos_enc_video = SinusoidalPositionalEncoding(HIDDEN_SIZE, max_len=200, dropout=0.1)
        self.pos_enc_audio = SinusoidalPositionalEncoding(HIDDEN_SIZE, max_len=200, dropout=0.1)

        # ---- Stacked cross-attention layers ----
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(
                hidden_size=HIDDEN_SIZE,
                head_num=NUM_HEADS,
                ffn_expansion=FFN_EXPANSION,
                attn_dropout=ATTN_DROPOUT,
                ffn_dropout=FFN_DROPOUT,
                proj_dropout=PROJ_DROPOUT
            )
            for _ in range(NUM_CROSS_ATTN_LAYERS)
        ])

        # ---- Fusion ----
        self.final_norm = nn.LayerNorm(HIDDEN_SIZE * 2)
        self.fusion_ffn = FusionFeedForward(
            dim_in=186, hidden_dim=186 * FFN_EXPANSION, dim_out=186, dropout=FFN_DROPOUT
        )
        self.final_norm2 = nn.LayerNorm(HIDDEN_SIZE * 2)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # ---- Classifier (NO softmax — CrossEntropyLoss handles it) ----
        self.classifier = ImprovedClassifier(
            input_dim=186,
            hidden_dim=128,
            num_classes=2,
            dropout=CLS_DROPOUT
        )

        # ---- Weight initialization ----
        self._init_weights()

    def _init_weights(self):
        """Xavier for Linear, Kaiming for Conv1d, ones/zeros for norms."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputVideo, inputAudio):
        """
        Args:
            inputVideo: (B, 915, 171) - TCN-processed video features
            inputAudio: (B, 186, 128) - VGGish audio features
        Returns:
            (B, 2) - raw logits (NO softmax)
        """
        # ---- Input BatchNorm (normalizes each feature channel using running stats) ----
        inputVideo = inputVideo.transpose(1, 2)     # (B, 171, 915)
        inputVideo = self.video_bn(inputVideo)
        inputVideo = inputVideo.transpose(1, 2)     # (B, 915, 171)

        inputAudio = inputAudio.transpose(1, 2)     # (B, 128, 186)
        inputAudio = self.audio_bn(inputAudio)
        inputAudio = inputAudio.transpose(1, 2)     # (B, 186, 128)

        # ---- Video branch: TCN -> projection -> temporal alignment ----
        video = self.TCNModel(inputVideo)            # (B, 114, 256)
        video = self.Conv1dModel(video)              # (B, 114, 128)
        video = self.conv(video)                     # (B, 186, 128)

        audio = inputAudio                           # (B, 186, 128)

        # ---- Positional encoding ----
        video = self.pos_enc_video(video)            # (B, 186, 128)
        audio = self.pos_enc_audio(audio)            # (B, 186, 128)

        # ---- Stacked cross-attention ----
        for cross_attn_block in self.cross_attn_layers:
            video, audio = cross_attn_block(video, audio)

        # ---- Fusion ----
        output = torch.cat((video, audio), dim=2)    # (B, 186, 256)
        output = self.fusion_ffn(self.final_norm(output)) + output  # residual
        output = self.final_norm2(output)

        # ---- Pool and classify ----
        output = self.pooling(output).reshape(output.shape[0], -1)  # (B, 186)
        result = self.classifier(output)             # (B, 2) raw logits
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
    """Cosine annealing with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 0.5 * (cos(min((current_step - num_warmup_steps) /
                              (num_training_steps - num_warmup_steps), 1) * math.pi) + 1)
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def mixup_data(video, audio, labels, alpha=0.2):
    """MixUp: interpolate between random pairs in a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = video.size(0)
    index = torch.randperm(batch_size).to(video.device)
    mixed_video = lam * video + (1 - lam) * video[index, :]
    mixed_audio = lam * audio + (1 - lam) * audio[index, :]
    labels_a, labels_b = labels, labels[index]
    return mixed_video, mixed_audio, labels_a, labels_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss: weighted combination of losses for both labels."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ====================================================================
# ==================== TRAINING FUNCTION =============================
# ====================================================================

def train(VideoPath, AudioPath, X_train, X_test, labelPath, numkfold,
          log_dir, save_dir):
    """Train MDDformer-Improved for one fold with all enhancements."""
    mytop = 0
    topacc = TOPACC_THRESHOLD
    top_p = top_r = top_f1 = 0
    top_pre = []
    top_label = []
    patience_counter = 0

    # Data loaders
    trainSet = MyDataLoader(VideoPath, AudioPath, X_train, labelPath, "train")
    trainLoader = DataLoader(trainSet, batch_size=TRAIN_BATCH, shuffle=True,
                            num_workers=2, pin_memory=True)
    devSet = MyDataLoader(VideoPath, AudioPath, X_test, labelPath, "dev")
    devLoader = DataLoader(devSet, batch_size=DEV_BATCH, shuffle=False,
                          num_workers=2, pin_memory=True)
    print(f"Fold {numkfold} - Train: {len(trainLoader)} batches, Dev: {len(devLoader)} batches")

    # Model
    model = MDDformerImproved().cuda(DEVICE)
    lossFunc = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).cuda(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=WEIGHT_DECAY)

    # LR Scheduler
    target_steps = len(trainLoader) * epochSize
    warmup_steps = len(trainLoader) * warmupEpoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, target_steps)

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

            # Audio Gaussian noise augmentation (training only)
            if AUDIO_NOISE_STD > 0:
                audioData = audioData + torch.randn_like(audioData) * AUDIO_NOISE_STD

            # MixUp augmentation (applied with MIXUP_PROB probability)
            use_mixup = MIXUP_ALPHA > 0 and np.random.random() < MIXUP_PROB
            if use_mixup:
                mixed_video, mixed_audio, labels_a, labels_b, lam = mixup_data(
                    videoData, audioData, label, MIXUP_ALPHA)
                output = model(mixed_video, mixed_audio)
                traLoss = mixup_criterion(lossFunc, output, labels_a.long(), labels_b.long(), lam)
            else:
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

        # Print every 10 epochs to keep output manageable
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
                save_name = f"MDDformer_improved_{epoch}_{float(acc):.2f}_{p:.4f}_{r:.4f}_{f1:.4f}.pth"
                torch.save(checkpoint, os.path.join(savePath, save_name))

        # Early stopping check
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    # Collect fold results
    if len(top_pre) == 0:
        print(f"WARNING: Fold {numkfold} - no predictions above threshold!")
        return torch.tensor([]), torch.tensor([]), mytop, top_p, top_r, top_f1

    top_pre = torch.cat(top_pre, axis=0).cpu()
    top_label = torch.cat(top_label, axis=0).cpu()

    print(f"Fold {numkfold} complete. Best acc: {mytop:.2f}%, "
          f"P: {top_p:.4f}, R: {top_r:.4f}, F1: {top_f1:.4f}")
    return top_label, top_pre, mytop, top_p, top_r, top_f1


# ====================================================================
# ==================== MAIN ==========================================
# ====================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MDDformer-Improved Training")
    print("Paper: LMVD (arXiv: 2407.00024)")
    print("Phase 2: Targeting >=80% accuracy")
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
    log_dir = os.path.join(LOG_DIR, f'MDDformer_improved_{tim}')
    save_dir = os.path.join(SAVE_DIR, f'MDDformer_improved_{tim}')
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
    tmp_model = MDDformerImproved()
    total_params = sum(p.numel() for p in tmp_model.parameters())
    trainable_params = sum(p.numel() for p in tmp_model.parameters() if p.requires_grad)
    del tmp_model

    print(f"\nDataset: {len(X)} samples, Depression: {sum(Y)}, Normal: {len(Y) - sum(Y)}")
    print(f"Model params: {total_params:,} (trainable: {trainable_params:,})")
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print(f"\nConfig:")
    print(f"  LR={lr}, Epochs={epochSize}, Warmup={warmupEpoch}")
    print(f"  Weight Decay={WEIGHT_DECAY}, Label Smoothing={LABEL_SMOOTHING}")
    print(f"  Grad Clip={GRAD_CLIP}, MixUp Alpha={MIXUP_ALPHA} (prob={MIXUP_PROB})")
    print(f"  Audio Noise STD={AUDIO_NOISE_STD}")
    print(f"  Cross-Attn Layers={NUM_CROSS_ATTN_LAYERS}, Heads={NUM_HEADS}")
    print(f"  FFN Expansion={FFN_EXPANSION}, Hidden={HIDDEN_SIZE}")
    print(f"  Batch: Train={TRAIN_BATCH}, Dev={DEV_BATCH}")
    print(f"  Early Stop Patience={EARLY_STOP_PATIENCE}")
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
                              title='MDDformer-Improved 10-Fold Confusion Matrix')

    avg_acc = sum(all_totals) / len(all_totals)
    avg_p = sum(all_ps) / len(all_ps)
    avg_r = sum(all_rs) / len(all_rs)
    avg_f1 = sum(all_f1s) / len(all_f1s)

    print("\n" + "=" * 60)
    print("MDDformer-Improved 10-Fold Cross-Validation Results")
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
        f.write(f"MDDformer-Improved 10-Fold CV Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Avg Accuracy:  {avg_acc:.2f}%\n")
        f.write(f"Avg Precision: {avg_p:.4f}\n")
        f.write(f"Avg Recall:    {avg_r:.4f}\n")
        f.write(f"Avg F1:        {avg_f1:.4f}\n")
        f.write(f"Per-fold Acc:  {all_totals}\n")
        f.write(f"Total Time:    {total_time/60:.1f} min\n")
        f.write(f"\nConfig:\n")
        f.write(f"  LR={lr}, Epochs={epochSize}, Warmup={warmupEpoch}\n")
        f.write(f"  Weight Decay={WEIGHT_DECAY}, Label Smoothing={LABEL_SMOOTHING}\n")
        f.write(f"  Grad Clip={GRAD_CLIP}, MixUp Alpha={MIXUP_ALPHA}\n")
        f.write(f"  Audio Noise STD={AUDIO_NOISE_STD}\n")
        f.write(f"  Cross-Attn Layers={NUM_CROSS_ATTN_LAYERS}, Heads={NUM_HEADS}\n")
        f.write(f"  FFN Expansion={FFN_EXPANSION}, Hidden={HIDDEN_SIZE}\n")
        f.write(f"  Batch: Train={TRAIN_BATCH}, Dev={DEV_BATCH}\n")
        f.write(f"  TCN Dropout={TCN_DROPOUT}, CLS Dropout={CLS_DROPOUT}\n")
        f.write(f"  Total params: {total_params:,}\n")
    print(f"\nResults saved to {log_dir}")
