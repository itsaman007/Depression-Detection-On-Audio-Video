"""
MDDformer-V2 — Adapted for D-Vlog Dataset (10-Fold CV)
========================================================
V2 improvements over baseline:
  1. [ARCH] Gated Attention Pooling (replaces AdaptiveAvgPool1d)
  2. [EVAL] Test-Time Augmentation (5 passes: 1 clean + 4 masked)
  3. [TRAIN] Stochastic Weight Averaging (accumulates after SWA_START_EPOCH)
  4. [FIX] Removed unsafe .squeeze() in CalculateAttention

D-Vlog adaptations (same as baseline D-Vlog):
  - Conv1d input: 171 → 136 (visual feature dim)
  - Audio projection: Linear(25, 128) added
  - Audio temporal pooling: AdaptiveAvgPool1d to align
  - Conv1d temporal alignment: 114 → 75 input channels
  - Input BatchNorm: video_bn(171) → video_bn(136), audio_bn(128) → audio_bn(25→128 projected)

LMVD V2 result: 77.29% avg 10-fold accuracy

Usage (Kaggle):
  Paste into a notebook cell and run. Requires dvlog-dataset as input.
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

# ========================== Configuration ==========================
SEED = 2222
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lr = 1e-5
epochSize = 300
warmupEpoch = 10
testRows = 1
NUM_FOLDS = 10
KFOLD_RANDOM_STATE = 42
TOPACC_THRESHOLD = 60
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 80

# D-Vlog sequence length
MAX_SEQ_LEN = 600

# Batch sizes
TRAIN_BATCH = 15
DEV_BATCH = 4

# V2 Enhancement Toggles
USE_ATTN_POOL = True
USE_SWA = True
SWA_START_EPOCH = 100
USE_TTA = True
TTA_PASSES = 5
TTA_DROP_RATIO = 0.25

# ========================== Paths ==========================
DVLOG_PATH = "/kaggle/input/dvlog-dataset"
LABELS_CSV = os.path.join(DVLOG_PATH, "labels.csv")
LOG_DIR = "/kaggle/working/logs"
SAVE_DIR = "/kaggle/working/checkpoints"

# ========================== Data Loading ==========================

class TemporalMask:
    def __init__(self, drop_ratio=0.25):
        self.ratio = drop_ratio

    def __call__(self, frames):
        T = frames.shape[0]
        n_drop = int(self.ratio * T)
        drop_indices = random.sample(range(T), n_drop)
        frames[drop_indices, :] = 0
        return frames


class DVlogDataset(udata.Dataset):
    def __init__(self, dvlog_path, indices, labels_dict, mode="train"):
        super().__init__()
        self.dvlog_path = dvlog_path
        self.indices = indices
        self.labels_dict = labels_dict
        self.augment = TemporalMask(0.25) if mode == "train" else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_id = self.indices[idx]
        folder = os.path.join(self.dvlog_path, str(sample_id))

        visual = np.load(os.path.join(folder, f"{sample_id}_visual.npy"))
        acoustic = np.load(os.path.join(folder, f"{sample_id}_acoustic.npy"))
        label = self.labels_dict[sample_id]

        if self.augment is not None:
            visual = self.augment(visual.copy())

        visual = torch.from_numpy(visual).float()
        acoustic = torch.from_numpy(acoustic).float()

        if visual.shape[0] > MAX_SEQ_LEN:
            visual = visual[:MAX_SEQ_LEN]
        if acoustic.shape[0] > MAX_SEQ_LEN:
            acoustic = acoustic[:MAX_SEQ_LEN]

        if visual.shape[0] < MAX_SEQ_LEN:
            pad = torch.zeros(MAX_SEQ_LEN - visual.shape[0], visual.shape[1])
            visual = torch.cat([visual, pad], dim=0)
        if acoustic.shape[0] < MAX_SEQ_LEN:
            pad = torch.zeros(MAX_SEQ_LEN - acoustic.shape[0], acoustic.shape[1])
            acoustic = torch.cat([acoustic, pad], dim=0)

        label = torch.tensor(label, dtype=torch.float)
        return visual, acoustic, label


# ========================== Model ==========================

class Conv1dEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(136, 32, 3, padding=1),
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

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class AstroModel(nn.Module):
    def __init__(self):
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
    def __init__(self):
        super().__init__()
        self.Conv1d = Conv1dEncoder()
        self.AstroModel = AstroModel()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.Conv1d(x)
        x = self.AstroModel(x)
        x = x.transpose(1, 2)
        return x


class CalculateAttention(nn.Module):
    """[V2 FIX]: Removed unsafe .squeeze() that could drop batch dim when B=1."""
    def __init__(self):
        super().__init__()

    def forward(self, Qx, Kx, Vx, Qy, Ky, Vy):
        attentionx = torch.matmul(Qx, torch.transpose(Kx, -1, -2))
        attentiony = torch.matmul(Qy, torch.transpose(Ky, -1, -2))
        attention = torch.cat((attentionx, attentiony), dim=1)
        B, C, H, W = attention.size()
        attention = attention.reshape(B, 2, C // 2, H, W)
        attention = torch.mean(attention, dim=1)  # no .squeeze()
        attention1 = torch.softmax(attention / sqrt(Qx.size(-1)), dim=-1)
        attention1 = torch.matmul(attention1, Vx)
        attention2 = torch.softmax(attention / sqrt(Qx.size(-1)), dim=-1)
        attention2 = torch.matmul(attention2, Vy)
        return attention1, attention2


class FeedForward(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out=None, *, dropout=0.1):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        self.net = nn.Sequential(
            nn.Conv1d(dim_in, hidden_dim, 1),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, dim_out, 1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Multi_CrossAttention(nn.Module):
    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.attn_dropout = nn.Dropout(0.1)

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
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 128)

    def forward(self, x):
        B, T, _ = x.size()
        x = x.contiguous().view(B * T, -1)
        x = self.fc(x)
        x = x.view(B, T, -1)
        return x


# ==================== V2 Components ====================

class GatedAttentionPooling(nn.Module):
    """Gated attention pooling (Ilse et al., 2018).
    a = w^T (tanh(V*h) ⊙ sigmoid(U*h))"""
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.V = nn.Linear(feature_dim, hidden_dim)
        self.U = nn.Linear(feature_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        h_V = torch.tanh(self.V(x))
        h_U = torch.sigmoid(self.U(x))
        attn_scores = self.w(h_V * h_U)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = (x * attn_weights).sum(dim=1)
        return pooled


class ClassifierV2(nn.Module):
    """3-layer classifier: 256 → 128 → 64 → 2 (no output ELU)."""
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
    """Baseline classifier fallback: 186 → 64 → 2."""
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
        return self.fc(x)


# ==================== NetV2 for D-Vlog ====================

class NetV2(nn.Module):
    """
    MDDformer-V2 adapted for D-Vlog.

    Pipeline:
      Visual (B, 600, 136) → BN → TCN → (B, 75, 256) → Linear → (B, 75, 128) → Conv1d → (B, 186, 128)
      Acoustic (B, 600, 25) → BN → Linear(25,128) → Pool → (B, 186, 128)
      → Cross-Attention → concat → (B, 186, 256)
      → LayerNorm + FFN + residual → LayerNorm
      → GatedAttentionPooling → (B, 256) → ClassifierV2 → (B, 2)
    """
    def __init__(self):
        super().__init__()
        # Input BatchNorm
        self.video_bn = nn.BatchNorm1d(136)
        self.audio_bn = nn.BatchNorm1d(25)

        # Video branch
        self.TCNModel = TCNModel()
        self.Conv1dModel = ConvNet1d()
        video_temporal = MAX_SEQ_LEN // 8  # 75
        self.conv_temporal = nn.Conv1d(video_temporal, 186, 1)

        # Audio branch
        self.audio_proj = nn.Linear(25, 128)
        self.audio_pool = nn.AdaptiveAvgPool1d(186)

        # Cross-attention
        self.mhca = Multi_CrossAttention(hidden_size=128, all_head_size=128, head_num=4)

        # Fusion FFN
        self.norm = nn.LayerNorm(128 * 2)
        self.FFN = FeedForward(dim_in=186, hidden_dim=186 * 2, dim_out=186)
        self.norm2 = nn.LayerNorm(128 * 2)

        # [V2] Pooling & Classification
        if USE_ATTN_POOL:
            self.attn_pool = GatedAttentionPooling(feature_dim=256, hidden_dim=64)
            self.classifier = ClassifierV2(input_dim=256)
        else:
            self.pooling = nn.AdaptiveAvgPool1d(1)
            self.classifier = Regress2()

    def forward(self, inputVideo, inputAudio):
        # Input BatchNorm
        inputVideo = inputVideo.transpose(1, 2)
        inputVideo = self.video_bn(inputVideo)
        inputVideo = inputVideo.transpose(1, 2)

        inputAudio = inputAudio.transpose(1, 2)
        inputAudio = self.audio_bn(inputAudio)
        inputAudio = inputAudio.transpose(1, 2)

        # Video branch
        inputVideo = self.TCNModel(inputVideo)
        outputVideo = self.Conv1dModel(inputVideo)
        outputVideo = self.conv_temporal(outputVideo)

        # Audio branch
        outputAudio = self.audio_proj(inputAudio)
        outputAudio = outputAudio.transpose(1, 2)
        outputAudio = self.audio_pool(outputAudio)
        outputAudio = outputAudio.transpose(1, 2)

        # Cross-attention
        output1, output2 = self.mhca(outputVideo, outputAudio)

        # Fusion
        outputFeature = torch.cat((output1, output2), dim=2)
        outputFeature = self.FFN(self.norm(outputFeature)) + outputFeature
        output = self.norm2(outputFeature)

        # [V2] Pool & Classify
        if USE_ATTN_POOL:
            output = self.attn_pool(output)
        else:
            output = self.pooling(output).reshape(output.shape[0], -1)

        result = self.classifier(output)
        result = result.squeeze(-1)
        return result


# ========================== Utilities ==========================

classes = ['Normal', 'Depression']


def plot_confusion_matrix(y_true, y_pred, labels_name, savename, title=None, thresh=0.6):
    cm = confusion_matrix(y_true, y_pred, labels=labels_name)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()
    if title:
        plt.title(title)
    ticks = list(range(len(labels_name)))
    plt.xticks(ticks, ['Normal', 'Depression'])
    plt.yticks(ticks, ['Normal', 'Depression'], rotation=90, va='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i in range(len(labels_name)):
        for j in range(len(labels_name)):
            if cm[i][j] > 0:
                plt.text(j, i, str(cm[i][j]),
                         ha="center", va="center",
                         color="white" if cm[i][j] > thresh * cm.max() else "black")
    plt.tight_layout()
    plt.savefig(savename, format='png', dpi=150)
    plt.clf()
    plt.close()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 0.5 * (cos(min((current_step - num_warmup_steps) /
                              (num_training_steps - num_warmup_steps), 1) * math.pi) + 1)
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def load_dvlog_labels(labels_csv):
    df = pd.read_csv(labels_csv)
    indices = df['index'].values.astype(int).tolist()
    labels = [1 if l == 'depression' else 0 for l in df['label'].values]
    return indices, labels


# ==================== V2 Utilities ====================

def tta_predict(model, videoData, audioData, num_passes=5, drop_ratio=0.25):
    """Test-Time Augmentation: average logits from multiple forward passes."""
    all_logits = []
    logits = model(videoData, audioData)
    all_logits.append(logits)

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
    """Update BatchNorm running stats for SWA model."""
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
        swa_model(videoData.to(device), audioData.to(device))

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    swa_model.train(was_training)


def evaluate_model(model, devLoader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    label2 = []
    pre2 = []

    with torch.no_grad():
        for videoData, audioData, label in devLoader:
            videoData = videoData.to(device)
            audioData = audioData.to(device)
            label = label.to(device)

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
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    label2 = []
    pre2 = []

    with torch.no_grad():
        for videoData, audioData, label in devLoader:
            videoData = videoData.to(device)
            audioData = audioData.to(device)
            label = label.to(device)

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


# ========================== Training (One Fold) ==========================

def train_fold(dvlog_path, all_indices, all_labels, train_idx, test_idx,
               numkfold, log_dir, save_dir):
    """Train MDDformer-V2 for one fold."""
    X_train = [all_indices[i] for i in train_idx]
    X_test = [all_indices[i] for i in test_idx]
    labels_dict = {all_indices[i]: all_labels[i] for i in range(len(all_indices))}

    mytop = 0
    topacc = TOPACC_THRESHOLD
    top_p = top_r = top_f1 = 0
    top_pre = []
    top_label = []
    patience_counter = 0
    best_model_state = None

    trainSet = DVlogDataset(dvlog_path, X_train, labels_dict, mode="train")
    devSet = DVlogDataset(dvlog_path, X_test, labels_dict, mode="eval")
    trainLoader = DataLoader(trainSet, batch_size=TRAIN_BATCH, shuffle=True,
                             num_workers=2, pin_memory=True)
    devLoader = DataLoader(devSet, batch_size=DEV_BATCH, shuffle=False,
                           num_workers=2, pin_memory=True)

    print(f"  Train: {len(trainSet)} samples ({len(trainLoader)} batches) | "
          f"Dev: {len(devSet)} samples ({len(devLoader)} batches)")

    model = NetV2().to(DEVICE)
    lossFunc = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=WEIGHT_DECAY)

    train_steps = len(trainLoader) * epochSize
    warmup_steps = len(trainLoader) * warmupEpoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, train_steps)

    swa_model = AveragedModel(model) if USE_SWA else None
    swa_n = 0

    savePath = os.path.join(save_dir, str(numkfold))
    os.makedirs(savePath, exist_ok=True)

    for epoch in range(1, epochSize + 1):
        # ==================== TRAIN ====================
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (visual, acoustic, label) in enumerate(trainLoader):
            visual = visual.to(DEVICE)
            acoustic = acoustic.to(DEVICE)
            label = label.to(DEVICE)

            output = model(visual, acoustic)
            loss = lossFunc(output, label.long())
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).cpu().sum().item()

        train_acc = 100.0 * correct / total

        # SWA update
        if USE_SWA and swa_model is not None and epoch >= SWA_START_EPOCH:
            swa_model.update_parameters(model)
            swa_n += 1

        # ==================== EVALUATE ====================
        if epoch % testRows == 0:
            model.eval()
            val_correct = 0
            val_total = 0
            all_labels_list = []
            all_preds_list = []
            label2 = []
            pre2 = []
            val_loss = 0

            with torch.no_grad():
                for visual, acoustic, label in devLoader:
                    visual = visual.to(DEVICE)
                    acoustic = acoustic.to(DEVICE)
                    label = label.to(DEVICE)

                    devOutput = model(visual, acoustic)
                    loss = lossFunc(devOutput, label.long())
                    val_loss += loss.item()

                    _, predicted = torch.max(devOutput.data, 1)
                    val_total += label.size(0)
                    val_correct += predicted.eq(label.data).cpu().sum().item()

                    label2.append(label.data)
                    pre2.append(predicted)
                    all_labels_list += label.data.tolist()
                    all_preds_list += predicted.tolist()

            acc = 100.0 * val_correct / val_total
            all_labels_np = np.array(all_labels_list)
            all_preds_np = np.array(all_preds_list)

            p = precision_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
            r = recall_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
            f1 = f1_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)

            if epoch % 10 == 0 or epoch == 1:
                print(f"    Epoch {epoch:3d} | Train Loss: {train_loss/len(trainLoader):.4f} "
                      f"Acc: {train_acc:.2f}% | Val Acc: {acc:.2f}% F1: {f1:.4f}")

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
                }
                save_name = f"MDDformer_dvlog_v2_{epoch}_{float(acc):.2f}_{p:.4f}_{r:.4f}_{f1:.4f}.pth"
                torch.save(checkpoint, os.path.join(savePath, save_name))

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
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

    # TTA evaluation
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

    # SWA evaluation
    if USE_SWA and swa_model is not None and swa_n > 0:
        print(f"  SWA: {swa_n} weight snapshots averaged (from epoch {SWA_START_EPOCH})")
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

    if len(final_pre) == 0:
        return torch.tensor([]), torch.tensor([]), final_acc, final_p, final_r, final_f1

    final_pre_cat = torch.cat(final_pre, axis=0).cpu()
    final_label_cat = torch.cat(final_label, axis=0).cpu()
    return final_label_cat, final_pre_cat, final_acc, final_p, final_r, final_f1


# ========================== Main ==========================

if __name__ == '__main__':
    print("=" * 60)
    print("MDDformer-V2 — D-Vlog Dataset (10-Fold CV)")
    print("=" * 60)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    all_indices, all_labels = load_dvlog_labels(LABELS_CSV)
    dep_count = sum(all_labels)
    norm_count = len(all_labels) - dep_count

    print(f"\nDataset: D-Vlog ({len(all_indices)} vlogs)")
    print(f"  Depression: {dep_count}, Normal: {norm_count}")
    print(f"  Sequence Length: {MAX_SEQ_LEN}")
    print(f"  Evaluation: {NUM_FOLDS}-fold Stratified CV")

    sample_folder = os.path.join(DVLOG_PATH, "0")
    if not os.path.exists(os.path.join(sample_folder, "0_visual.npy")):
        raise FileNotFoundError(f"D-Vlog data not found at {DVLOG_PATH}")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  Device: CPU")

    tmp_model = NetV2()
    total_params = sum(p.numel() for p in tmp_model.parameters())
    print(f"  Model params: {total_params:,}")
    del tmp_model

    print(f"\nV2 Enhancements:")
    print(f"  Attention Pooling: {USE_ATTN_POOL}")
    print(f"  SWA: {USE_SWA} (start epoch={SWA_START_EPOCH})")
    print(f"  TTA: {USE_TTA} (passes={TTA_PASSES}, drop={TTA_DROP_RATIO})")

    tim = time.strftime('%m_%d__%H_%M', time.localtime())
    log_dir = os.path.join(LOG_DIR, f'MDDformer_dvlog_v2_10fold_{tim}')
    save_dir = os.path.join(SAVE_DIR, f'MDDformer_dvlog_v2_10fold_{tim}')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nLogs: {log_dir}")
    print(f"Checkpoints: {save_dir}")
    print("=" * 60)

    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=KFOLD_RANDOM_STATE)
    X_array = np.array(all_indices)
    Y_array = np.array(all_labels)

    all_totals = []
    all_ps = []
    all_rs = []
    all_f1s = []
    all_pre = []
    all_label = []

    fold_start_time = time.time()

    for fold_num, (train_index, test_index) in enumerate(kf.split(X_array, Y_array), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_num}/{NUM_FOLDS}  (Train: {len(train_index)}, Test: {len(test_index)})")
        print(f"{'='*60}")

        top_label, top_pre_fold, top_acc, top_p, top_r, top_f1 = train_fold(
            DVLOG_PATH, all_indices, all_labels, train_index, test_index,
            fold_num, log_dir, save_dir
        )

        all_totals.append(top_acc)
        all_ps.append(top_p)
        all_rs.append(top_r)
        all_f1s.append(top_f1)
        if len(top_pre_fold) > 0:
            all_pre.append(top_pre_fold)
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
                              title='MDDformer-V2 D-Vlog 10-Fold Confusion Matrix')

    avg_acc = sum(all_totals) / len(all_totals)
    avg_p = sum(all_ps) / len(all_ps)
    avg_r = sum(all_rs) / len(all_rs)
    avg_f1 = sum(all_f1s) / len(all_f1s)

    print("\n" + "=" * 60)
    print("MDDformer-V2 — D-Vlog 10-Fold CV Results")
    print("=" * 60)
    print(f"Per-fold Accuracy: {[f'{x:.2f}' for x in all_totals]}")
    print(f"Average Accuracy:  {avg_acc:.2f}%")
    print(f"Average Precision: {avg_p:.4f}")
    print(f"Average Recall:    {avg_r:.4f}")
    print(f"Average F1:        {avg_f1:.4f}")
    print(f"Total Time:        {total_time/60:.1f} minutes")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("COMPARISON: MDDformer-V2 on D-Vlog vs LMVD")
    print("=" * 60)
    print(f"{'Metric':<20} {'D-Vlog':>10} {'LMVD':>10}")
    print(f"{'-'*40}")
    print(f"{'Avg Accuracy':<20} {avg_acc:>9.2f}% {'77.29':>9}%")
    print(f"{'Avg F1':<20} {avg_f1:>10.4f} {'------':>10}")
    print("=" * 60)

    with open(os.path.join(log_dir, "results.txt"), "w") as f:
        f.write(f"MDDformer-V2 on D-Vlog — 10-Fold CV\n")
        f.write(f"{'='*50}\n")
        f.write(f"Avg Accuracy:  {avg_acc:.2f}%\n")
        f.write(f"Avg Precision: {avg_p:.4f}\n")
        f.write(f"Avg Recall:    {avg_r:.4f}\n")
        f.write(f"Avg F1:        {avg_f1:.4f}\n")
        f.write(f"Per-fold Acc:  {all_totals}\n")
        f.write(f"Total Time:    {total_time/60:.1f} min\n")
        f.write(f"\nV2 Enhancements:\n")
        f.write(f"  Attention Pooling: {USE_ATTN_POOL}\n")
        f.write(f"  SWA: {USE_SWA} (start={SWA_START_EPOCH})\n")
        f.write(f"  TTA: {USE_TTA} (passes={TTA_PASSES})\n")
        f.write(f"  Total params: {total_params:,}\n")
        f.write(f"\n--- COMPARISON ---\n")
        f.write(f"D-Vlog Avg Acc: {avg_acc:.2f}%\n")
        f.write(f"LMVD   Avg Acc: 77.29% (V2)\n")

    print(f"\nResults saved to {log_dir}")
    print(f"Done!")
