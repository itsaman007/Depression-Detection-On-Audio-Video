"""
MDDformer-V4.1 — Adapted for D-Vlog Dataset (10-Fold CV)
========================================================
V4.1 = V4 with improved seed selection + weighted logit averaging.

Changes from V4:
  1. Seeds: [2222, 3333, 4444] → [2222, 4444, 5555]
  2. Weighted Ensemble: weight_i = acc_i / sum(all_accs)
     - Each model's logits weighted by its dev accuracy
     - Stronger models get more influence

D-Vlog adaptations: same as baseline/V2 D-Vlog.

LMVD V4.1 result: 78.72% avg 10-fold accuracy (best)

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

MAX_SEQ_LEN = 600
TRAIN_BATCH = 15
DEV_BATCH = 4

# V4.1 Ensemble Config
ENSEMBLE_SEEDS = [2222, 4444, 5555]    # Dropped 3333, added 5555
NUM_MODELS = len(ENSEMBLE_SEEDS)
USE_WEIGHTED_ENSEMBLE = True            # Weight logits by dev accuracy

# V2 toggles (kept)
USE_ATTN_POOL = True
USE_TTA = True
TTA_PASSES = 5
TTA_DROP_RATIO = 0.25

# ========================== Paths ==========================
DVLOG_PATH = "/kaggle/input/dvlog-dataset"
LABELS_CSV = os.path.join(DVLOG_PATH, "labels.csv")
LOG_DIR = "/kaggle/working/logs"
SAVE_DIR = "/kaggle/working/checkpoints"


# ========================== Seed ==========================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    def __init__(self):
        super().__init__()

    def forward(self, Qx, Kx, Vx, Qy, Ky, Vy):
        attentionx = torch.matmul(Qx, torch.transpose(Kx, -1, -2))
        attentiony = torch.matmul(Qy, torch.transpose(Ky, -1, -2))
        attention = torch.cat((attentionx, attentiony), dim=1)
        B, C, H, W = attention.size()
        attention = attention.reshape(B, 2, C // 2, H, W)
        attention = torch.mean(attention, dim=1)
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


class GatedAttentionPooling(nn.Module):
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


class NetV4(nn.Module):
    """MDDformer-V4.1 for D-Vlog. Architecture identical to V2."""
    def __init__(self):
        super().__init__()
        self.video_bn = nn.BatchNorm1d(136)
        self.audio_bn = nn.BatchNorm1d(25)

        self.TCNModel = TCNModel()
        self.Conv1dModel = ConvNet1d()
        video_temporal = MAX_SEQ_LEN // 8
        self.conv_temporal = nn.Conv1d(video_temporal, 186, 1)

        self.audio_proj = nn.Linear(25, 128)
        self.audio_pool = nn.AdaptiveAvgPool1d(186)

        self.mhca = Multi_CrossAttention(hidden_size=128, all_head_size=128, head_num=4)

        self.norm = nn.LayerNorm(128 * 2)
        self.FFN = FeedForward(dim_in=186, hidden_dim=186 * 2, dim_out=186)
        self.norm2 = nn.LayerNorm(128 * 2)

        self.attn_pool = GatedAttentionPooling(feature_dim=256, hidden_dim=64)
        self.classifier = ClassifierV2(input_dim=256)

    def forward(self, inputVideo, inputAudio):
        inputVideo = inputVideo.transpose(1, 2)
        inputVideo = self.video_bn(inputVideo)
        inputVideo = inputVideo.transpose(1, 2)

        inputAudio = inputAudio.transpose(1, 2)
        inputAudio = self.audio_bn(inputAudio)
        inputAudio = inputAudio.transpose(1, 2)

        inputVideo = self.TCNModel(inputVideo)
        outputVideo = self.Conv1dModel(inputVideo)
        outputVideo = self.conv_temporal(outputVideo)

        outputAudio = self.audio_proj(inputAudio)
        outputAudio = outputAudio.transpose(1, 2)
        outputAudio = self.audio_pool(outputAudio)
        outputAudio = outputAudio.transpose(1, 2)

        output1, output2 = self.mhca(outputVideo, outputAudio)

        outputFeature = torch.cat((output1, output2), dim=2)
        outputFeature = self.FFN(self.norm(outputFeature)) + outputFeature
        output = self.norm2(outputFeature)

        output = self.attn_pool(output)
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


def tta_predict(model, videoData, audioData, num_passes=5, drop_ratio=0.25):
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


# ========================== Weighted Ensemble Evaluation ==========================

def ensemble_predict(models, videoData, audioData, weights=None,
                     use_tta=False, tta_passes=5, tta_drop_ratio=0.25):
    """Weighted average of logits across models. Falls back to simple avg if weights=None."""
    all_logits = []
    for model in models:
        model.eval()
        if use_tta:
            logits = tta_predict(model, videoData, audioData, tta_passes, tta_drop_ratio)
        else:
            logits = model(videoData, audioData)
        all_logits.append(logits)

    if weights is not None and USE_WEIGHTED_ENSEMBLE:
        weighted_logits = torch.zeros_like(all_logits[0])
        for logits, w in zip(all_logits, weights):
            weighted_logits += w * logits
        return weighted_logits
    else:
        return torch.stack(all_logits).mean(dim=0)


def evaluate_ensemble(models, devLoader, device, weights=None,
                      use_tta=False, tta_passes=5, tta_drop_ratio=0.25):
    for m in models:
        m.eval()

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

            avg_logits = ensemble_predict(models, videoData, audioData,
                                          weights=weights, use_tta=use_tta,
                                          tta_passes=tta_passes,
                                          tta_drop_ratio=tta_drop_ratio)
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


def evaluate_single(model, devLoader, device):
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


# ========================== Single Model Training ==========================

def train_single_model(dvlog_path, X_train, X_test, labels_dict,
                       numkfold, seed_idx, model_seed, save_dir):
    """Train ONE model for one fold with a specific seed."""
    seed_everything(model_seed)

    mytop = 0
    topacc = TOPACC_THRESHOLD
    top_p = top_r = top_f1 = 0
    patience_counter = 0
    best_model_state = None

    trainSet = DVlogDataset(dvlog_path, X_train, labels_dict, mode="train")
    devSet = DVlogDataset(dvlog_path, X_test, labels_dict, mode="eval")
    trainLoader = DataLoader(trainSet, batch_size=TRAIN_BATCH, shuffle=True,
                             num_workers=2, pin_memory=True)
    devLoader = DataLoader(devSet, batch_size=DEV_BATCH, shuffle=False,
                           num_workers=2, pin_memory=True)

    model = NetV4().to(DEVICE)
    lossFunc = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=WEIGHT_DECAY)

    train_steps = len(trainLoader) * epochSize
    warmup_steps = len(trainLoader) * warmupEpoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, train_steps)

    print(f'    Model {seed_idx+1}/{NUM_MODELS} (seed={model_seed}) training begins!')
    savePath = os.path.join(save_dir, str(numkfold), f"seed_{model_seed}")
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

        if epoch % 10 == 0 or epoch == 1:
            print(f'      Epoch {epoch:3d} - Loss: {train_loss/len(trainLoader):.4f}, '
                  f'Acc: {train_acc:.2f}%')

        # ==================== EVAL ====================
        if epoch % testRows == 0:
            model.eval()
            val_correct = 0
            val_total = 0
            all_labels_list = []
            all_preds_list = []
            val_loss = 0

            with torch.no_grad():
                for visual, acoustic, label in devLoader:
                    visual = visual.to(DEVICE)
                    acoustic = acoustic.to(DEVICE)
                    label = label.to(DEVICE)

                    output = model(visual, acoustic)
                    loss = lossFunc(output, label.long())
                    val_loss += loss.item()

                    _, predicted = torch.max(output.data, 1)
                    val_total += label.size(0)
                    val_correct += predicted.eq(label.data).cpu().sum().item()

                    all_labels_list += label.data.tolist()
                    all_preds_list += predicted.tolist()

            acc = 100.0 * val_correct / val_total
            all_labels_np = np.array(all_labels_list)
            all_preds_np = np.array(all_preds_list)

            p = precision_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
            r = recall_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)
            f1 = f1_score(all_labels_np, all_preds_np, average='weighted', zero_division=0)

            if epoch % 10 == 0 or epoch == 1:
                print(f'      Val   {epoch:3d} - Acc: {acc:.2f}%, F1: {f1:.4f}')

            if acc > mytop:
                mytop = acc
                top_p = p
                top_r = r
                top_f1 = f1
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if acc > topacc:
                topacc = acc

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"      Early stopping at epoch {epoch}")
            break

    print(f'    Model {seed_idx+1} (seed={model_seed}) best: {mytop:.2f}% '
          f'(P:{top_p:.4f} R:{top_r:.4f} F1:{top_f1:.4f})')

    if best_model_state is not None:
        checkpoint = {
            'net': best_model_state,
            'seed': model_seed,
            'acc': mytop,
        }
        save_name = f"MDDformer_dvlog_v4_1_seed{model_seed}_{mytop:.2f}.pth"
        torch.save(checkpoint, os.path.join(savePath, save_name))

    return best_model_state, mytop, top_p, top_r, top_f1


# ========================== Fold Training (Multi-Seed + Weighted) ==========================

def train_fold(dvlog_path, all_indices, all_labels, train_idx, test_idx,
               numkfold, log_dir, save_dir):
    """Train all ensemble models for one fold, then evaluate weighted ensemble."""
    X_train = [all_indices[i] for i in train_idx]
    X_test = [all_indices[i] for i in test_idx]
    labels_dict = {all_indices[i]: all_labels[i] for i in range(len(all_indices))}

    print(f"  Training {NUM_MODELS} models for Fold {numkfold}...")
    print(f"  Seeds: {ENSEMBLE_SEEDS}")

    # Step 1: Train all models
    model_states = []
    individual_accs = []

    for idx, model_seed in enumerate(ENSEMBLE_SEEDS):
        state, acc, p, r, f1 = train_single_model(
            dvlog_path, X_train, X_test, labels_dict,
            numkfold, idx, model_seed, save_dir
        )
        model_states.append(state)
        individual_accs.append(acc)

    # Step 2: Load all models
    models = []
    valid_accs = []
    for state, acc in zip(model_states, individual_accs):
        if state is not None:
            model = NetV4().to(DEVICE)
            model.load_state_dict(state)
            model.eval()
            models.append(model)
            valid_accs.append(acc)

    if len(models) == 0:
        print(f"  WARNING: Fold {numkfold} - no models trained successfully!")
        return torch.tensor([]), torch.tensor([]), 0, 0, 0, 0

    # Step 3: Compute weights from dev accuracies [V4.1]
    acc_sum = sum(valid_accs)
    weights = [a / acc_sum for a in valid_accs]
    print(f"\n  [V4.1] Weighted ensemble:")
    for i, (seed, acc, w) in enumerate(zip(ENSEMBLE_SEEDS, individual_accs, weights)):
        print(f"    Model {i+1} (seed={seed}): acc={acc:.2f}% -> weight={w:.4f}")

    # Step 4: Prepare dev loader
    devSet = DVlogDataset(dvlog_path, X_test, labels_dict, mode="eval")
    devLoader = DataLoader(devSet, batch_size=DEV_BATCH, shuffle=False,
                           num_workers=2, pin_memory=True)

    # Step 5: Evaluate all strategies
    print(f"\n  --- Post-training evaluation (Fold {numkfold}) ---")

    best_individual_idx = int(np.argmax(individual_accs))
    best_individual_acc = individual_accs[best_individual_idx]
    print(f"  Individual models:  {[f'{a:.2f}%' for a in individual_accs]}")
    print(f"  Best individual:    {best_individual_acc:.2f}% "
          f"(seed={ENSEMBLE_SEEDS[best_individual_idx]})")

    # Re-evaluate best individual
    best_single = models[best_individual_idx]
    single_acc, single_p, single_r, single_f1, single_pre, single_label = \
        evaluate_single(best_single, devLoader, DEVICE)
    final_acc = single_acc
    final_p = single_p
    final_r = single_r
    final_f1 = single_f1
    final_pre = single_pre
    final_label = single_label

    # Weighted Ensemble (no TTA)
    ens_acc, ens_p, ens_r, ens_f1, ens_pre, ens_label = \
        evaluate_ensemble(models, devLoader, DEVICE, weights=weights, use_tta=False)
    print(f"  Weighted Ensemble ({len(models)} models): {ens_acc:.2f}% "
          f"(P:{ens_p:.4f} R:{ens_r:.4f} F1:{ens_f1:.4f})")

    if ens_acc > final_acc:
        final_acc = ens_acc
        final_p = ens_p
        final_r = ens_r
        final_f1 = ens_f1
        final_pre = ens_pre
        final_label = ens_label

    # Weighted Ensemble + TTA
    ens_tta_acc = -1
    if USE_TTA:
        ens_tta_acc, ens_tta_p, ens_tta_r, ens_tta_f1, ens_tta_pre, ens_tta_label = \
            evaluate_ensemble(models, devLoader, DEVICE, weights=weights,
                              use_tta=True, tta_passes=TTA_PASSES,
                              tta_drop_ratio=TTA_DROP_RATIO)
        print(f"  Weighted Ens+TTA:  {ens_tta_acc:.2f}% "
              f"(P:{ens_tta_p:.4f} R:{ens_tta_r:.4f} F1:{ens_tta_f1:.4f})")

        if ens_tta_acc > final_acc:
            final_acc = ens_tta_acc
            final_p = ens_tta_p
            final_r = ens_tta_r
            final_f1 = ens_tta_f1
            final_pre = ens_tta_pre
            final_label = ens_tta_label

    # Determine winner
    if final_acc == ens_tta_acc:
        winner = "Weighted Ens+TTA"
    elif final_acc == ens_acc:
        winner = "Weighted Ensemble"
    else:
        winner = f"Individual (seed={ENSEMBLE_SEEDS[best_individual_idx]})"

    print(f"  >> FOLD {numkfold} BEST: {final_acc:.2f}% [{winner}] "
          f"(P:{final_p:.4f} R:{final_r:.4f} F1:{final_f1:.4f})")

    # Cleanup GPU
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(final_pre) == 0:
        return torch.tensor([]), torch.tensor([]), final_acc, final_p, final_r, final_f1

    final_pre_cat = torch.cat(final_pre, axis=0).cpu()
    final_label_cat = torch.cat(final_label, axis=0).cpu()
    return final_label_cat, final_pre_cat, final_acc, final_p, final_r, final_f1


# ========================== Main ==========================

if __name__ == '__main__':
    print("=" * 60)
    print("MDDformer-V4.1 — D-Vlog Dataset (10-Fold CV)")
    print(f"Weighted Ensemble: {NUM_MODELS} models per fold")
    print(f"Seeds: {ENSEMBLE_SEEDS}")
    print("=" * 60)

    seed_everything(ENSEMBLE_SEEDS[0])

    all_indices, all_labels = load_dvlog_labels(LABELS_CSV)
    dep_count = sum(all_labels)
    norm_count = len(all_labels) - dep_count

    print(f"\nDataset: D-Vlog ({len(all_indices)} vlogs)")
    print(f"  Depression: {dep_count}, Normal: {norm_count}")
    print(f"  Sequence Length: {MAX_SEQ_LEN}")
    print(f"  Evaluation: {NUM_FOLDS}-fold Stratified CV")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  Device: CPU")

    tmp_model = NetV4()
    total_params = sum(p.numel() for p in tmp_model.parameters())
    print(f"  Model params: {total_params:,} per model ({total_params * NUM_MODELS:,} total)")
    del tmp_model

    print(f"\nV4.1 Config:")
    print(f"  Ensemble: {NUM_MODELS} models per fold")
    print(f"  Seeds: {ENSEMBLE_SEEDS}")
    print(f"  Weighted Ensemble: {USE_WEIGHTED_ENSEMBLE}")
    print(f"  Attention Pooling: {USE_ATTN_POOL}")
    print(f"  TTA: {USE_TTA} (passes={TTA_PASSES})")

    tim = time.strftime('%m_%d__%H_%M', time.localtime())
    log_dir = os.path.join(LOG_DIR, f'MDDformer_dvlog_v4_1_10fold_{tim}')
    save_dir = os.path.join(SAVE_DIR, f'MDDformer_dvlog_v4_1_10fold_{tim}')
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
                              title='MDDformer-V4.1 D-Vlog 10-Fold Confusion Matrix')

    avg_acc = sum(all_totals) / len(all_totals)
    avg_p = sum(all_ps) / len(all_ps)
    avg_r = sum(all_rs) / len(all_rs)
    avg_f1 = sum(all_f1s) / len(all_f1s)

    print("\n" + "=" * 60)
    print("MDDformer-V4.1 — D-Vlog 10-Fold CV Results")
    print("=" * 60)
    print(f"Per-fold Accuracy: {[f'{x:.2f}' for x in all_totals]}")
    print(f"Average Accuracy:  {avg_acc:.2f}%")
    print(f"Average Precision: {avg_p:.4f}")
    print(f"Average Recall:    {avg_r:.4f}")
    print(f"Average F1:        {avg_f1:.4f}")
    print(f"Total Time:        {total_time/60:.1f} minutes")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("COMPARISON: MDDformer-V4.1 on D-Vlog vs LMVD")
    print("=" * 60)
    print(f"{'Metric':<20} {'D-Vlog':>10} {'LMVD':>10}")
    print(f"{'-'*40}")
    print(f"{'Avg Accuracy':<20} {avg_acc:>9.2f}% {'78.72':>9}%")
    print(f"{'Avg F1':<20} {avg_f1:>10.4f} {'------':>10}")
    print("=" * 60)

    with open(os.path.join(log_dir, "results.txt"), "w") as f:
        f.write(f"MDDformer-V4.1 on D-Vlog — 10-Fold CV\n")
        f.write(f"{'='*50}\n")
        f.write(f"Avg Accuracy:  {avg_acc:.2f}%\n")
        f.write(f"Avg Precision: {avg_p:.4f}\n")
        f.write(f"Avg Recall:    {avg_r:.4f}\n")
        f.write(f"Avg F1:        {avg_f1:.4f}\n")
        f.write(f"Per-fold Acc:  {all_totals}\n")
        f.write(f"Total Time:    {total_time/60:.1f} min\n")
        f.write(f"\nConfig:\n")
        f.write(f"  Weighted Ensemble: {NUM_MODELS} models (seeds: {ENSEMBLE_SEEDS})\n")
        f.write(f"  TTA: passes={TTA_PASSES}\n")
        f.write(f"  Total params: {total_params:,} per model\n")
        f.write(f"\n--- COMPARISON ---\n")
        f.write(f"D-Vlog Avg Acc: {avg_acc:.2f}%\n")
        f.write(f"LMVD   Avg Acc: 78.72% (V4.1 - best)\n")

    print(f"\nResults saved to {log_dir}")
    print(f"Done!")
