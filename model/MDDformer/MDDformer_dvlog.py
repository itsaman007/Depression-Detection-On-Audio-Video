"""
MDDformer Baseline — Adapted for D-Vlog Dataset (10-Fold CV)
==============================================================
Trains and tests the MDDformer baseline architecture on the D-Vlog dataset
using 10-fold stratified cross-validation, identical to LMVD evaluation
for direct comparison.

D-Vlog features:
  - Visual: (T, 136)  — 68 dlib facial landmarks × 2 (x,y)
  - Acoustic: (T, 25) — 25 OpenSMILE low-level descriptors
  Both at 1-second resolution, T varies per sample.

Architecture changes vs LMVD baseline:
  - Conv1d input: 171 → 136 (visual feature dim)
  - Audio projection: Linear(25, 128) added (acoustic 25 → 128)
  - Audio temporal pooling: AdaptiveAvgPool1d to align with video's post-CNN length
  - Conv1d temporal alignment: 114 → 75 input channels (600/8=75 vs 915/8≈114)
  - Everything else (TCN, cross-attention, FFN, classifier) is unchanged.

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
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import random
import logging
from math import sqrt, cos
import math
from sklearn.model_selection import StratifiedKFold

# ========================== Configuration ==========================
SEED = 2222
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lr = 1e-5
epochSize = 300
warmupEpoch = 0
testRows = 1
schedule = 'cosine'
classes = ['Normal', 'Depression']
NUM_FOLDS = 10
KFOLD_RANDOM_STATE = 42

# D-Vlog sequence length (padded/truncated). 600 divides cleanly by 8 for MaxPool.
MAX_SEQ_LEN = 600

# Batch sizes (same as baseline)
TRAIN_BATCH = 15
DEV_BATCH = 4

# ========================== Paths ==========================
# Kaggle paths (upload dvlog-dataset as a Kaggle dataset)
DVLOG_PATH = "/kaggle/input/dvlog-dataset"
LABELS_CSV = os.path.join(DVLOG_PATH, "labels.csv")

LOG_DIR = "/kaggle/working/logs"
SAVE_DIR = "/kaggle/working/checkpoints"

# ========================== Data Loading ==========================

class TemporalMask:
    """Randomly zeros out a fraction of temporal frames as data augmentation."""
    def __init__(self, drop_ratio=0.25):
        self.ratio = drop_ratio

    def __call__(self, frames):
        T = frames.shape[0]
        n_drop = int(self.ratio * T)
        drop_indices = random.sample(range(T), n_drop)
        frames[drop_indices, :] = 0
        return frames


class DVlogDataset(udata.Dataset):
    """
    Dataset loader for D-Vlog.

    Args:
        dvlog_path: Root path to dvlog-dataset (contains folders 0/, 1/, ..., 960/)
        indices: List of sample indices (ints) to include
        labels_dict: Dict mapping index -> label (0=normal, 1=depression)
        mode: "train" or "eval"
    """
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

        visual = np.load(os.path.join(folder, f"{sample_id}_visual.npy"))   # (T, 136)
        acoustic = np.load(os.path.join(folder, f"{sample_id}_acoustic.npy"))  # (T, 25)
        label = self.labels_dict[sample_id]

        # Training augmentation
        if self.augment is not None:
            visual = self.augment(visual.copy())

        visual = torch.from_numpy(visual).float()
        acoustic = torch.from_numpy(acoustic).float()

        # Truncate if longer than MAX_SEQ_LEN
        if visual.shape[0] > MAX_SEQ_LEN:
            visual = visual[:MAX_SEQ_LEN]
        if acoustic.shape[0] > MAX_SEQ_LEN:
            acoustic = acoustic[:MAX_SEQ_LEN]

        # Zero-pad if shorter than MAX_SEQ_LEN
        if visual.shape[0] < MAX_SEQ_LEN:
            pad = torch.zeros(MAX_SEQ_LEN - visual.shape[0], visual.shape[1])
            visual = torch.cat([visual, pad], dim=0)
        if acoustic.shape[0] < MAX_SEQ_LEN:
            pad = torch.zeros(MAX_SEQ_LEN - acoustic.shape[0], acoustic.shape[1])
            acoustic = torch.cat([acoustic, pad], dim=0)

        label = torch.tensor(label, dtype=torch.float)
        return visual, acoustic, label


# ========================== Model (Adapted for D-Vlog) ==========================

class Conv1dEncoder(nn.Module):
    """3-layer 1D CNN for visual feature extraction. Input: 136-dim (D-Vlog)."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(136, 32, 3, padding=1),  # 171 → 136 for D-Vlog
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
    """Dilated Temporal Convolutional Network with residual connections."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(128, 256, 1)
        self.dropout = nn.Dropout(0.2)

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
    """TCN visual feature encoder: Conv1d -> Dilated TCN."""
    def __init__(self):
        super().__init__()
        self.Conv1d = Conv1dEncoder()
        self.AstroModel = AstroModel()

    def forward(self, x):
        x = x.transpose(1, 2)   # (B, 136, T) for Conv1d
        x = self.Conv1d(x)       # (B, 128, T/8)
        x = self.AstroModel(x)   # (B, 256, T/8)
        x = x.transpose(1, 2)   # (B, T/8, 256)
        return x


class CalculateAttention(nn.Module):
    """Cross-attention computation for bidirectional modality fusion."""
    def __init__(self):
        super().__init__()

    def forward(self, Qx, Kx, Vx, Qy, Ky, Vy):
        attentionx = torch.matmul(Qx, torch.transpose(Kx, -1, -2))
        attentiony = torch.matmul(Qy, torch.transpose(Ky, -1, -2))
        attention = torch.cat((attentionx, attentiony), dim=1)
        B, C, H, W = attention.size()
        attention = attention.reshape(B, 2, C // 2, H, W)
        attention = torch.mean(attention, dim=1).squeeze()
        attention1 = torch.softmax(attention / sqrt(Qx.size(-1)), dim=-1)
        attention1 = torch.matmul(attention1, Vx)
        attention2 = torch.softmax(attention / sqrt(Qx.size(-1)), dim=-1)
        attention2 = torch.matmul(attention2, Vy)
        return attention1, attention2


class FeedForward(nn.Module):
    """Position-wise feed-forward with Conv1d."""
    def __init__(self, dim_in, hidden_dim, dim_out=None, *, dropout=0.0):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        self.net = nn.Sequential(
            nn.Conv1d(dim_in, hidden_dim, 1),
            nn.ELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv1d(hidden_dim, dim_out, 1),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class Multi_CrossAttention(nn.Module):
    """Multi-head bidirectional cross-attention."""
    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num

        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)

    def forward(self, x, y):
        batch_size = x.size(0)
        q_sx = self.linear_q(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_sx = self.linear_k(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_sx = self.linear_v(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        q_sy = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_sy = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_sy = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        attention1, attention2 = CalculateAttention()(q_sx, k_sx, v_sx, q_sy, k_sy, v_sy)
        attention1 = attention1.transpose(1, 2).contiguous().view(batch_size, -1, self.all_head_size) + x
        attention2 = attention2.transpose(1, 2).contiguous().view(batch_size, -1, self.all_head_size) + y
        return attention1, attention2


class ConvNet1d(nn.Module):
    """Linear projection 256 -> 128."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 128)

    def forward(self, x):
        B, T, _ = x.size()
        x = x.contiguous().view(B * T, -1)
        x = self.fc(x)
        x = x.view(B, T, -1)
        return x


class Regress2(nn.Module):
    """Classification head: temporal_dim -> 64 -> 2."""
    def __init__(self, input_dim=186):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 2),
            nn.ELU()
        )

    def forward(self, x):
        return self.fc(x)


class Net(nn.Module):
    """
    MDDformer adapted for D-Vlog.

    Pipeline:
      Visual (B, 600, 136) → TCN → (B, 75, 256) → Linear → (B, 75, 128) → Conv1d → (B, 186, 128)
      Acoustic (B, 600, 25) → Linear(25,128) → Pool temporal → (B, 186, 128)
      → Cross-Attention → concat → (B, 186, 256)
      → LayerNorm + FFN + residual → LayerNorm
      → AdaptiveAvgPool1d → (B, 186) → FC → (B, 2)
    """
    def __init__(self):
        super().__init__()
        # Video branch
        self.TCNModel = TCNModel()
        self.Conv1dModel = ConvNet1d()

        # Temporal alignment: 75 (=600/8) → 186
        video_temporal = MAX_SEQ_LEN // 8  # 75
        self.conv_temporal = nn.Conv1d(in_channels=video_temporal, out_channels=186,
                                       kernel_size=1, padding=0, stride=1)

        # Audio branch: project 25 → 128 features, then pool to 186 time steps
        self.audio_proj = nn.Linear(25, 128)
        self.audio_pool = nn.AdaptiveAvgPool1d(186)

        # Cross-attention
        self.mhca = Multi_CrossAttention(hidden_size=128, all_head_size=128, head_num=4)

        # Feed-forward
        self.norm = nn.LayerNorm(128 * 2)
        self.FFN = FeedForward(dim_in=186, hidden_dim=186 * 2, dim_out=186)
        self.norm2 = nn.LayerNorm(128 * 2)

        # Pooling and classifier
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.Regress = Regress2(input_dim=186)

    def forward(self, inputVideo, inputAudio):
        # Video branch: TCN → project → temporal align
        inputVideo = self.TCNModel(inputVideo)            # (B, 75, 256)
        outputVideo = self.Conv1dModel(inputVideo)         # (B, 75, 128)
        outputVideo = self.conv_temporal(outputVideo)      # (B, 186, 128)

        # Audio branch: project features → pool temporal
        outputAudio = self.audio_proj(inputAudio)          # (B, 600, 128)
        outputAudio = outputAudio.transpose(1, 2)          # (B, 128, 600)
        outputAudio = self.audio_pool(outputAudio)         # (B, 128, 186)
        outputAudio = outputAudio.transpose(1, 2)          # (B, 186, 128)

        # Cross-attention fusion
        output1, output2 = self.mhca(outputVideo, outputAudio)

        # Concatenate attended features
        outputFeature = torch.cat((output1, output2), dim=2)  # (B, 186, 256)

        # Feed-forward + residual
        outputFeature = self.FFN(self.norm(outputFeature)) + outputFeature
        output = self.norm2(outputFeature)

        # Pool and classify
        output = self.pooling(output).reshape(output.shape[0], -1)  # (B, 186)
        result = self.Regress(output)
        result = result.squeeze(-1)
        return result


# ========================== Utilities ==========================

def plot_confusion_matrix(y_true, y_pred, labels_name, savename, title=None, thresh=0.6):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name)
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
    """Load D-Vlog labels.csv and return all indices + labels arrays."""
    df = pd.read_csv(labels_csv)
    indices = df['index'].values.astype(int).tolist()
    labels = [1 if l == 'depression' else 0 for l in df['label'].values]
    return indices, labels


# ========================== Training (One Fold) ==========================

def train_fold(dvlog_path, all_indices, all_labels, train_idx, test_idx,
               numkfold, log_dir, save_dir):
    """Train MDDformer for one fold of 10-fold CV."""

    X_train = [all_indices[i] for i in train_idx]
    X_test = [all_indices[i] for i in test_idx]
    labels_dict = {all_indices[i]: all_labels[i] for i in range(len(all_indices))}

    # Data loaders
    trainSet = DVlogDataset(dvlog_path, X_train, labels_dict, mode="train")
    devSet = DVlogDataset(dvlog_path, X_test, labels_dict, mode="eval")
    trainLoader = DataLoader(trainSet, batch_size=TRAIN_BATCH, shuffle=True,
                             num_workers=2, pin_memory=True)
    devLoader = DataLoader(devSet, batch_size=DEV_BATCH, shuffle=False,
                           num_workers=2, pin_memory=True)

    print(f"  Train: {len(trainSet)} samples ({len(trainLoader)} batches) | "
          f"Dev: {len(devSet)} samples ({len(devLoader)} batches)")

    # Model
    model = Net().to(DEVICE)

    # Loss, optimizer, scheduler
    lossFunc = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0, amsgrad=False)

    train_steps = len(trainLoader) * epochSize
    warmup_steps = len(trainLoader) * warmupEpoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, train_steps)

    savePath = os.path.join(save_dir, str(numkfold))
    os.makedirs(savePath, exist_ok=True)

    mytop = 0
    topacc = 60
    top_p = 0
    top_r = 0
    top_f1 = 0
    top_pre = []
    top_label = []

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
            optimizer.step()
            scheduler.step()

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).cpu().sum().item()

        train_acc = 100.0 * correct / total

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

            if acc > topacc:
                topacc = acc
                checkpoint = {
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }
                save_name = f"MDDformer_dvlog_{epoch}_{float(acc):.2f}_{p:.4f}_{r:.4f}_{f1:.4f}.pth"
                torch.save(checkpoint, os.path.join(savePath, save_name))

    print(f"  >> Fold {numkfold} BEST: {mytop:.2f}% (P:{top_p:.4f} R:{top_r:.4f} F1:{top_f1:.4f})")

    if len(top_pre) == 0:
        return torch.tensor([]), torch.tensor([]), mytop, top_p, top_r, top_f1

    top_pre = torch.cat(top_pre, axis=0).cpu()
    top_label = torch.cat(top_label, axis=0).cpu()
    return top_label, top_pre, mytop, top_p, top_r, top_f1


# ========================== Main ==========================

if __name__ == '__main__':
    print("=" * 60)
    print("MDDformer Baseline — D-Vlog Dataset (10-Fold CV)")
    print("=" * 60)

    # Seed everything
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load all indices and labels
    all_indices, all_labels = load_dvlog_labels(LABELS_CSV)
    dep_count = sum(all_labels)
    norm_count = len(all_labels) - dep_count

    print(f"\nDataset: D-Vlog ({len(all_indices)} vlogs)")
    print(f"  Depression: {dep_count}, Normal: {norm_count}")
    print(f"  Sequence Length: {MAX_SEQ_LEN}")
    print(f"  Visual: 136-dim, Acoustic: 25-dim")
    print(f"  Evaluation: {NUM_FOLDS}-fold Stratified CV (random_state={KFOLD_RANDOM_STATE})")

    # Verify data exists
    sample_folder = os.path.join(DVLOG_PATH, "0")
    if not os.path.exists(os.path.join(sample_folder, "0_visual.npy")):
        raise FileNotFoundError(f"D-Vlog data not found at {DVLOG_PATH}")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  Device: CPU")

    # Model info
    tmp_model = Net()
    total_params = sum(p.numel() for p in tmp_model.parameters())
    print(f"  Model params: {total_params:,}")
    del tmp_model

    # Timestamp
    tim = time.strftime('%m_%d__%H_%M', time.localtime())
    log_dir = os.path.join(LOG_DIR, f'MDDformer_dvlog_10fold_{tim}')
    save_dir = os.path.join(SAVE_DIR, f'MDDformer_dvlog_10fold_{tim}')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nLogs: {log_dir}")
    print(f"Checkpoints: {save_dir}")
    print("=" * 60)

    # 10-Fold Stratified CV
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
                              title='MDDformer D-Vlog 10-Fold Confusion Matrix')

    avg_acc = sum(all_totals) / len(all_totals)
    avg_p = sum(all_ps) / len(all_ps)
    avg_r = sum(all_rs) / len(all_rs)
    avg_f1 = sum(all_f1s) / len(all_f1s)

    print("\n" + "=" * 60)
    print("MDDformer Baseline — D-Vlog 10-Fold CV Results")
    print("=" * 60)
    print(f"Per-fold Accuracy: {[f'{x:.2f}' for x in all_totals]}")
    print(f"Average Accuracy:  {avg_acc:.2f}%")
    print(f"Average Precision: {avg_p:.4f}")
    print(f"Average Recall:    {avg_r:.4f}")
    print(f"Average F1:        {avg_f1:.4f}")
    print(f"Total Time:        {total_time/60:.1f} minutes")
    print("=" * 60)

    # Comparison table
    print("\n" + "=" * 60)
    print("COMPARISON: MDDformer on D-Vlog vs LMVD")
    print("=" * 60)
    print(f"{'Metric':<20} {'D-Vlog':>10} {'LMVD':>10}")
    print(f"{'-'*40}")
    print(f"{'Dataset size':<20} {'961':>10} {'1823':>10}")
    print(f"{'Visual features':<20} {'136-dim':>10} {'171-dim':>10}")
    print(f"{'Audio features':<20} {'25-dim':>10} {'128-dim':>10}")
    print(f"{'Avg Accuracy':<20} {avg_acc:>9.2f}% {'74.39':>9}%")
    print(f"{'Avg Precision':<20} {avg_p:>10.4f} {'------':>10}")
    print(f"{'Avg Recall':<20} {avg_r:>10.4f} {'------':>10}")
    print(f"{'Avg F1':<20} {avg_f1:>10.4f} {'------':>10}")
    print("=" * 60)

    # Save results
    with open(os.path.join(log_dir, "results.txt"), "w") as f:
        f.write(f"MDDformer Baseline on D-Vlog — 10-Fold CV\n")
        f.write(f"{'='*50}\n")
        f.write(f"Avg Accuracy:  {avg_acc:.2f}%\n")
        f.write(f"Avg Precision: {avg_p:.4f}\n")
        f.write(f"Avg Recall:    {avg_r:.4f}\n")
        f.write(f"Avg F1:        {avg_f1:.4f}\n")
        f.write(f"Per-fold Acc:  {all_totals}\n")
        f.write(f"Per-fold P:    {all_ps}\n")
        f.write(f"Per-fold R:    {all_rs}\n")
        f.write(f"Per-fold F1:   {all_f1s}\n")
        f.write(f"Total Time:    {total_time/60:.1f} min\n")
        f.write(f"\nConfig:\n")
        f.write(f"  LR={lr}, Epochs={epochSize}, Seq Len={MAX_SEQ_LEN}\n")
        f.write(f"  Batch: Train={TRAIN_BATCH}, Dev={DEV_BATCH}\n")
        f.write(f"  Folds={NUM_FOLDS}, Seed={SEED}, KFold RS={KFOLD_RANDOM_STATE}\n")
        f.write(f"  Model params: {total_params:,}\n")
        f.write(f"\n--- COMPARISON ---\n")
        f.write(f"D-Vlog Avg Acc: {avg_acc:.2f}%\n")
        f.write(f"LMVD   Avg Acc: 74.39% (baseline)\n")

    print(f"\nResults saved to {log_dir}")
    print(f"Done!")
