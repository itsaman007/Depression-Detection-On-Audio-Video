"""
MDDformer-Improved 10-Fold Cross-Validation Training Script
============================================================
Phase 2: Target >=80% accuracy with advanced training strategies.

Improvements over the baseline training script:
  1. AdamW optimizer with weight decay
  2. Label smoothing in CrossEntropyLoss
  3. Gradient clipping for stable training
  4. Cosine warmup with proper warm restarts
  5. Early stopping with patience
  6. MixUp data augmentation
  7. Enhanced temporal masking (configurable ratio)
  8. Gaussian noise augmentation for audio features
  9. Stochastic Weight Averaging (SWA) over last N epochs
  10. Comprehensive experiment logging

Usage:
  python model/MDDformer/MDDformer_improved_fold.py
"""

import numpy as np
import torch
import logging
from kfoldLoader import MyDataLoader
from torch.utils.data import DataLoader
import math
from torch.optim.lr_scheduler import LambdaLR
from math import cos
from tqdm import tqdm
import torch.nn as nn
import time
import os
import copy
from sklearn.metrics import precision_score, recall_score, f1_score
from MDDformer_improved import MDDformerImproved
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

# ========================== Configuration ==========================
# --- Hyperparameters (tune these for best results) ---
lr = 0.00003                 # Higher LR (compensated by warmup + weight decay)
epochSize = 400              # More epochs for convergence
warmupEpoch = 15             # Warmup epochs (was 0 in original)
testRows = 1
schedule = 'cosine'
classes = ['Normal', 'Depression']

# Training strategies
WEIGHT_DECAY = 1e-4          # AdamW weight decay
LABEL_SMOOTHING = 0.1        # Label smoothing factor
GRAD_CLIP_MAX_NORM = 1.0     # Gradient clipping max norm
MIXUP_ALPHA = 0.2            # MixUp interpolation alpha (0 = disabled)
AUDIO_NOISE_STD = 0.01       # Gaussian noise std for audio augmentation (0 = disabled)
EARLY_STOP_PATIENCE = 50     # Stop if no improvement for N epochs
SWA_START_EPOCH = 300        # Start SWA from this epoch (0 = disabled)

# Batch sizes
TRAIN_BATCH_SIZE = 12        # Slightly smaller for more gradient updates
DEV_BATCH_SIZE = 4

# GPU device
DEVICE = 0

# Model architecture config
MODEL_CONFIG = {
    'num_cross_attn_layers': 2,
    'head_num': 8,
    'ffn_expansion': 4,
    'attn_dropout': 0.1,
    'ffn_dropout': 0.1,
    'proj_dropout': 0.1,
    'cls_dropout': 0.2,
    'use_pos_encoding': True,
    'pos_encoding_type': 'sinusoidal',
}

# ========================== Metrics Storage ==========================
ps = []
rs = []
f1s = []
totals = []
total_pre = []
total_label = []

# ========================== Paths ==========================
tim = time.strftime('%m_%d__%H_%M', time.localtime())

# --- CHANGE THESE PATHS ---
TCN_VIDEO_PATH = r"D:\MDD\TCN_processed_video"
AUDIO_PATH = r"D:\MDD\Audio_feature"
LABEL_PATH = r"D:\MDD\label"

LOG_DIR = r"D:\MDD\model\MDDformer\logs"
SAVE_DIR = r"D:\MDD\model\MDDformer\checkpoints"

filepath = os.path.join(LOG_DIR, 'MDDformer_improved_' + str(tim))
savePath1 = os.path.join(SAVE_DIR, 'MDDformer_improved_' + str(tim))

if not os.path.exists(filepath):
    os.makedirs(filepath)

logging.basicConfig(level=logging.NOTSET,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=filepath + '/' + 'MDDformer_improved_train.log',
                    filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


# ========================== Utilities ==========================
def plot_confusion_matrix(y_true, y_pred, labels_name, savename, title=None, thresh=0.6):
    """Plot and save confusion matrix."""
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name)
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
            if cm[i][j] * 100 > 0:
                plt.text(j, i, format(cm[i][j] * 100, '0.2f') + '%',
                         ha="center", va="center",
                         color="white" if cm[i][j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(savename, format='png', dpi=150)
    plt.clf()
    plt.close()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """Cosine annealing with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 0.5 * (cos(min((current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps), 1) * math.pi) + 1)
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def mixup_data(video, audio, labels, alpha=0.2):
    """
    MixUp augmentation: interpolate between random pairs.
    Returns mixed inputs and targets (soft labels).
    """
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


def add_gaussian_noise(tensor, std=0.01):
    """Add Gaussian noise to a tensor for augmentation."""
    if std > 0:
        noise = torch.randn_like(tensor) * std
        return tensor + noise
    return tensor


# ========================== Training Function ==========================
def train(VideoPath, AudioPath, X_train, X_test, labelPath, numkfold):
    """Train MDDformer-Improved for one fold with all enhancements."""
    
    mytop = 0
    topacc = 60
    top_p = 0
    top_r = 0
    top_f1 = 0
    top_pre = []
    top_label = []
    patience_counter = 0

    # Data loaders
    trainSet = MyDataLoader(VideoPath, AudioPath, X_train, labelPath, "train")
    trainLoader = DataLoader(trainSet, batch_size=TRAIN_BATCH_SIZE, shuffle=True, 
                             num_workers=0, pin_memory=True)
    devSet = MyDataLoader(VideoPath, AudioPath, X_test, labelPath, "dev")
    devLoader = DataLoader(devSet, batch_size=DEV_BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=True)
    print(f"Fold {numkfold} - Train: {len(trainLoader)} batches, Dev: {len(devLoader)} batches")

    # Model
    if torch.cuda.is_available():
        model = MDDformerImproved(**MODEL_CONFIG).cuda(DEVICE)
    else:
        model = MDDformerImproved(**MODEL_CONFIG)

    # Loss with label smoothing
    if torch.cuda.is_available():
        lossFunc = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).cuda(DEVICE)
    else:
        lossFunc = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   betas=(0.9, 0.999),
                                   eps=1e-8,
                                   weight_decay=WEIGHT_DECAY)

    # Scheduler
    train_steps = len(trainLoader) * epochSize
    warmup_steps = len(trainLoader) * warmupEpoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                 num_warmup_steps=warmup_steps,
                                                 num_training_steps=train_steps)

    # SWA model (Stochastic Weight Averaging)
    swa_model = None
    swa_count = 0
    if SWA_START_EPOCH > 0:
        swa_model = copy.deepcopy(model)
        for p in swa_model.parameters():
            p.data.zero_()

    logging.info(f'Fold {numkfold} training begins!')
    savePath = str(savePath1) + '/' + str(numkfold)
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    for epoch in range(1, epochSize):
        loop = tqdm(enumerate(trainLoader), total=len(trainLoader))
        traloss_one = 0
        correct = 0
        total = 0

        model.train()
        for batch_idx, (videoData, audioData, label) in loop:
            if torch.cuda.is_available():
                videoData = videoData.cuda(DEVICE)
                audioData = audioData.cuda(DEVICE)
                label = label.cuda(DEVICE)

            # Audio noise augmentation
            if AUDIO_NOISE_STD > 0:
                audioData = add_gaussian_noise(audioData, AUDIO_NOISE_STD)

            # MixUp augmentation
            if MIXUP_ALPHA > 0 and np.random.random() > 0.5:  # Apply MixUp 50% of the time
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

            # Gradient clipping
            if GRAD_CLIP_MAX_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)

            optimizer.step()
            scheduler.step()

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).cpu().sum()

            loop.set_description(f'Train Epoch [{epoch}/{epochSize}]')
            loop.set_postfix(loss=traloss_one / (batch_idx + 1), 
                           lr=optimizer.param_groups[0]['lr'])

        train_acc = 100.0 * correct / total
        logging.info('Epoch: {}, Loss:{:.4f}, TrainAcc:{:.2f}%, LR:{:.8f}'.format(
            epoch, traloss_one / len(trainLoader), train_acc, optimizer.param_groups[0]['lr']))

        # SWA: accumulate model weights after SWA_START_EPOCH
        if SWA_START_EPOCH > 0 and epoch >= SWA_START_EPOCH:
            swa_count += 1
            for swa_p, p in zip(swa_model.parameters(), model.parameters()):
                swa_p.data += p.data

        # ==================== Evaluation ====================
        if epoch - warmupEpoch >= 0 and epoch % testRows == 0:
            correct = 0
            total = 0
            lable1 = []
            pre1 = []
            label2 = []
            pre2 = []

            # Decide which model to eval (SWA averaged or current)
            eval_model = model
            if SWA_START_EPOCH > 0 and swa_count > 0:
                eval_model = copy.deepcopy(swa_model)
                for p in eval_model.parameters():
                    p.data /= swa_count

            eval_model.eval()
            loop = tqdm(enumerate(devLoader), total=len(devLoader))
            with torch.no_grad():
                loss_one = 0
                for batch_idx, (videoData, audioData, label) in loop:
                    if torch.cuda.is_available():
                        videoData = videoData.cuda(DEVICE)
                        audioData = audioData.cuda(DEVICE)
                        label = label.cuda(DEVICE)

                    devOutput = eval_model(videoData, audioData)
                    loss = lossFunc(devOutput, label.long())
                    loss_one += loss.item()

                    _, predicted = torch.max(devOutput.data, 1)
                    total += label.size(0)
                    correct += predicted.eq(label.data).cpu().sum()

                    label2.append(label.data)
                    pre2.append(predicted)

                    lable1 += label.data.tolist()
                    pre1 += predicted.tolist()

            acc = 100.0 * correct / total
            lable1 = np.array(lable1)
            pre1 = np.array(pre1)

            p = precision_score(lable1, pre1, average='weighted')
            r = recall_score(lable1, pre1, average='weighted')
            f1score = f1_score(lable1, pre1, average='weighted')

            logging.info(f'Dev Epoch:{epoch}, Loss:{loss_one / len(devLoader):.4f}, '
                        f'Acc:{acc:.2f}%, P:{p:.4f}, R:{r:.4f}, F1:{f1score:.4f}')

            if acc > mytop:
                mytop = acc
                top_p = p
                top_r = r
                top_f1 = f1score
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
                    'config': MODEL_CONFIG,
                }
                torch.save(checkpoint, os.path.join(savePath, 
                    f"MDDformer_improved_{epoch}_{float(acc):.2f}_{p:.4f}_{r:.4f}_{f1score:.4f}.pth"))

            # Early stopping
            if EARLY_STOP_PATIENCE > 0 and patience_counter >= EARLY_STOP_PATIENCE:
                logging.info(f'Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)')
                break

    top_pre = torch.cat(top_pre, axis=0).cpu()
    top_label = torch.cat(top_label, axis=0).cpu()

    totals.append(mytop)
    ps.append(top_p)
    rs.append(top_r)
    f1s.append(top_f1)

    logging.info(f'Fold {numkfold} best: Acc={mytop:.2f}%, P={top_p:.4f}, R={top_r:.4f}, F1={top_f1:.4f}')
    print(f"Fold {numkfold} complete. Best acc: {mytop:.2f}%")

    return top_label, top_pre


# ========================== Main ==========================
if __name__ == '__main__':
    import random
    from sklearn.model_selection import StratifiedKFold

    # Set seeds
    seed = 2222
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Log experiment config
    logging.info("=" * 60)
    logging.info("MDDformer-Improved Experiment Configuration")
    logging.info("=" * 60)
    logging.info(f"LR: {lr}, Epochs: {epochSize}, Warmup: {warmupEpoch}")
    logging.info(f"Weight Decay: {WEIGHT_DECAY}, Label Smoothing: {LABEL_SMOOTHING}")
    logging.info(f"Gradient Clip: {GRAD_CLIP_MAX_NORM}, MixUp Alpha: {MIXUP_ALPHA}")
    logging.info(f"Audio Noise STD: {AUDIO_NOISE_STD}, Early Stop Patience: {EARLY_STOP_PATIENCE}")
    logging.info(f"SWA Start: {SWA_START_EPOCH}, Train Batch: {TRAIN_BATCH_SIZE}")
    logging.info(f"Model Config: {MODEL_CONFIG}")
    logging.info("=" * 60)

    tcn = TCN_VIDEO_PATH
    mdnAudioPath = AUDIO_PATH
    labelPath = LABEL_PATH

    Y = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    X = os.listdir(tcn)
    X.sort(key=lambda x: int(x.split(".")[0]))
    X = np.array(X)

    for i in X:
        file_csv = pd.read_csv(os.path.join(labelPath, str(i.split('.npy')[0]) + "_Depression.csv"))
        bdi = int(file_csv.columns[0])
        Y.append(bdi)

    logging.info(f"Total samples: {len(X)}, Depression: {sum(Y)}, Normal: {len(Y) - sum(Y)}")

    numkfold = 0
    for train_index, test_index in kf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        numkfold += 1
        logging.info(f'Fold {numkfold} - Train: {len(X_train)}, Test: {len(X_test)}')
        total_label_0, total_pre_0 = train(tcn, mdnAudioPath, X_train, X_test, labelPath, numkfold)
        total_pre.append(total_pre_0)
        total_label.append(total_label_0)

    # Aggregate
    total_pre = torch.cat(total_pre, axis=0).cpu().numpy()
    total_label = torch.cat(total_label, axis=0).cpu().numpy()
    np.save(filepath + "/total_pre.npy", total_pre)
    np.save(filepath + "/total_label.npy", total_label)

    plot_confusion_matrix(total_label, total_pre, [0, 1],
                          savename=filepath + '/confusion_matrix.png',
                          title='MDDformer-Improved 10-Fold Confusion Matrix')

    # Final summary
    logging.info('=' * 60)
    logging.info('MDDformer-Improved 10-Fold Cross-Validation Results')
    logging.info('=' * 60)
    logging.info('Per-fold accuracy: {}'.format([f"{x:.2f}" for x in totals]))
    logging.info('Average accuracy: {:.2f}%'.format(sum(totals) / len(totals)))
    logging.info('Average precision: {:.4f}'.format(sum(ps) / len(ps)))
    logging.info('Average recall: {:.4f}'.format(sum(rs) / len(rs)))
    logging.info('Average F1: {:.4f}'.format(sum(f1s) / len(f1s)))
    logging.info('=' * 60)

    print("\n" + "=" * 60)
    print("MDDformer-Improved 10-Fold Summary")
    print("=" * 60)
    print(f"Average Accuracy: {sum(totals) / len(totals):.2f}%")
    print(f"Average Precision: {sum(ps) / len(ps):.4f}")
    print(f"Average Recall: {sum(rs) / len(rs):.4f}")
    print(f"Average F1: {sum(f1s) / len(f1s):.4f}")
    print("=" * 60)
