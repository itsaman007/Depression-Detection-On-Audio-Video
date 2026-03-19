"""
MDDformer 10-Fold Cross-Validation Training Script
===================================================
Phase 1: Reproduce the paper's reported accuracy (~76.88%).

This script follows the exact same conventions as the other model training 
scripts (BiLSTM, ViT, SEResnet, Xception) in the LMVD codebase:
  - 10-fold stratified cross-validation (random_state=42)
  - Global seed = 2222
  - Adam optimizer, lr=1e-5, cosine warmup schedule
  - CrossEntropyLoss
  - Train batch=15, Dev batch=4
  - Weighted precision/recall/F1 metrics
  - Best model checkpointing when acc > 60%

Usage:
  python model/MDDformer/MDDformerfold.py
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
from sklearn.metrics import precision_score, recall_score, f1_score
from MDDformermodel import Net
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

# ========================== Configuration ==========================
lr = 0.00001
epochSize = 300
warmupEpoch = 0
testRows = 1          # Evaluate every N epochs
schedule = 'cosine'
classes = ['Normal', 'Depression']

# GPU device (change to 0 for single-GPU setups)
DEVICE = 0

# ========================== Metrics Storage ==========================
ps = []
rs = []
f1s = []
totals = []

total_pre = []
total_label = []

# ========================== Paths ==========================
tim = time.strftime('%m_%d__%H_%M', time.localtime())

# --- CHANGE THESE PATHS TO YOUR LOCAL SETUP ---
# Path to TCN-processed video features (.npy files, shape 915x171)
TCN_VIDEO_PATH = r"D:\MDD\TCN_processed_video"

# Path to audio features (.npy files, shape Tx128) 
AUDIO_PATH = r"D:\MDD\Audio_feature"

# Path to label CSV files ({id}_Depression.csv)
LABEL_PATH = r"D:\MDD\label"

# Log and checkpoint paths
LOG_DIR = r"D:\MDD\model\MDDformer\logs"
SAVE_DIR = r"D:\MDD\model\MDDformer\checkpoints"

filepath = os.path.join(LOG_DIR, 'MDDformer_' + str(tim))
savePath1 = os.path.join(SAVE_DIR, 'MDDformer_' + str(tim))

if not os.path.exists(filepath):
    os.makedirs(filepath)

logging.basicConfig(level=logging.NOTSET,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=filepath + '/' + 'MDDformer_train.log',
                    filemode='w')

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


# ========================== Utilities ==========================
def plot_confusion_matrix(y_true, y_pred, labels_name, savename, title=None, thresh=0.6, axis_labels=None):
    """Plot and save confusion matrix."""
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()

    if title is not None:
        plt.title(title)

    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = classes
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


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """Linear decay with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


# ========================== Training Function ==========================
def train(VideoPath, AudioPath, X_train, X_test, labelPath, numkfold):
    """
    Train MDDformer for one fold.
    
    Args:
        VideoPath: Path to TCN-processed video features
        AudioPath: Path to audio features
        X_train: Array of training .npy filenames
        X_test: Array of test .npy filenames
        labelPath: Path to label CSV directory
        numkfold: Current fold number
    
    Returns:
        top_label, top_pre: Best predictions for this fold
    """
    mytop = 0
    topacc = 60   # Minimum accuracy threshold for saving checkpoints
    top_p = 0
    top_r = 0
    top_f1 = 0
    top_pre = []
    top_label = []

    # Data loaders
    trainSet = MyDataLoader(VideoPath, AudioPath, X_train, labelPath, "train")
    trainLoader = DataLoader(trainSet, batch_size=15, shuffle=True)
    devSet = MyDataLoader(VideoPath, AudioPath, X_test, labelPath, "dev")
    devLoader = DataLoader(devSet, batch_size=4, shuffle=False)
    print(f"Fold {numkfold} - trainLoader: {len(trainLoader)} batches, devLoader: {len(devLoader)} batches")

    # Model
    if torch.cuda.is_available():
        model = Net().cuda(DEVICE)
    else:
        model = Net()

    # Loss function
    if torch.cuda.is_available():
        lossFunc = nn.CrossEntropyLoss().cuda(DEVICE)
    else:
        lossFunc = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 weight_decay=0,
                                 amsgrad=False)

    # Learning rate scheduler
    train_steps = len(trainLoader) * epochSize
    warmup_steps = len(trainLoader) * warmupEpoch
    target_steps = len(trainLoader) * epochSize

    if schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=target_steps)

    logging.info('The {} fold training begins!'.format(numkfold))
    savePath = str(savePath1) + '/' + str(numkfold)
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    for epoch in range(1, epochSize):
        loop = tqdm(enumerate(trainLoader), total=len(trainLoader))
        traloss_one = 0
        correct = 0
        total = 0
        lable1 = []
        pre1 = []

        model.train()
        for batch_idx, (videoData, audioData, label) in loop:
            if torch.cuda.is_available():
                videoData, audioData, label = videoData.cuda(DEVICE), audioData.cuda(DEVICE), label.cuda(DEVICE)

            output = model(videoData, audioData)
            traLoss = lossFunc(output, label.long())
            traloss_one += traLoss.item()

            optimizer.zero_grad()
            traLoss.backward()
            optimizer.step()
            scheduler.step()

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data).cpu().sum()

            loop.set_description(f'Train Epoch [{epoch}/{epochSize}]')
            loop.set_postfix(loss=traloss_one / (batch_idx + 1))

        train_acc = 100.0 * correct / total
        logging.info('EpochSize: {}, Train batch: {}, Loss:{:.4f}, Acc:{:.2f}%'.format(
            epoch, batch_idx + 1, traloss_one / len(trainLoader), train_acc))

        # ==================== Evaluation ====================
        if epoch - warmupEpoch >= 0 and epoch % testRows == 0:
            train_num = 0
            correct = 0
            total = 0
            label2 = []
            pre2 = []

            model.eval()
            loop = tqdm(enumerate(devLoader), total=len(devLoader))
            with torch.no_grad():
                loss_one = 0
                for batch_idx, (videoData, audioData, label) in loop:
                    if torch.cuda.is_available():
                        videoData, audioData, label = videoData.cuda(DEVICE), audioData.cuda(DEVICE), label.cuda(DEVICE)

                    devOutput = model(videoData, audioData)
                    loss = lossFunc(devOutput, label.long())
                    loss_one += loss.item()
                    train_num += label.size(0)

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

            p = precision_score(lable1, pre1, average='weighted', zero_division=0)
            r = recall_score(lable1, pre1, average='weighted', zero_division=0)
            f1score = f1_score(lable1, pre1, average='weighted', zero_division=0)
            logging.info('precision:{:.4f}'.format(p))
            logging.info('recall:{:.4f}'.format(r))
            logging.info('f1:{:.4f}'.format(f1score))

            logging.info('Dev epoch:{}, Loss:{:.4f}, Acc:{:.2f}%'.format(
                epoch, loss_one / len(devLoader), acc))
            print('Dev epoch:{}, Loss:{:.4f}, Acc:{:.2f}%'.format(
                epoch, loss_one / len(devLoader), acc))

            if acc > mytop:
                mytop = max(acc, mytop)
                top_p = p
                top_r = r
                top_f1 = f1score
                top_pre = pre2
                top_label = label2

            if acc > topacc:
                topacc = max(acc, topacc)
                checkpoint = {
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'scheduler': scheduler.state_dict()
                }
                torch.save(checkpoint, savePath + '/' + "MDDformer" + '_' +
                           str(epoch) + '_' + str(float(acc)) + '_' +
                           str(p) + '_' + str(r) + '_' + str(f1score) + '.pth')

    top_pre = torch.cat(top_pre, axis=0).cpu()
    top_label = torch.cat(top_label, axis=0).cpu()

    totals.append(mytop)
    ps.append(top_p)
    rs.append(top_r)
    f1s.append(top_f1)

    logging.info('Fold {} best accuracy: {:.2f}%'.format(numkfold, mytop))
    logging.info('')

    print(f"Fold {numkfold} training complete. Best acc: {mytop:.2f}%")

    return top_label, top_pre


# ========================== Main ==========================
if __name__ == '__main__':
    import random
    from sklearn.model_selection import StratifiedKFold

    # Set seeds for reproducibility
    seed = 2222
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Paths (configured above)
    tcn = TCN_VIDEO_PATH
    mdnAudioPath = AUDIO_PATH
    labelPath = LABEL_PATH

    # Build sample list and labels for stratified K-fold
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
        logging.info('Fold {} - Train set: {} samples, Test set: {} samples'.format(
            numkfold, len(X_train), len(X_test)))
        total_label_0, total_pre_0 = train(tcn, mdnAudioPath, X_train, X_test, labelPath, numkfold)
        total_pre.append(total_pre_0)
        total_label.append(total_label_0)

    # Aggregate all fold predictions
    total_pre = torch.cat(total_pre, axis=0).cpu().numpy()
    total_label = torch.cat(total_label, axis=0).cpu().numpy()
    np.save(filepath + "/total_pre.npy", total_pre)
    np.save(filepath + "/total_label.npy", total_label)

    # Plot confusion matrix
    plot_confusion_matrix(total_label, total_pre, [0, 1],
                          savename=filepath + '/confusion_matrix.png',
                          title='MDDformer 10-Fold Confusion Matrix')

    # Final metrics summary
    logging.info('=' * 60)
    logging.info('MDDformer 10-Fold Cross-Validation Results')
    logging.info('=' * 60)
    logging.info('Per-fold accuracy: {}'.format([f"{x:.2f}" for x in totals]))
    logging.info('Average accuracy: {:.2f}%'.format(sum(totals) / len(totals)))
    logging.info('Per-fold precision: {}'.format([f"{x:.4f}" for x in ps]))
    logging.info('Average precision: {:.4f}'.format(sum(ps) / len(ps)))
    logging.info('Per-fold recall: {}'.format([f"{x:.4f}" for x in rs]))
    logging.info('Average recall: {:.4f}'.format(sum(rs) / len(rs)))
    logging.info('Per-fold F1: {}'.format([f"{x:.4f}" for x in f1s]))
    logging.info('Average F1: {:.4f}'.format(sum(f1s) / len(f1s)))
    logging.info('=' * 60)

    print("\n" + "=" * 60)
    print("MDDformer 10-Fold Cross-Validation Summary")
    print("=" * 60)
    print(f"Average Accuracy: {sum(totals) / len(totals):.2f}%")
    print(f"Average Precision: {sum(ps) / len(ps):.4f}")
    print(f"Average Recall: {sum(rs) / len(rs):.4f}")
    print(f"Average F1: {sum(f1s) / len(f1s):.4f}")
    print("=" * 60)
