"""
Data loader for MDDformer model.
Loads TCN-processed video features (.npy, shape 915x171) and audio features (.npy, shape Tx128).
Pads/truncates to fixed lengths: video=915, audio=186.
Applies temporal masking augmentation during training.
"""

import torch.utils.data as udata
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import torch
import random

normalVideoShape = 915
normalAudioShape = 186


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


class AffectnetSampler(torch.utils.data.sampler.Sampler):
    """Inverse-frequency weighted sampler for class balancing."""
    def __init__(self, dataset):
        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)

        expression_count = [0] * 63
        for idx in self.indices:
            label = dataset.label[idx]
            expression_count[int(label)] += 1

        self.weights = torch.zeros(self.num_samples)
        for idx in self.indices:
            label = dataset.label[idx]
            self.weights[idx] = 1. / expression_count[int(label)]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class MyDataLoader(udata.Dataset):
    """
    Dataset for multimodal depression detection.
    
    Args:
        videoFileName: Path to directory containing TCN-processed video .npy files
        AudioFileName: Path to directory containing audio .npy files
        Kfolds: List of .npy filenames for this fold split
        labelPath: Path to directory containing {id}_Depression.csv label files
        type: "train" or "dev"
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
            id = file.split('.')[0]
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
        videoData = torch.from_numpy(videoData)
        audioData = torch.from_numpy(audioData)

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

        videoData = videoData.type(torch.float)
        audioData = audioData.type(torch.float)

        return videoData, audioData, label

    def __len__(self) -> int:
        return len(self.videoList)
