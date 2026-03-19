"""
TCN Video Feature Extraction Script
=====================================
Processes raw OpenFace video CSV files into compact .npy feature arrays
suitable for the MDDformer model.

Input: Raw OpenFace CSV files from Video_feature/ directory
Output: .npy files with shape (915, 171) saved to TCN_processed_video/

Features extracted (171 total):
  - 17 Action Unit features (AU)
  - 136 Facial landmark features
  - 12 Gaze features (doubled with zeros)
  - 6 Head pose features

Pipeline:
  1. Invalid frame interpolation (propagate last valid frame)
  2. Temporal subsampling (every 6th frame from first 5490 frames -> 915 frames)
  3. Feature selection and concatenation
  4. Zero-padding if shorter than 915 frames

Usage:
  python model/MDDformer/extract_tcn_features.py
"""

import pandas as pd
import os
import numpy as np


def validFrame(frames):
    """Fill invalid frames with the last valid frame's features."""
    validF = None
    for row in range(frames.shape[0]):
        if frames[row][4] == 1:
            validF = frames[row].copy()
            break
    
    if validF is None:
        print("Warning: No valid frames found!")
        return frames
        
    for row in range(frames.shape[0]):
        if frames[row][4] == 0:
            frames[row][5:] = validF[5:]
        if frames[row][4] == 1:
            validF = frames[row].copy()
    return frames


def chouzhen(_feature):
    """Temporal subsampling: take every 6th frame."""
    flag = 0
    for i in range(0, len(_feature), 6):
        if flag == 0:
            feature = _feature[i]
            flag = 1
        else:
            feature = np.vstack((feature, _feature[i]))
    return feature


def split(data):
    """Take first 5490 frames, subsample to 915, zero-pad if shorter."""
    _data = chouzhen(data[:5490, ])
    
    if _data.shape[0] < 915:
        zeros = np.zeros([(915 - _data.shape[0]), _data.shape[1]])
        _data = np.vstack((_data, zeros))
    return _data


def extract_features_from_csv(file_path):
    """
    Extract 171-dim features from a single OpenFace CSV file.
    
    Returns:
        np.ndarray of shape (915, 171) or None if extraction fails
    """
    try:
        file_csv = pd.read_csv(file_path, low_memory=False)
        # Force all columns to numeric, coercing errors to NaN
        file_csv = file_csv.apply(pd.to_numeric, errors='coerce')
        data = file_csv.to_numpy(dtype=np.float64)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    try:
        data = validFrame(data)
    except Exception as e:
        print(f"Video frame issue in {file_path}: {e}")
        return None
    
    data = split(data)
    
    # Drop first 5 columns (frame metadata: frame, face_id, timestamp, confidence, success)
    data = np.delete(data, [0, 1, 2, 3, 4], axis=1)
    
    # Extract feature groups
    gaze = data[:, 0:6]                  # 6 gaze features
    gaze_zero = np.zeros_like(gaze)
    gaze = np.hstack((gaze, gaze_zero))  # 12 gaze features (doubled with zeros)
    
    pose = data[:, 288:294]              # 6 pose features
    features = data[:, 294:430]          # 136 landmark features
    au = data[:, 447:465]                # 18 AU features
    
    # Process AUs: remove one, reorder
    au = np.delete(au, [5], axis=1)      # Remove 6th AU -> 17 AUs
    au[:, [12, 13]] = au[:, [13, 12]]    # Swap
    au[:, [13, 14]] = au[:, [14, 13]]    # Swap
    
    # Concatenate all features: 17 + 136 + 12 + 6 = 171
    feature = np.hstack((au, features, gaze, pose))
    
    # Check for NaN values
    feature = feature.astype(np.float64)
    nan_count = np.isnan(feature).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values in {file_path}. Replacing with 0.")
        feature = np.nan_to_num(feature, nan=0.0)
    
    return feature


def getTCNVideoFeature(trainPath, targetPath):
    """
    Process all video CSV files and save as .npy features.
    
    Args:
        trainPath: Directory containing raw video CSV files
        targetPath: Directory to save processed .npy files
    """
    if not os.path.exists(targetPath):
        os.makedirs(targetPath)
    
    files = os.listdir(trainPath)
    csv_files = [f for f in files if f.endswith('.csv')]
    csv_files.sort(key=lambda x: int(x.split('.')[0]))
    
    total = len(csv_files)
    success = 0
    failed = 0
    
    print(f"Processing {total} video CSV files...")
    print(f"Source: {trainPath}")
    print(f"Target: {targetPath}")
    print("-" * 60)
    
    for idx, file in enumerate(csv_files):
        file_path = os.path.join(trainPath, file)
        feature = extract_features_from_csv(file_path)
        
        if feature is not None:
            save_name = file.split(".")[0]
            np.save(os.path.join(targetPath, save_name), feature)
            success += 1
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{total} files... ({success} success, {failed} failed)")
        else:
            failed += 1
            print(f"  FAILED: {file}")
    
    print("-" * 60)
    print(f"Feature extraction complete!")
    print(f"  Total: {total}, Success: {success}, Failed: {failed}")
    print(f"  Output shape: (915, 171)")
    print(f"  Saved to: {targetPath}")


def verify_features(feature_path, audio_path):
    """Verify that video and audio features are properly aligned."""
    video_files = set(f.replace('.npy', '') for f in os.listdir(feature_path) if f.endswith('.npy'))
    audio_files = set(f.replace('.npy', '') for f in os.listdir(audio_path) if f.endswith('.npy'))
    
    common = video_files & audio_files
    video_only = video_files - audio_files
    audio_only = audio_files - video_files
    
    print(f"\nFeature Alignment Verification:")
    print(f"  Video features: {len(video_files)}")
    print(f"  Audio features: {len(audio_files)}")
    print(f"  Common (aligned): {len(common)}")
    
    if video_only:
        print(f"  Video-only (no audio): {len(video_only)}")
    if audio_only:
        print(f"  Audio-only (no video): {len(audio_only)}")
    
    # Check shapes of a sample
    if common:
        sample = list(common)[0]
        v = np.load(os.path.join(feature_path, sample + '.npy'))
        a = np.load(os.path.join(audio_path, sample + '.npy'))
        print(f"\n  Sample '{sample}':")
        print(f"    Video shape: {v.shape} (expected: 915x171)")
        print(f"    Audio shape: {a.shape} (expected: Tx128)")
    
    return len(common)


if __name__ == "__main__":
    # Paths
    VIDEO_SOURCE = r"D:\MDD\Video_feature"
    TCN_TARGET = r"D:\MDD\TCN_processed_video"
    AUDIO_PATH = r"D:\MDD\Audio_feature"
    
    # Step 1: Extract TCN features from raw video CSVs
    getTCNVideoFeature(VIDEO_SOURCE, TCN_TARGET)
    
    # Step 2: Verify alignment with audio features
    verify_features(TCN_TARGET, AUDIO_PATH)
