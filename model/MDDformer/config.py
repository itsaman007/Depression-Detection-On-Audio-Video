"""
Experiment Configuration for MDDformer
=======================================
Central configuration file for all experiments.
Modify this file to run different experiment configurations.

Usage:
  from config import get_config
  cfg = get_config('baseline')  # or 'improved', 'ablation_no_pos', etc.
"""

import os


# ========================== Data Paths ==========================
# Change these to match your local setup
DATA_ROOT = r"D:\MDD"

PATHS = {
    'tcn_video': os.path.join(DATA_ROOT, "TCN_processed_video"),
    'audio': os.path.join(DATA_ROOT, "Audio_feature"),
    'label': os.path.join(DATA_ROOT, "label"),
    'raw_video': os.path.join(DATA_ROOT, "Video_feature"),
    'log_dir': os.path.join(DATA_ROOT, "model", "MDDformer", "logs"),
    'save_dir': os.path.join(DATA_ROOT, "model", "MDDformer", "checkpoints"),
}


# ========================== Experiment Configs ==========================

CONFIGS = {
    # Phase 1: Reproduce paper results (~76.88%)
    'baseline': {
        'model': 'MDDformer',         # Original MDDformer (Net class)
        'lr': 1e-5,
        'epochs': 300,
        'warmup_epochs': 0,
        'batch_size_train': 15,
        'batch_size_dev': 4,
        'weight_decay': 0,
        'label_smoothing': 0.0,
        'grad_clip': 0,
        'mixup_alpha': 0,
        'audio_noise_std': 0,
        'early_stop_patience': 0,     # Disabled
        'swa_start_epoch': 0,         # Disabled
        'optimizer': 'Adam',
        'schedule': 'cosine',
        'seed': 2222,
        'device': 0,
        'model_config': {},            # Uses default Net() params
    },

    # Phase 2: Improved model targeting 80%+
    'improved': {
        'model': 'MDDformerImproved',
        'lr': 3e-5,
        'epochs': 400,
        'warmup_epochs': 15,
        'batch_size_train': 12,
        'batch_size_dev': 4,
        'weight_decay': 1e-4,
        'label_smoothing': 0.1,
        'grad_clip': 1.0,
        'mixup_alpha': 0.2,
        'audio_noise_std': 0.01,
        'early_stop_patience': 50,
        'swa_start_epoch': 300,
        'optimizer': 'AdamW',
        'schedule': 'cosine',
        'seed': 2222,
        'device': 0,
        'model_config': {
            'num_cross_attn_layers': 2,
            'head_num': 8,
            'ffn_expansion': 4,
            'attn_dropout': 0.1,
            'ffn_dropout': 0.1,
            'proj_dropout': 0.1,
            'cls_dropout': 0.2,
            'use_pos_encoding': True,
            'pos_encoding_type': 'sinusoidal',
        },
    },

    # =============== Ablation Studies ===============
    # Each ablation modifies ONE thing from the 'improved' config
    
    # Ablation: No positional encoding
    'ablation_no_pos': {
        'model': 'MDDformerImproved',
        'lr': 3e-5, 'epochs': 400, 'warmup_epochs': 15,
        'batch_size_train': 12, 'batch_size_dev': 4,
        'weight_decay': 1e-4, 'label_smoothing': 0.1,
        'grad_clip': 1.0, 'mixup_alpha': 0.2, 'audio_noise_std': 0.01,
        'early_stop_patience': 50, 'swa_start_epoch': 300,
        'optimizer': 'AdamW', 'schedule': 'cosine', 'seed': 2222, 'device': 0,
        'model_config': {
            'num_cross_attn_layers': 2, 'head_num': 8, 'ffn_expansion': 4,
            'attn_dropout': 0.1, 'ffn_dropout': 0.1, 'proj_dropout': 0.1,
            'cls_dropout': 0.2,
            'use_pos_encoding': False,  # <-- Changed
        },
    },

    # Ablation: Single cross-attention layer (like original)
    'ablation_1_layer': {
        'model': 'MDDformerImproved',
        'lr': 3e-5, 'epochs': 400, 'warmup_epochs': 15,
        'batch_size_train': 12, 'batch_size_dev': 4,
        'weight_decay': 1e-4, 'label_smoothing': 0.1,
        'grad_clip': 1.0, 'mixup_alpha': 0.2, 'audio_noise_std': 0.01,
        'early_stop_patience': 50, 'swa_start_epoch': 300,
        'optimizer': 'AdamW', 'schedule': 'cosine', 'seed': 2222, 'device': 0,
        'model_config': {
            'num_cross_attn_layers': 1,  # <-- Changed from 2
            'head_num': 8, 'ffn_expansion': 4,
            'attn_dropout': 0.1, 'ffn_dropout': 0.1, 'proj_dropout': 0.1,
            'cls_dropout': 0.2, 'use_pos_encoding': True, 'pos_encoding_type': 'sinusoidal',
        },
    },

    # Ablation: 4 heads (like original) instead of 8
    'ablation_4_heads': {
        'model': 'MDDformerImproved',
        'lr': 3e-5, 'epochs': 400, 'warmup_epochs': 15,
        'batch_size_train': 12, 'batch_size_dev': 4,
        'weight_decay': 1e-4, 'label_smoothing': 0.1,
        'grad_clip': 1.0, 'mixup_alpha': 0.2, 'audio_noise_std': 0.01,
        'early_stop_patience': 50, 'swa_start_epoch': 300,
        'optimizer': 'AdamW', 'schedule': 'cosine', 'seed': 2222, 'device': 0,
        'model_config': {
            'num_cross_attn_layers': 2,
            'head_num': 4,  # <-- Changed from 8
            'ffn_expansion': 4,
            'attn_dropout': 0.1, 'ffn_dropout': 0.1, 'proj_dropout': 0.1,
            'cls_dropout': 0.2, 'use_pos_encoding': True, 'pos_encoding_type': 'sinusoidal',
        },
    },

    # Ablation: No label smoothing
    'ablation_no_ls': {
        'model': 'MDDformerImproved',
        'lr': 3e-5, 'epochs': 400, 'warmup_epochs': 15,
        'batch_size_train': 12, 'batch_size_dev': 4,
        'weight_decay': 1e-4,
        'label_smoothing': 0.0,  # <-- Changed
        'grad_clip': 1.0, 'mixup_alpha': 0.2, 'audio_noise_std': 0.01,
        'early_stop_patience': 50, 'swa_start_epoch': 300,
        'optimizer': 'AdamW', 'schedule': 'cosine', 'seed': 2222, 'device': 0,
        'model_config': {
            'num_cross_attn_layers': 2, 'head_num': 8, 'ffn_expansion': 4,
            'attn_dropout': 0.1, 'ffn_dropout': 0.1, 'proj_dropout': 0.1,
            'cls_dropout': 0.2, 'use_pos_encoding': True, 'pos_encoding_type': 'sinusoidal',
        },
    },

    # Ablation: No MixUp
    'ablation_no_mixup': {
        'model': 'MDDformerImproved',
        'lr': 3e-5, 'epochs': 400, 'warmup_epochs': 15,
        'batch_size_train': 12, 'batch_size_dev': 4,
        'weight_decay': 1e-4, 'label_smoothing': 0.1,
        'grad_clip': 1.0,
        'mixup_alpha': 0.0,  # <-- Changed
        'audio_noise_std': 0.01,
        'early_stop_patience': 50, 'swa_start_epoch': 300,
        'optimizer': 'AdamW', 'schedule': 'cosine', 'seed': 2222, 'device': 0,
        'model_config': {
            'num_cross_attn_layers': 2, 'head_num': 8, 'ffn_expansion': 4,
            'attn_dropout': 0.1, 'ffn_dropout': 0.1, 'proj_dropout': 0.1,
            'cls_dropout': 0.2, 'use_pos_encoding': True, 'pos_encoding_type': 'sinusoidal',
        },
    },

    # Ablation: Learnable positional encoding
    'ablation_learnable_pos': {
        'model': 'MDDformerImproved',
        'lr': 3e-5, 'epochs': 400, 'warmup_epochs': 15,
        'batch_size_train': 12, 'batch_size_dev': 4,
        'weight_decay': 1e-4, 'label_smoothing': 0.1,
        'grad_clip': 1.0, 'mixup_alpha': 0.2, 'audio_noise_std': 0.01,
        'early_stop_patience': 50, 'swa_start_epoch': 300,
        'optimizer': 'AdamW', 'schedule': 'cosine', 'seed': 2222, 'device': 0,
        'model_config': {
            'num_cross_attn_layers': 2, 'head_num': 8, 'ffn_expansion': 4,
            'attn_dropout': 0.1, 'ffn_dropout': 0.1, 'proj_dropout': 0.1,
            'cls_dropout': 0.2, 'use_pos_encoding': True,
            'pos_encoding_type': 'learnable',  # <-- Changed
        },
    },
}


def get_config(name='baseline'):
    """Get experiment configuration by name."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Available: {list(CONFIGS.keys())}")
    cfg = CONFIGS[name].copy()
    cfg['name'] = name
    cfg['paths'] = PATHS
    return cfg


def list_configs():
    """List all available experiment configurations."""
    for name, cfg in CONFIGS.items():
        model = cfg.get('model', 'N/A')
        lr = cfg.get('lr', 'N/A')
        epochs = cfg.get('epochs', 'N/A')
        print(f"  {name:25s} | model={model:20s} | lr={lr} | epochs={epochs}")


if __name__ == '__main__':
    print("Available experiment configurations:")
    print("-" * 80)
    list_configs()
    print("-" * 80)
    
    # Test loading a config
    cfg = get_config('improved')
    print(f"\n'improved' config details:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
