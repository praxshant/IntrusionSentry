"""Patch Ultralytics to work with PyTorch 2.6"""
import torch
from ultralytics.nn import tasks

# Store original torch.load
_original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, *, weights_only=None, **kwargs):
    """Patched torch.load that sets weights_only=False for Ultralytics models"""
    # Force weights_only=False for .pt files (YOLO weights)
    if isinstance(f, str) and f.endswith('.pt'):
        weights_only = False
    return _original_torch_load(f, map_location=map_location, pickle_module=pickle_module, 
                                 weights_only=weights_only, **kwargs)

# Apply patch
torch.load = patched_torch_load
print("[INFO] ✅ Applied PyTorch 2.6 compatibility patch for Ultralytics")