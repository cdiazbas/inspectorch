"""
Example: Smart Device Selection in Inspectorch

Demonstrates the auto-fallback behavior when requesting GPU devices
on different hardware platforms.
"""

import torch
from inspectorch import DensityEstimator
from inspectorch.utils import resolve_device

print("=" * 70)
print("Smart Device Selection in Inspectorch")
print("=" * 70)
print()

# ============================================================================
# Example 1: Using resolve_device() directly
# ============================================================================
print("Example 1: resolve_device() - Direct device resolution")
print("-" * 70)
print()

# This is perfect for scripts - you can hardcode "cuda" and it will
# automatically work on any platform!
print("On a Mac with MPS:")
print("  resolve_device('cuda') ->", resolve_device('cuda', verbose=True))
print()

print("On a Linux/Windows machine with CUDA:")
print("  resolve_device('cuda') -> 'cuda:0'  (from the error message above)")
print()

print("Using 'auto' for automatic selection:")
print("  resolve_device('auto') ->", resolve_device('auto', verbose=True))
print()

# ============================================================================
# Example 2: Using DensityEstimator with smart device handling
# ============================================================================
print()
print("Example 2: Using DensityEstimator with device='cuda'")
print("-" * 70)
print()

# This is the recommended way! Simply request 'cuda' and it will work on any platform.
# On a Mac: Automatically uses MPS
# On Linux/Windows: Uses CUDA
# Anywhere: Falls back to CPU if neither CUDA nor MPS is available

device = "cuda"  # Request GPU (works everywhere!)

print(f"Requesting device='{device}'...")
model = DensityEstimator(type="normalizing_flow")

# The model will be configured to use the best available device

print()
print("=" * 70)
print("Key Benefits:")
print("=" * 70)
print("✓ Write once, run anywhere: specify 'cuda' and it works on:")
print("  - Linux/Windows with NVIDIA GPU")
print("  - macOS with Apple Silicon (auto-falls back to MPS)")
print("  - Any machine (auto-falls back to CPU)")
print()
print("✓ No code changes needed when moving between platforms")
print("✓ Automatic device migration handled by inspectorch")
print()
