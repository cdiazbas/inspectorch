#!/usr/bin/env python
"""
Verification script to test that all flow matching backends support all
velocity models.
"""

import sys

sys.path.insert(0, "/mn/stornext/d20/RoCS/carlosjd/projects/INSPECTORCH/inspectorch")

from inspectorch import DensityEstimator

backends = ["flow_matching_ffm", "flow_matching_sbi", "flow_matching_cfm"]
architectures = ["AdaMLP", "ResNet", "ResNetFlow", "FourierMLP", "MLPLegacy"]

print("=" * 70)
print("TESTING ARCHITECTURE SUPPORT ACROSS ALL BACKENDS")
print("=" * 70)

results = {}
for backend in backends:
    results[backend] = {}
    print(f"\n{'=' * 70}")
    print(f"Testing backend: {backend}")
    print(f"{'=' * 70}")

    for arch in architectures:
        try:
            print(f"\n  Testing {arch}...", end=" ")
            model = DensityEstimator(type=backend)
            model.create_flow(
                input_size=2, architecture=arch, num_layers=2, hidden_features=32
            )
            print("✓ SUCCESS")
            results[backend][arch] = "✓"
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results[backend][arch] = "✗"

# Print summary table
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Backend':<25} " + " ".join([f"{arch:<12}" for arch in architectures]))
print("-" * 70)
for backend in backends:
    row = f"{backend:<25} "
    for arch in architectures:
        row += f"{results[backend][arch]:<12} "
    print(row)
print("=" * 70)

# Check if all passed
all_passed = all(results[b][a] == "✓" for b in backends for a in architectures)
if all_passed:
    print("\n✓ ALL TESTS PASSED! All backends support all architectures.")
    sys.exit(0)
else:
    print("\n✗ SOME TESTS FAILED. Check the table above for details.")
    sys.exit(1)
