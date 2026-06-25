import numpy as np

X = np.load("X_cnn.npy")
y = np.load("y_cnn.npy")

print(f"X shape: {X.shape}")
print(f"X min: {X.min():.3f}")
print(f"X max: {X.max():.3f}")
print(f"X mean: {X.mean():.3f}")
print(f"X has NaN: {np.isnan(X).any()}")
print(f"X has Inf: {np.isinf(X).any()}")
print(f"\ny unique values: {np.unique(y)}")
print(f"y distribution: {dict(zip(*np.unique(y, return_counts=True)))}")