# save as debug.py
import numpy as np
from sklearn.model_selection import train_test_split

X = np.load("X_cnn.npy")
y = np.load("y_cnn.npy")

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, random_state=42, stratify=y_trainval
)

print(f"Train distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"Val distribution:   {dict(zip(*np.unique(y_val, return_counts=True)))}")
print(f"Test distribution:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

# Check actual data values
print(f"\nX_train min: {X_train.min():.3f}")
print(f"X_train max: {X_train.max():.3f}")
print(f"X_train mean: {X_train.mean():.3f}")
print(f"X_train std: {X_train.std():.3f}")
