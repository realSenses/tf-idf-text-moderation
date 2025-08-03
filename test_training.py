import pandas as pd
import numpy as np
from train_model import train_toxic_comment_model
import warnings
warnings.filterwarnings('ignore')

# Test with a small subset
print("Loading dataset...")
df = pd.read_csv('train.csv')
print(f"Original dataset shape: {df.shape}")

# Create a smaller subset for testing
subset_size = 1000
df_subset = df.sample(n=subset_size, random_state=42)

# Save subset
df_subset.to_csv('train_subset.csv', index=False)
print(f"Created subset with {subset_size} samples")

# Try training on subset
print("\nTraining on subset...")
try:
    train_toxic_comment_model('train_subset.csv')
    print("\nTraining completed successfully!")
except Exception as e:
    print(f"\nError during training: {e}")
    import traceback
    traceback.print_exc()