import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import pandas
    print("✓ pandas imported successfully")
except ImportError as e:
    print(f"✗ pandas import failed: {e}")

try:
    import numpy
    print("✓ numpy imported successfully")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")

try:
    import sklearn
    print("✓ scikit-learn imported successfully")
except ImportError as e:
    print(f"✗ scikit-learn import failed: {e}")

try:
    import matplotlib
    print("✓ matplotlib imported successfully")
except ImportError as e:
    print(f"✗ matplotlib import failed: {e}")

try:
    import seaborn
    print("✓ seaborn imported successfully")
except ImportError as e:
    print(f"✗ seaborn import failed: {e}")

print("\nAll basic imports successful!")