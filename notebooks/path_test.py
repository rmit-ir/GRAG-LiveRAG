"""
Test script to verify that notebooks/setup_path.py works correctly.
"""
print("Testing notebooks path setup...")

# Import setup_path to configure Python path
import setup_path

# Try importing from services package
try:
    from services import aws_utils
    print("✅ Successfully imported from services package")
    print(f"Available modules in services: {dir(aws_utils)}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    
print("Test complete.")
