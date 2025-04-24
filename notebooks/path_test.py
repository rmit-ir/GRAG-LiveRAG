"""
Test script to verify that notebooks/setup_path.py works correctly.
"""
print("Testing notebooks path setup...")

# Try importing from services package
try:
    from services import live_rag_aws_utils
    print("✅ Successfully imported from services package")
    print(f"Available modules in services: {dir(live_rag_aws_utils)}")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    
print("Test complete.")
