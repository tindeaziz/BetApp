
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Attempting to import app...")
try:
    import app
    print("✅ app.py imported successfully. Syntax is correct.")
except Exception as e:
    print(f"❌ Syntax Error in app.py: {e}")
    sys.exit(1)
