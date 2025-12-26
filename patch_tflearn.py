"""
Patch script to fix tflearn compatibility with TensorFlow 2.x
Fixes the is_sequence import error in tflearn/layers/recurrent.py
"""
import os
import tflearn

# Find tflearn installation directory
tflearn_dir = os.path.dirname(tflearn.__file__)
recurrent_file = os.path.join(tflearn_dir, 'layers', 'recurrent.py')

print(f"Patching: {recurrent_file}")

# Read the file
with open(recurrent_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the problematic import
old_import = "from tensorflow.python.util.nest import is_sequence"
new_import = """# Compatibility fix for TensorFlow 2.x
try:
    from tensorflow.python.util.nest import is_sequence
except ImportError:
    # TensorFlow 2.x moved is_sequence
    from tensorflow.python.util import nest
    is_sequence = nest._is_sequence"""

if old_import in content:
    content = content.replace(old_import, new_import)
    
    # Write the patched file
    with open(recurrent_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Successfully patched recurrent.py!")
    print("  Fixed: is_sequence import for TensorFlow 2.x compatibility")
else:
    print("✗ Import statement not found or already patched")

# Test the import
print("\nTesting tflearn import...")
try:
    import tflearn
    print("✓ tflearn imported successfully!")
except Exception as e:
    print(f"✗ Error: {e}")
