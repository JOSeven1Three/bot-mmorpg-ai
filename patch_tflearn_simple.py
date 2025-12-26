"""Simple Python script to patch tflearn recurrent.py for TensorFlow 2.x compatibility"""
import os

# Find the tflearn recurrent.py file
venv_path = r".venv\Lib\site-packages\tflearn\layers\recurrent.py"

print(f"Patching: {venv_path}")

# Read the file
with open(venv_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and replace the problematic import
new_lines = []
for line in lines:
    if line.strip() == "from tensorflow.python.util.nest import is_sequence":
        # Replace with compatibility fix
        new_lines.append("# Patched for TensorFlow 2.x compatibility\n")
        new_lines.append("try:\n")
        new_lines.append("    from tensorflow.python.util.nest import is_sequence\n")
        new_lines.append("except ImportError:\n")
        new_lines.append("    # TensorFlow 2.x: is_sequence was moved/renamed\n")
        new_lines.append("    def is_sequence(seq):\n")
        new_lines.append("        try:\n")
        new_lines.append("            return isinstance(seq, (list, tuple))\n")
        new_lines.append("        except:\n")
        new_lines.append("            return False\n")
    else:
        new_lines.append(line)

# Write back
with open(venv_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✓ Successfully patched recurrent.py")

# Test import
try:
    import tflearn
    print("✓ tflearn imported successfully!")
except Exception as e:
    print(f"✗ Import failed: {e}")
