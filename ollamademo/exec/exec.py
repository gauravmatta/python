import io
import sys

code = """
import random
print("hello")
print(random.randint(1,10))"""
exec(code)
try:
    exec ("""print("Hello world")""")
except Exception as e:
    print(f"Error: {e}")

result = exec(code)
print(result)

old_stdout = sys.stdout
sys.stdout = buffer = io.StringIO()
code = """
import random
print("hello from system")
print(random.randint(1,10))"""
try:
    exec (code)
except Exception as e:
    print(f"Error FROM SYSTEM: {e}")

sys.stdout = old_stdout
output = buffer.getvalue()
print("Captured output:")
print(output)