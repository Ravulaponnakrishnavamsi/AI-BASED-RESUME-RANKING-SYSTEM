from dotenv import load_dotenv
import os

print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir(os.getcwd())}")

load_dotenv()
key = os.getenv("GROQ_API_KEY")

if key:
    print(f"SUCCESS: GROQ_API_KEY found! (Length: {len(key)})")
    if key.startswith("gsk_"):
        print("Key format looks correct (starts with gsk_).")
    else:
        print("WARNING: Key does not start with 'gsk_'. It might be invalid.")
else:
    print("ERROR: GROQ_API_KEY not found in environment.")

email = os.getenv("SENDER_EMAIL")
password = os.getenv("SENDER_PASSWORD")

if email:
    print(f"SUCCESS: SENDER_EMAIL found: {email}")
else:
    print("ERROR: SENDER_EMAIL not found in environment.")

if password:
    print("SUCCESS: SENDER_PASSWORD found.")
else:
    print("ERROR: SENDER_PASSWORD not found in environment.")

try:
    with open(".env", "r") as f:
        content = f.read()
        print("Found .env file. Content preview (first 10 chars):", content[:10])
except FileNotFoundError:
    print("ERROR: .env file NOT found in current directory.")
