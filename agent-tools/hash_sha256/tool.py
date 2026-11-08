import hashlib

def hash_sha256(text: str) -> str:
    """Use when the user asks to hash a string using SHA-256."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()