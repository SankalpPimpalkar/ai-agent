def sha256(text: str) -> str:
    """Use when the user asks for a SHA-256 hash."""
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()