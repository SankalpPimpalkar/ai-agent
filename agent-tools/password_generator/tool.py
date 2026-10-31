import secrets
import string
import math

def password_generator(length: int = 20, min_digits: int = 2, min_symbols: int = 2) -> str:
    """Use when the user asks to generate a random password with specific requirements."""
    digits = string.digits
    symbols = string.punctuation
    alphabet = string.ascii_letters + digits + symbols
    
    password = []
    # Ensure requirements are met
    for _ in range(min_digits):
        password.append(secrets.choice(digits))
    for _ in range(min_symbols):
        password.append(secrets.choice(symbols))
    
    # Fill the rest
    for _ in range(length - (min_digits + min_symbols)):
        password.append(secrets.choice(alphabet))
    
    # Shuffle
    secrets.SystemRandom().shuffle(password)
    
    password_str = "".join(password)
    
    # Calculate entropy
    # The entropy of a password is log2(charset_size ^ length)
    # which is length * log2(charset_size)
    charset_size = len(alphabet)
    entropy = length * math.log2(charset_size)
    
    return f"Password: {password_str}\nEntropy: {entropy:.2f} bits"
