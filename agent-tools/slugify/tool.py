import re

def slugify(text: str) -> str:
    """
    Converts text into a URL-friendly slug.
    - Lowercase it
    - Replace spaces and symbols with single hyphens
    - Trim hyphens from the ends
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Replace non-alphanumeric characters with a hyphen
    # This handles spaces and symbols
    slug = re.sub(r'[^a-z0-9]+', '-', text)
    
    # Trim hyphens from ends and ensure we don't have double hyphens
    # The regex [^a-z0-9]+ already collapses consecutive symbols into one hyphen
    return slug.strip('-')
