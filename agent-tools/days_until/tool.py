from datetime import datetime

def days_until(target_date_str: str) -> int:
    """Use when the user asks for the number of days until a specific date.
    The target_date_str should be in 'YYYY-MM-DD' format.
    """
    today = datetime.now().date()
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    return (target_date - today).days
