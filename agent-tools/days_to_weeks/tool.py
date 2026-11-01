def days_to_weeks(total_days: int) -> dict:
    """Use when the user asks to convert a number of days into weeks and remaining days."""
    weeks = total_days // 7
    remaining_days = total_days % 7
    return {"weeks": weeks, "remaining_days": remaining_days}