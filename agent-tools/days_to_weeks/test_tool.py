from tool import days_to_weeks
assert days_to_weeks(14) == {"weeks": 2, "remaining_days": 0}
assert days_to_weeks(10) == {"weeks": 1, "remaining_days": 3}