def count_pascal_ones(num_rows: int) -> int:
    """Use when the user asks to count the ones in Pascal's triangle for rows 0 to N."""
    if num_rows <= 0:
        return 0
    # Row 0 has one '1'. Rows 1 to N-1 have two '1's each.
    # This covers rows 0 up to row num_rows-1.
    # For rows 0 to 20, num_rows is 21.
    return 1 + (max(0, num_rows - 1) * 2)
