def count_pascal_ones(max_row: int) -> int:
    """Use when the user asks to count the number of 1s in Pascal's triangle up to a certain row.
    
    Args:
        max_row: The maximum row index to include in the count.
        
    Returns:
        The total number of 1s in rows 0 through max_row.
    """
    if max_row < 0:
        return 0
    if max_row == 0:
        return 1
    return 1 + (max_row * 2)