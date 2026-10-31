import math
import operator as op

BINARY = {
    "+": op.add,
    "-": op.sub,
    "*": op.mul,
    "/": op.truediv,
    "//": op.floordiv,
    "%": op.mod,
    "**": op.pow,
    "max": max,
    "min": min,
    "gcd": lambda a, b: math.gcd(int(a), int(b)),
    "lcm": lambda a, b: math.lcm(int(a), int(b)),
    "root": lambda a, b: a ** (1 / b),                  # b-th root of a
    "log": lambda a, b: math.log(a, b),                 # log of a with base b
    "percent_of": lambda a, b: a / 100 * b,             # a percent of b
    "percent_change": lambda a, b: (b - a) / a * 100,   # change from a to b
    "round": lambda a, b: round(a, int(b)),             # round a to b decimals
    "hypot": math.hypot,
}

UNARY = {
    "sqrt": math.sqrt,
    "abs": abs,
    "ln": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": lambda a: math.factorial(int(a)) if a == int(a) and 0 <= a <= 1000 else (_ for _ in ()).throw(ValueError),
    "sin": lambda a: math.sin(math.radians(a)),         # degrees
    "cos": lambda a: math.cos(math.radians(a)),
    "tan": lambda a: math.tan(math.radians(a)),
}


def _fmt(x) -> str:
    if isinstance(x, float):
        if x.is_integer() and abs(x) < 1e15:
            return str(int(x))
        return str(round(x, 10))
    return str(x)


def calculator_tool(op1: float, operator: str, op2: float = 0.0) -> str:
    """
    Use when the user asks for a calculation. Always use this instead of doing math yourself.
    """
    try:
        a, b = float(op1), float(op2)
        if operator in BINARY:
            return _fmt(BINARY[operator](a, b))
        if operator in UNARY:
            return _fmt(UNARY[operator](a))
        return f"Error: unsupported operator. Use one of: {', '.join(list(BINARY) + list(UNARY))}"
    except ZeroDivisionError:
        return "Error: division by zero"
    except (ValueError, OverflowError):
        return "Error: input is outside the valid range for this operation"
    except Exception:
        return "Error: invalid input"