import math
import operator as op
from ollama import Tool

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

_OPS = list(BINARY.keys()) + list(UNARY.keys())


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


calculator_tool_spec = Tool(
    type="function",
    function=Tool.Function(
        name="calculator_tool",
        description="""
                Perform math calculations. Always use this instead of calculating in your head.
                Two-number operations (op1, operator, op2): + - * / // % ** max min gcd lcm hypot,
                root (op2-th root of op1), log (op1 with base op2), percent_of (op1 percent of op2),
                percent_change (change from op1 to op2), round (op1 to op2 decimals).
                One-number operations (op1, operator, no op2): sqrt abs ln log10 log2 ceil floor factorial,
                sin cos tan (angle in degrees).
                For longer expressions, call it step by step and use each result in the next call.
            """,
        parameters=Tool.Function.Parameters(
            type="object",
            properties={
                "op1": {"type": "float", "description": "First Operand"},
                "op2": {
                    "type": "float",
                    "description": "Second number. Omit for one-number operations like sqrt or factorial",
                },
                "operator": {
                    "type": "string",
                    "enum": _OPS,
                    "description": "Operation to apply. Two-number: + - * / // % ** max min gcd lcm root (op2-th root of op1) log (op1 with base op2) percent_of (op1 percent of op2) percent_change (from op1 to op2) round (op1 to op2 decimals) hypot. One-number (op2 ignored): sqrt abs ln log10 log2 ceil floor factorial sin cos tan (degrees).",
                },
            },
            required=["op1", "operator"],
        ),
    ),
)
