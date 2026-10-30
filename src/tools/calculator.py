import operator

_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
}

def calculator_tool(op1: float, op2: float, operator: str) -> str:
    if operator not in _OPS:
        return "Error: unsupported operator"
    try:
        return str(_OPS[operator](op1, op2))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception:
        return "Error: invalid input"