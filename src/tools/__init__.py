from ollama import Tool
from typing import List, Dict
from .calculator import _OPS, calculator_tool

tools: List[Tool] = []
functions: Dict[str, function] = {}

# Calculator Tool
functions['calculator_tool'] = calculator_tool
tools.append(Tool(
    type='function',
    function=Tool.Function(
        name="calculator_tool",
        description="Perform basic arithmetic on two numbers. After final operations explain the answer naturally",
        parameters=Tool.Function.Parameters(
            type="object",
            properties={
                "op1": {
                    "type": "float",
                    "description": "First Operand"
                },
                "op2": {
                    "type": "float",
                    "description": "Second Operand"
                },
                "operator": {
                    "type": "string",
                    "enum": list(_OPS.keys()),
                    "description": "Operator"
                },
            },
            required=["op1", "op2", "operator"]
        )
    )
))