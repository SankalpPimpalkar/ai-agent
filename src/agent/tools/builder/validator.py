import ast
from typing import Tuple


class ToolAstValidator:
    """Parses and validates Python tool code using the AST."""

    @staticmethod
    def json_type(annotation: str) -> str:
        base = annotation.split("[", 1)[0].strip().lower()
        return {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }.get(base, "string")

    def build_spec(self, code: str, name: str) -> Tuple[dict, list, str]:
        tree = ast.parse(code)
        fn = next(
            (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name),
            None,
        )
        if fn is None:
            raise ValueError(f"code must define a function named exactly '{name}'")
        args = fn.args.args
        first_default = len(args) - len(fn.args.defaults)
        props, required = {}, []
        for i, a in enumerate(args):
            t = ast.unparse(a.annotation) if a.annotation else "str"
            props[a.arg] = {
                "type": self.json_type(t),
                "description": a.arg.replace("_", " "),
            }
            if i < first_default:
                required.append(a.arg)
        doc = (ast.get_docstring(fn) or "").strip().splitlines()
        return props, required, (doc[0] if doc else "")
