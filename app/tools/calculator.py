"""A safe calculator tool. Parses the expression into an AST and evaluates
only a small, explicit whitelist of arithmetic node types — never calls
`eval()` on the input string. A malicious or malformed expression can only
ever fail with a clear ValueError; there is no code path from user input
to arbitrary code execution (no names, no attribute access, no function
calls, no subscripts are ever evaluated).
"""

import ast
import operator
from collections.abc import Callable

from pydantic import BaseModel, Field

_BINOPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARYOPS: dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorInput(BaseModel):
    expression: str = Field(..., description="An arithmetic expression, e.g. '(2 + 3) * 4'.")


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        binary_func = _BINOPS.get(type(node.op))
        if binary_func is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return binary_func(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        unary_func = _UNARYOPS.get(type(node.op))
        if unary_func is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return unary_func(_safe_eval(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


class CalculatorTool:
    name = "calculator"
    description = (
        "Evaluates a basic arithmetic expression (+, -, *, /, //, %, **, "
        "parentheses, unary +/-). No variables, names, or function calls."
    )
    input_schema = CalculatorInput

    async def execute(self, expression: str) -> float:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"invalid expression syntax: {exc}") from exc

        try:
            return _safe_eval(tree)
        except ZeroDivisionError as exc:
            raise ValueError("division by zero") from exc
