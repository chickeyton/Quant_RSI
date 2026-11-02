"""Shared finite factor-expression grammar and scalar arithmetic."""

import math
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.contracts_base import StrictModel

Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]


class ExpressionNode(StrictModel):
    name: Name
    op: Literal["leaf", "add", "subtract", "multiply"]
    inputs: tuple[Name, ...] = ()
    leaf: Name | None = None

    @model_validator(mode="after")
    def arity(self) -> Self:
        if self.op == "leaf":
            if self.leaf is None or self.inputs:
                raise ValueError("leaf node requires one named source")
        elif self.leaf is not None or len(self.inputs) != 2:
            raise ValueError("binary node requires two preceding nodes")
        return self


class FactorExpression(StrictModel):
    schema_version: Literal[1] = 1
    nodes: tuple[ExpressionNode, ...] = Field(min_length=1, max_length=7)

    @model_validator(mode="after")
    def dag(self) -> Self:
        depths = {}
        for node in self.nodes:
            if node.name in depths or any(name not in depths for name in node.inputs):
                raise ValueError("expression requires unique topologically ordered nodes")
            depths[node.name] = 1 + max((depths[name] for name in node.inputs), default=0)
            if depths[node.name] > 3:
                raise ValueError("expression depth exceeds registered bound")
        needed = {self.nodes[-1].name}
        for node in reversed(self.nodes):
            if node.name in needed:
                needed.update(node.inputs)
        if set(depths) != needed:
            raise ValueError("expression contains unused nodes")
        return self

    @property
    def ref(self):
        # Names and commutative argument order do not create novel definitions.
        values = {}
        for node in self.nodes:
            args = [values[name] for name in node.inputs]
            if node.op in {"add", "multiply"}:
                args.sort(key=canonical)
            values[node.name] = [node.op, node.leaf] if node.op == "leaf" else [node.op, args]
        return digest(canonical(values[self.nodes[-1].name]))


def evaluate_expression(expression, sources):
    reasons = []
    values = {}
    for node in expression.nodes:
        if node.op == "leaf":
            if node.leaf not in sources:
                raise ValueError("expression refers to an unregistered leaf")
            value = sources[node.leaf]
        else:
            a, b = (values[name] for name in node.inputs)
            value = (
                None
                if a is None or b is None
                else (a + b if node.op == "add" else a - b if node.op == "subtract" else a * b)
            )
        if value is not None and not math.isfinite(value):
            reasons.append("expression_nonfinite")
            value = None
        values[node.name] = value
    return dict(
        value=values[expression.nodes[-1].name],
        reasons=sorted(set(reasons)),
    )
