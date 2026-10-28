"""Finite typed factor expressions; no generated code or implicit source discovery."""

import math
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import Field, model_validator

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.factors.contracts import Name, NumericFactorDefinition


class FactorLeaf(StrictModel):
    name: Name
    role: Literal["target", "industry", "market", "news", "event"]
    definition: NumericFactorDefinition | None = None
    instance_ref: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    binding_ref: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def typed_source(self) -> Self:
        if (self.role in {"news", "event"}) != (self.definition is None):
            raise ValueError(
                "price leaves require numeric definitions; news uses its registered task"
            )
        if (self.role == "event") != (self.instance_ref is not None):
            raise ValueError("event leaves require exactly one frozen instance snapshot")
        return self


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


def leaf_expression(name):
    return FactorExpression(nodes=(ExpressionNode(name="root", op="leaf", leaf=name),))


def combine(left, right, op):
    nodes = []
    for prefix, graph in (("a_", left), ("b_", right)):
        nodes.extend(
            ExpressionNode(
                name=prefix + node.name,
                op=node.op,
                inputs=tuple(prefix + name for name in node.inputs),
                leaf=node.leaf,
            )
            for node in graph.nodes
        )
    nodes.append(
        ExpressionNode(
            name="root",
            op=op,
            inputs=("a_" + left.nodes[-1].name, "b_" + right.nodes[-1].name),
        )
    )
    return FactorExpression(nodes=tuple(nodes))


def grammar(leaves):
    """Finite non-incumbent frontier: leaves plus all distinct two-leaf compositions."""
    roots = [leaf_expression(leaf.name) for leaf in leaves]
    graphs = list(roots)
    for index, left in enumerate(roots):
        for right in roots[index + 1 :]:
            graphs.extend(combine(left, right, op) for op in ("add", "subtract", "multiply"))
    return graphs


def propose(frontier, history):
    """Feedback selects an operator, while a failed candidate never discards the frontier.

    Only development error deltas enter this pure interface. No protected fields,
    unseen labels, randomness, or generated instructions are accepted.
    """
    visited = {item["expression_ref"] for item in history}
    remaining = [graph for graph in frontier if graph.ref not in visited]
    if not remaining:
        return None
    preferred = (
        "leaf"
        if not history
        else (
            "multiply"
            if history[-1]["improvement_pp"] is not None and history[-1]["improvement_pp"] > 0
            else "subtract"
        )
    )
    return min(remaining, key=lambda g: (g.nodes[-1].op != preferred, frontier.index(g)))


def source_leaves(inputs, leaves, origin, as_of, basis):
    from quant_rsi.forecast.forecast_features import _history
    from quant_rsi.market.asof import session_index

    target = inputs.target[0]
    index = session_index(target.contract, origin)
    values, reasons, refs = {}, [], set()
    for leaf in leaves:
        if leaf.role == "event":
            from quant_rsi.core.artifacts import decode
            from quant_rsi.factors.lifecycle import value_at

            event = decode(inputs.evidence[leaf.instance_ref])
            result = value_at(event, as_of)
            values[leaf.name] = result["value"]
            reasons.extend(result["reasons"])
            refs.update(result["source_refs"])
            refs.add(leaf.instance_ref)
            continue
        if leaf.role == "news":
            from quant_rsi.factors.news import news_scalar

            result = news_scalar(inputs, origin, as_of)
            reasons.extend(result["reasons"])
            refs.update(result["source_refs"])
            # Registered dimensionless indicator, not an inferred quantity.
            values[leaf.name] = None if result["value"] is None else float(result["value"] > 0)
            continue
        source = inputs.target if leaf.role == "target" else inputs.contexts.get(leaf.role)
        window = leaf.definition.recipe.window
        if source is None or index < window:
            reasons.append(leaf.name + ":source_or_history_missing")
            values[leaf.name] = None
            continue
        closes = [s.close_at for s in target.contract.calendar.sessions[index - window : index + 1]]
        prices, problems, evidence = _history(source, closes, as_of, basis)
        reasons.extend(leaf.name + ":" + p for p in problems)
        refs.update(evidence)
        value = None
        if prices is not None:
            if leaf.definition.recipe.op == "momentum":
                value = prices[-1] / prices[0] - 1
            else:
                value = sum(b / a - 1 for a, b in zip(prices, prices[1:], strict=False)) / window
        values[leaf.name] = value
    return values, reasons, refs


def evaluate(expression, leaves, inputs, origin, as_of, basis=None):
    sources, reasons, refs = source_leaves(inputs, leaves, origin, as_of, basis)
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
        source_refs=sorted(refs),
    )


@dataclass(frozen=True)
class DiscoveryRecipe:
    """A frozen additional column, distinct from legacy ForecastRecipe identities."""

    graph: FactorExpression
    leaves: tuple[FactorLeaf, ...]
    group: str = "D1"

    @property
    def names(self):
        return ("momentum_5", "return_1", "intraday_return", "discovered_" + self.graph.ref[:16])

    @property
    def definition(self):
        return dict(
            schema_version=3,
            kind="discovered_factor_recipe",
            group=self.group,
            expression=self.graph.model_dump(mode="json"),
            leaves=[leaf.model_dump(mode="json") for leaf in self.leaves],
            feature_names=list(self.names),
            feature_units=["dimensionless_return"] * 3 + ["dimensionless"],
            missingness="common_registered_leaf_universe",
        )

    @property
    def ref(self):
        return digest(canonical(self.definition))


def recipe_from_binding(binding):
    return DiscoveryRecipe(
        FactorExpression.model_validate_json(canonical(binding["expression"])),
        tuple(FactorLeaf.model_validate_json(canonical(v)) for v in binding["leaves"]),
    )
