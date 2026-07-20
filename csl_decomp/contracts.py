"""Loader for CSL-Agent contract YAML files.

Reads the ground_truth/*.yaml (or extracted/*.yaml) contract format from
CSL_Agent_Experiments into a typed dataclass, so decomposers and dispatchers
can consume contracts uniformly regardless of which extraction config
(S1 / S1+S2 / S1+S2+T / human) produced them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class AgentContract:
    agent_id: str
    contract_id: str = ""
    agent_name: str = ""
    description: str = ""
    tin: Dict[str, object] = field(default_factory=dict)
    tout: Dict[str, object] = field(default_factory=dict)
    pre_structural: str = ""
    pre_semantic: List[str] = field(default_factory=list)
    post_structural: str = ""
    post_semantic: List[str] = field(default_factory=list)
    satisfaction_rate: float = 0.0
    comp_type: str = ""
    raw: Dict[str, object] = field(default_factory=dict)

    @staticmethod
    def from_dict(agent_id: str, data: Dict[str, object]) -> "AgentContract":
        data = data or {}
        pre = data.get("pre") or {}
        post = data.get("post") or {}
        prob = data.get("prob") or {}
        comp = data.get("comp") or {}
        return AgentContract(
            agent_id=agent_id,
            contract_id=str(data.get("contract_id", "")),
            agent_name=str(data.get("agent_name", "")),
            description=str(data.get("description", "")),
            tin=data.get("tin") or {},
            tout=data.get("tout") or {},
            pre_structural=str(pre.get("structural", "")) if isinstance(pre, dict) else "",
            pre_semantic=list(pre.get("semantic", []) or []) if isinstance(pre, dict) else [],
            post_structural=str(post.get("structural", "")) if isinstance(post, dict) else "",
            post_semantic=list(post.get("semantic", []) or []) if isinstance(post, dict) else [],
            satisfaction_rate=float(prob.get("satisfaction_rate", 0.0) or 0.0) if isinstance(prob, dict) else 0.0,
            comp_type=str(comp.get("type", "")) if isinstance(comp, dict) else "",
            raw=data,
        )

    def capability_summary(self) -> str:
        """One-line natural-language capability card, used in decomposer prompts."""

        required = []
        tin_props = self.tin.get("properties") if isinstance(self.tin, dict) else None
        if isinstance(tin_props, dict):
            required = list(tin_props.keys())
        preds = ", ".join(self.pre_semantic) if self.pre_semantic else "none"
        return (
            f"{self.agent_id} ({self.agent_name or 'unnamed'}): {self.description or 'no description'} "
            f"| accepts: {', '.join(required) or 'unspecified'} "
            f"| pre: {preds} "
            f"| reliability: {self.satisfaction_rate:.2f}"
        )


def load_contracts(directory: str | Path, pattern: str = "*.yaml") -> Dict[str, AgentContract]:
    """Load every contract YAML in a directory, keyed by agent id.

    Agent id is taken from the filename prefix up to the first underscore
    (e.g. ground_truth/A01_contract_R1.yaml -> "A01"). Files that fail to
    parse are skipped with a warning rather than aborting the whole load.
    """

    directory = Path(directory)
    contracts: Dict[str, AgentContract] = {}
    for path in sorted(directory.glob(pattern)):
        agent_id = path.stem.split("_")[0]
        try:
            with open(path, encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
        except Exception as exc:
            print(f"[contracts] skipping {path.name}: {exc}")
            continue
        if not isinstance(data, dict):
            print(f"[contracts] skipping {path.name}: not a mapping")
            continue
        contracts[agent_id] = AgentContract.from_dict(agent_id, data)
    return contracts
