"""
FastMCP-based memory server implementing a simple knowledge graph with
entities, relations, and observations persisted to a JSONL file.

Environment:
- MEMORY_FILE_PATH: absolute or relative path for the JSONL storage file.
  If relative, it's resolved next to this script. Defaults to memory.jsonl
  next to this file.
"""

import os
import json
from pathlib import Path
from typing import Any, Callable

from fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict


def _find_latest_run_dir(candidates: list[Path]) -> Path | None:
    runs: list[Path] = []
    for root in candidates:
        try:
            rd = root / "runs"
            if rd.is_dir():
                runs.extend([p for p in rd.iterdir() if p.is_dir()])
        except Exception:
            continue
    if not runs:
        return None
    # Choose most recently modified, falling back to lexicographic order
    runs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return runs[0]


def _resolve_memory_path() -> Path:
    # Prefer explicit RUN_DIR if provided
    run_dir_env = os.environ.get("RUN_DIR")
    if run_dir_env:
        return Path(run_dir_env).resolve() / "memories" / "memory.jsonl"

    # Try to infer from common roots
    candidates: list[Path] = []
    # Modal sandbox default root
    candidates.append(Path("/root/sandbox"))
    # Local macOS automount pattern (see tools/bash.py)
    try:
        repo_parent = Path(os.path.dirname(os.path.abspath(os.getcwd())))
        candidates.append(repo_parent / "sandbox")
    except Exception:
        pass
    # Current working directory as a last resort
    candidates.append(Path.cwd())

    latest = _find_latest_run_dir(candidates)
    if latest is None:
        # Create a dev run dir under CWD
        latest = Path.cwd() / "runs" / "dev"
    return latest / "memories" / "memory.jsonl"


def memory_file_path() -> Path:
    return _resolve_memory_path()


class Entity(BaseModel):
    name: str
    entityType: str
    observations: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    from_: str = Field(alias="from")
    to: str
    relationType: str


class KnowledgeGraph(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


class ObservationInput(BaseModel):
    entityName: str
    contents: list[str]


class KnowledgeGraphManager:
    def __init__(self, path_provider: Callable[[], Path]):
        self._path_provider = path_provider

    def _path(self) -> Path:
        return self._path_provider()

    def _load_graph(self) -> KnowledgeGraph:
        path = self._path()
        try:
            text = path.read_text()
        except FileNotFoundError:
            return KnowledgeGraph()

        entities: list[Entity] = []
        relations: list[Relation] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            t = item.get("type")
            if t == "entity":
                entities.append(Entity(**{k: v for k, v in item.items() if k != "type"}))
            elif t == "relation":
                relations.append(Relation(**{k: v for k, v in item.items() if k != "type"}))

        return KnowledgeGraph(entities=entities, relations=relations)

    def _save_graph(self, graph: KnowledgeGraph) -> None:
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for e in graph.entities:
            d = e.model_dump()
            d["type"] = "entity"
            lines.append(json.dumps(d))
        for r in graph.relations:
            d = r.model_dump(by_alias=True)
            d["type"] = "relation"
            lines.append(json.dumps(d))
        path.write_text("\n".join(lines))

    # API methods
    def create_entities(self, entities: list[Entity]) -> list[Entity]:
        graph = self._load_graph()
        existing = {e.name for e in graph.entities}
        new_entities = [e for e in entities if e.name not in existing]
        graph.entities.extend(new_entities)
        self._save_graph(graph)
        return new_entities

    def create_relations(self, relations: list[Relation]) -> list[Relation]:
        graph = self._load_graph()
        existing = {(r.from_, r.to, r.relationType) for r in graph.relations}
        new_rel = [r for r in relations if (r.from_, r.to, r.relationType) not in existing]
        graph.relations.extend(new_rel)
        self._save_graph(graph)
        return new_rel

    def add_observations(self, observations: list[ObservationInput]) -> list[dict[str, Any]]:
        graph = self._load_graph()
        results: list[dict[str, Any]] = []
        by_name = {e.name: e for e in graph.entities}
        for o in observations:
            e = by_name.get(o.entityName)
            if not e:
                raise ValueError(f"Entity with name {o.entityName} not found")
            added: list[str] = []
            seen = set(e.observations)
            for c in o.contents:
                if c not in seen:
                    e.observations.append(c)
                    added.append(c)
                    seen.add(c)
            results.append({"entityName": o.entityName, "addedObservations": added})
        self._save_graph(graph)
        return results

    def delete_entities(self, names: list[str]) -> None:
        graph = self._load_graph()
        names_set = set(names)
        graph.entities = [e for e in graph.entities if e.name not in names_set]
        graph.relations = [
            r for r in graph.relations if r.from_ not in names_set and r.to not in names_set
        ]
        self._save_graph(graph)

    def delete_observations(self, deletions: list[dict[str, Any]]) -> None:
        graph = self._load_graph()
        by_name = {e.name: e for e in graph.entities}
        for d in deletions:
            name = d.get("entityName")
            obs = d.get("observations", [])
            e = by_name.get(name)
            if e:
                remove = set(obs)
                e.observations = [o for o in e.observations if o not in remove]
        self._save_graph(graph)

    def delete_relations(self, relations: list[Relation]) -> None:
        graph = self._load_graph()
        remove = {(r.from_, r.to, r.relationType) for r in relations}
        graph.relations = [
            r for r in graph.relations if (r.from_, r.to, r.relationType) not in remove
        ]
        self._save_graph(graph)

    def read_graph(self) -> KnowledgeGraph:
        return self._load_graph()

    def search_nodes(self, query: str) -> KnowledgeGraph:
        graph = self._load_graph()
        q = query.lower()
        ents = [
            e
            for e in graph.entities
            if (q in e.name.lower())
            or (q in e.entityType.lower())
            or any(q in o.lower() for o in e.observations)
        ]
        ent_names = {e.name for e in ents}
        rels = [r for r in graph.relations if r.from_ in ent_names and r.to in ent_names]
        return KnowledgeGraph(entities=ents, relations=rels)

    def open_nodes(self, names: list[str]) -> KnowledgeGraph:
        graph = self._load_graph()
        names_set = set(names)
        ents = [e for e in graph.entities if e.name in names_set]
        rels = [r for r in graph.relations if r.from_ in names_set and r.to in names_set]
        return KnowledgeGraph(entities=ents, relations=rels)


mcp = FastMCP("memory-server")
_kg = KnowledgeGraphManager(memory_file_path)


@mcp.tool
def read_graph() -> dict[str, Any]:
    """Read the entire knowledge graph."""
    return _kg.read_graph().model_dump(by_alias=True)


@mcp.tool
def create_entities(entities: list[Entity]) -> list[dict[str, Any]]:
    """Create multiple new entities in the knowledge graph."""
    created = _kg.create_entities(entities)
    return [e.model_dump() for e in created]


@mcp.tool
def create_relations(relations: list[Relation]) -> list[dict[str, Any]]:
    """Create relations (active voice) between entities in the graph."""
    created = _kg.create_relations(relations)
    return [r.model_dump(by_alias=True) for r in created]


@mcp.tool
def add_observations(observations: list[ObservationInput]) -> list[dict[str, Any]]:
    """Add new observations to existing entities."""
    return _kg.add_observations(observations)


@mcp.tool
def delete_entities(entityNames: list[str]) -> str:
    """Delete entities (and attached relations) by names."""
    _kg.delete_entities(entityNames)
    return "Entities deleted successfully"


@mcp.tool
def delete_observations(deletions: list[dict[str, Any]]) -> str:
    """Delete observations from entities."""
    _kg.delete_observations(deletions)
    return "Observations deleted successfully"


@mcp.tool
def delete_relations(relations: list[Relation]) -> str:
    """Delete specific relations from the graph."""
    _kg.delete_relations(relations)
    return "Relations deleted successfully"


@mcp.tool
def search_nodes(query: str) -> dict[str, Any]:
    """Search for nodes by matching names, types, or observation text."""
    return _kg.search_nodes(query).model_dump(by_alias=True)


@mcp.tool
def open_nodes(names: list[str]) -> dict[str, Any]:
    """Open specific nodes by names, returning subgraph of those nodes and their relations."""
    return _kg.open_nodes(names).model_dump(by_alias=True)


if __name__ == "__main__":
    mcp.run()
