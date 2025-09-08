"""
Memory tool container: provides a simple knowledge graph stored in JSONL.
Supports local filesystem and Modal sandbox via LazySandBox, mirroring
patterns used in tools/edit.py.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Any, Annotated

from pydantic import BaseModel, Field, ConfigDict
from .shared import LazySandBox


# Determine default automount path similar to tools/edit.py
if sys.platform == "darwin":
    DEFAULT_AUTOMOUNT = os.path.join(os.path.dirname(os.path.abspath(os.getcwd())), "sandbox")
    os.makedirs(DEFAULT_AUTOMOUNT, exist_ok=True)
else:
    DEFAULT_AUTOMOUNT = "/root/sandbox"
    os.makedirs(DEFAULT_AUTOMOUNT, exist_ok=True)


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


class MemoryContainer:
    """
    Stateful memory/knowledge-graph container. Persists to a JSONL file under
    the selected root directory. Works both locally and inside a Modal sandbox.
    """

    def __init__(
        self,
        sandbox: LazySandBox | None = None,
        automount_path: str = DEFAULT_AUTOMOUNT,
        storage_file: str | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._automount_path = automount_path
        # Resolve storage file
        storage_env = os.environ.get("MEMORY_FILE_PATH")
        raw_path = storage_file or storage_env or "memory.jsonl"
        self._storage_path = str(self._resolve_path(Path(raw_path)))

    # --------- Filesystem helpers (local vs sandbox) ---------
    def _resolve_path(self, path: Path) -> Path:
        p_str = str(path)
        if p_str.startswith(self._automount_path):
            return Path(p_str)
        if p_str.startswith("/root/sandbox"):
            suffix = p_str[len("/root/sandbox"):].lstrip("/")
        else:
            suffix = p_str.lstrip("/") if path.is_absolute() else p_str
        return Path(self._automount_path) / suffix

    def _exists(self, path: str) -> bool:
        if self._sandbox is None:
            return Path(path).exists()
        p = self._sandbox.exec('bash', '-c', f"if test -e {path}; then echo yes; else echo no; fi")
        p.wait()
        return p.stdout.read().strip().endswith("yes")

    def _read_text(self, path: str) -> str:
        if self._sandbox is None:
            try:
                return Path(path).read_text()
            except FileNotFoundError:
                return ""
        if not self._exists(path):
            return ""
        p = self._sandbox.exec('bash', '-c', f"cat {path}")
        p.wait()
        return p.stdout.read()

    def _write_text(self, path: str, content: str) -> None:
        parent = str(Path(path).parent)
        if self._sandbox is None:
            pth = Path(path)
            pth.parent.mkdir(parents=True, exist_ok=True)
            pth.write_text(content)
            return
        heredoc = (
            f"mkdir -p {parent} && "
            f"cat > {path} << 'EOF'\n{content}\nEOF\n"
        )
        p = self._sandbox.exec('bash', '-c', heredoc)
        p.wait()

    # --------- Graph load/save ---------
    def _load_graph(self) -> KnowledgeGraph:
        text = self._read_text(self._storage_path)
        if not text:
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
        lines: list[str] = []
        for e in graph.entities:
            d = e.model_dump()
            d["type"] = "entity"
            lines.append(json.dumps(d))
        for r in graph.relations:
            d = r.model_dump(by_alias=True)
            d["type"] = "relation"
            lines.append(json.dumps(d))
        self._write_text(self._storage_path, "\n".join(lines))

    # --------- Public API (to be registered as tools) ---------
    def read_graph(self) -> Annotated[dict[str, Any], Field(description="Entire knowledge graph with entities and relations")]:
        return self._load_graph().model_dump(by_alias=True)

    def create_entities(
        self,
        entities: Annotated[list[Entity], Field(description="Entities to create (unique by name)")],
    ) -> Annotated[list[dict[str, Any]], Field(description="List of created entities (may be fewer if duplicates were ignored)")]:
        graph = self._load_graph()
        existing = {e.name for e in graph.entities}
        new_entities = [e for e in entities if e.name not in existing]
        graph.entities.extend(new_entities)
        self._save_graph(graph)
        return [e.model_dump() for e in new_entities]

    def create_relations(
        self,
        relations: Annotated[list[Relation], Field(description="Relations to create between existing entities")],
    ) -> Annotated[list[dict[str, Any]], Field(description="List of created relations (deduplicated)")]:
        graph = self._load_graph()
        existing = {(r.from_, r.to, r.relationType) for r in graph.relations}
        new_rel = [r for r in relations if (r.from_, r.to, r.relationType) not in existing]
        graph.relations.extend(new_rel)
        self._save_graph(graph)
        return [r.model_dump(by_alias=True) for r in new_rel]

    def add_observations(
        self,
        observations: Annotated[list[ObservationInput], Field(description="Observations to append to existing entities; duplicates are ignored")],
    ) -> Annotated[list[dict[str, Any]], Field(description="Per-entity results with addedObservations list")]:
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

    def delete_entities(
        self,
        entityNames: Annotated[list[str], Field(description="Names of entities to delete (also removes attached relations)")],
    ) -> Annotated[str, Field(description="Confirmation message")]:
        graph = self._load_graph()
        names_set = set(entityNames)
        graph.entities = [e for e in graph.entities if e.name not in names_set]
        graph.relations = [
            r for r in graph.relations if r.from_ not in names_set and r.to not in names_set
        ]
        self._save_graph(graph)
        return "Entities deleted successfully"

    def delete_observations(
        self,
        deletions: Annotated[list[dict[str, Any]], Field(description="List of objects with entityName and observations to remove")],
    ) -> Annotated[str, Field(description="Confirmation message")]:
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
        return "Observations deleted successfully"

    def delete_relations(
        self,
        relations: Annotated[list[Relation], Field(description="Specific relations to delete (by from, to, and relationType)")],
    ) -> Annotated[str, Field(description="Confirmation message")]:
        graph = self._load_graph()
        remove = {(r.from_, r.to, r.relationType) for r in relations}
        graph.relations = [
            r for r in graph.relations if (r.from_, r.to, r.relationType) not in remove
        ]
        self._save_graph(graph)
        return "Relations deleted successfully"

    def search_nodes(
        self,
        query: Annotated[str, Field(description="Search text to match against entity names, types, and observations")],
    ) -> Annotated[dict[str, Any], Field(description="Subgraph matching the search query")]:
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
        return KnowledgeGraph(entities=ents, relations=rels).model_dump(by_alias=True)

    def open_nodes(
        self,
        names: Annotated[list[str], Field(description="Exact entity names to open in a focused subgraph")],
    ) -> Annotated[dict[str, Any], Field(description="Subgraph containing the requested nodes and their relations")]:
        graph = self._load_graph()
        names_set = set(names)
        ents = [e for e in graph.entities if e.name in names_set]
        rels = [r for r in graph.relations if r.from_ in names_set and r.to in names_set]
        return KnowledgeGraph(entities=ents, relations=rels).model_dump(by_alias=True)
