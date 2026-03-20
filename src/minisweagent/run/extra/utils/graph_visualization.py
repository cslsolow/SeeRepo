#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple
import pickle
import typer

try:
    import networkx as nx
except Exception as e:  # pragma: no cover
    raise RuntimeError("mswea_graph requires networkx to be installed") from e


def _get_digraph_class():
    """Lazy import of graphviz.Digraph — only needed for PNG rendering."""
    try:
        from graphviz import Digraph
        return Digraph
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "mswea_graph requires graphviz python package + system graphviz binary (dot). "
            "Try: pip install graphviz && sudo apt-get install graphviz"
        ) from e


# ===== Constants aligned with loc-agent graph schema =====
NODE_TYPE_DIRECTORY = "directory"
NODE_TYPE_FILE = "file"
NODE_TYPE_CLASS = "class"
NODE_TYPE_FUNCTION = "function"

EDGE_TYPE_CONTAINS = "contains"
EDGE_TYPE_IMPORTS = "imports"
EDGE_TYPE_INHERITS = "inherits"
EDGE_TYPE_INVOKES = "invokes"

NODE_TYPE_KEY = "type"
EDGE_TYPE_KEY = "type"

NODE_NAME_KEYS = ["name", "path", "fullname", "qualified_name"]

ICON = {
    NODE_TYPE_DIRECTORY: "📁",
    NODE_TYPE_FILE: "📄",
    NODE_TYPE_CLASS: "🧩",
    NODE_TYPE_FUNCTION: "⚙️",
}

PRIORITY = {
    NODE_TYPE_DIRECTORY: 0,
    NODE_TYPE_FILE: 1,
    NODE_TYPE_CLASS: 2,
    NODE_TYPE_FUNCTION: 3,
}


app = typer.Typer(add_completion=False, rich_markup_mode="rich")


@dataclass
class RunContext:
    instance_dir: Optional[Path]
    pkl_path: Path
    out_dir: Path
    log_jsonl: Optional[Path]


def _get_node_name(node_id: Any, data: Dict[str, Any]) -> str:
    for k in NODE_NAME_KEYS:
        v = data.get(k)
        if v:
            return str(v)
    return str(node_id)


def _make_label(node_id: Any, data: Dict[str, Any]) -> str:
    t = data.get(NODE_TYPE_KEY, "unknown")
    name = _get_node_name(node_id, data)
    icon = ICON.get(t, "•")
    return f"""<
    <TABLE BORDER="0" CELLBORDER="0" CELLPADDING="2">
      <TR>
        <TD ALIGN="LEFT">{icon}</TD>
        <TD ALIGN="LEFT">{name}</TD>
      </TR>
    </TABLE>
    >"""


def _child_sort_key(H: "nx.DiGraph", n: Any) -> Tuple[int, str]:
    t = H.nodes[n].get(NODE_TYPE_KEY, "unknown")
    pri = PRIORITY.get(t, 9)
    return (pri, _get_node_name(n, H.nodes[n]).lower())


def _extract_edge_digraph(G: Any, edge_type: str) -> "nx.DiGraph":
    """
    Extract only edges with data['type'] == edge_type. Works for DiGraph/MultiDiGraph.
    Keeps node attributes for all nodes that appear in the extracted subgraph.
    """
    H = nx.DiGraph()

    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        for u, v, _k, d in G.edges(keys=True, data=True):
            if d.get(EDGE_TYPE_KEY) == edge_type:
                H.add_edge(u, v)
    else:
        for u, v, d in G.edges(data=True):
            if d.get(EDGE_TYPE_KEY) == edge_type:
                H.add_edge(u, v)

    # keep node attrs (only for nodes that exist in H)
    for n, d in G.nodes(data=True):
        if n in H:
            H.nodes[n].update(d)

    return H


def _ensure_root(H: "nx.DiGraph", root: str = "/") -> str:
    if root not in H:
        H.add_node(root, **{NODE_TYPE_KEY: NODE_TYPE_DIRECTORY})
        roots = [n for n in H.nodes if H.in_degree(n) == 0 and n != root]
        for r in roots:
            H.add_edge(root, r)
    return root


def _collect_bidir_nodes(
    H: "nx.DiGraph", center: Any, up_depth: int, down_depth: int
) -> Tuple[Set[Any], Dict[Any, int]]:
    if center not in H:
        raise KeyError(f"center node not found: {center!r}")

    nodes: Set[Any] = {center}
    dist: Dict[Any, int] = {center: 0}

    # Upstream
    q = deque([(center, 0)])
    visited_up = {center}
    while q:
        u, dcur = q.popleft()
        if dcur >= up_depth:
            continue
        for p in H.predecessors(u):
            if p in visited_up:
                continue
            visited_up.add(p)
            nodes.add(p)
            dist[p] = min(dist.get(p, 10**9), -(dcur + 1))
            q.append((p, dcur + 1))

    # Downstream
    q = deque([(center, 0)])
    visited_down = {center}
    while q:
        u, dcur = q.popleft()
        if dcur >= down_depth:
            continue
        for v in H.successors(u):
            if v in visited_down:
                continue
            visited_down.add(v)
            nodes.add(v)
            dist[v] = min(dist.get(v, 10**9), dcur + 1)
            q.append((v, dcur + 1))

    return nodes, dist


def _add_edges_with_junction(dot: Digraph, H: "nx.DiGraph", nodes: Set[Any], dist: Dict[Any, int], center: Any = None,center_only=False) -> None:

    children_map: Dict[Any, list[Any]] = {}

    for u in nodes:
        # successors within the selected node set
        childs = [v for v in H.successors(u) if v in nodes]
        if not childs:
            continue

        # NEW: when center_only, only keep edges that touch center
        if center_only:
            if center is None:
                continue
            if u == center:
                # keep all outgoing edges from center to nodes
                childs = [v for v in childs if v != center]  # avoid self-loop if any
            else:
                # only keep edges u -> center
                childs = [v for v in childs if v == center]

            if not childs:
                continue

        childs.sort(key=lambda n: _child_sort_key(H, n))
        children_map[u] = childs

    junction_id = 0
    for par, childs in children_map.items():
        if len(childs) == 1:
            dot.edge(_dot_id(par), _dot_id(childs[0]))
        else:
            junc = f"__junction_{junction_id}"
            junction_id += 1
            dot.node(junc, label="", shape="point", width="0.01", height="0.01")

            # Force junction into the "middle" rank between parent and children.
            # With dist step=2, parent is at D and children at D±2, so junction sits at D±1.
            par_d = dist.get(par)
            if par_d is not None:
                # Default: for edges par -> child, junction should be to the right (rankdir=LR): D+1
                junc_d = par_d + 1
                with dot.subgraph(name=f"rank_{junc_d}__junc_{junction_id}") as s:
                    s.attr(rank="same")
                    s.node(junc)

            dot.edge(_dot_id(par), junc)
            for c in childs:
                dot.edge(junc, _dot_id(c))


def _render_bidir_edges(
    H: "nx.DiGraph",
    center: Any,
    out_prefix: Path,
    up_depth: int,
    down_depth: int,
    graph_name: str,
    edge_type:str,
) -> Path:
    nodes, dist = _collect_bidir_nodes(H, center, up_depth=up_depth, down_depth=down_depth)
    Digraph = _get_digraph_class()
    dot = Digraph(graph_name, format="png", engine="dot")
    dot.attr(
        "graph",
        rankdir="LR",
        splines="ortho",
        nodesep="0.25",
        ranksep="0.9",
        newrank="true",
        ordering="out",
    )
    dot.attr("node", shape="plain", fontname="Helvetica", fontsize="12")
    dot.attr("edge", color="#777777", arrowhead="normal", arrowsize="0.7")

    for n in nodes:
        if n == center:
            dot.node(
                _dot_id(n),
                label=_make_label(n, H.nodes[n]),
                style="filled",
                fillcolor="#FFF2CC",
            )
        else:
            dot.node(_dot_id(n), label=_make_label(n, H.nodes[n]))

    # For relational edges (imports, inherits, invokes), usually we want to see relationships
    # strictly tied to the center node (ego graph) to reduce noise.
    # For structural edges (contains), seeing the whole subtree is usually fine.
    is_relational = (edge_type != EDGE_TYPE_CONTAINS)
    _add_edges_with_junction(dot, H, nodes, dist, center=center, center_only=is_relational)

    layers: Dict[int, list[Any]] = {}
    for n, d in dist.items():
        layers.setdefault(d, []).append(n)
    for d, ns in layers.items():
        ns.sort(key=lambda x: _child_sort_key(H, x))
        with dot.subgraph(name=f"rank_{d}") as s:
            s.attr(rank="same")
            for n in ns:
                s.node(_dot_id(n))

    out_path_str = dot.render(str(out_prefix), cleanup=True)
    return Path(out_path_str)


def _sanitize_filename(s: str) -> str:
    # keep it stable & filesystem-safe
    s = s.strip()
    s = s.replace(os.sep, "__")
    s = re.sub(r"[^a-zA-Z0-9._\-]+", "_", s)
    return s[:180] if len(s) > 180 else s

def _dot_id(x: Any) -> str:
    # Stable, safe DOT id (no special chars)
    return f"n_{abs(hash(x))}"

def _get_project_root() -> Optional[Path]:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "mini-swe-agent":
            return parent
    return None

def _resolve_context(
    pkl: Optional[Path],
    out_dir: Optional[Path],
    log_jsonl: Optional[Path],
) -> RunContext:
    instance_dir_env = os.environ.get("MSWEA_INSTANCE_DIR")
    instance_dir = Path(instance_dir_env) if instance_dir_env else None
    if instance_dir is not None and not instance_dir.is_absolute():
        project_root = _get_project_root()
        if project_root:
            instance_dir = (project_root / instance_dir).resolve()

    # Agents always pass --pkl repo_graph.pkl as a bare filename; treat it as a sentinel
    # that triggers env-var / instance_dir resolution rather than a literal relative path.
    pkl_env = os.environ.get("MSWEA_REPO_GRAPH_PKL")
    is_default_pkl_arg = (
        pkl is not None
        and getattr(pkl, "name", str(pkl)) == "repo_graph.pkl"
        and not (hasattr(pkl, "is_absolute") and pkl.is_absolute())
    )
    if pkl_env:
        pkl_path = Path(pkl_env)
    elif is_default_pkl_arg:
        # Resolve repo_graph.pkl via instance_dir; do not treat it as a literal relative path.
        if instance_dir is not None:
            pkl_path = instance_dir / "repo_graph.pkl"
        else:
            raise typer.BadParameter(
                "Cannot resolve graph pkl. Set MSWEA_REPO_GRAPH_PKL or MSWEA_INSTANCE_DIR when using --pkl repo_graph.pkl."
            )
    elif pkl is not None:
        pkl_path = Path(pkl)
    elif instance_dir is not None:
        pkl_path = instance_dir / "repo_graph.pkl"
    else:
        raise typer.BadParameter(
            "Cannot resolve graph pkl. Provide --pkl or set MSWEA_REPO_GRAPH_PKL or MSWEA_INSTANCE_DIR."
        )

    if out_dir is not None:
        out_dir_path = out_dir
    elif instance_dir is not None:
        out_dir_path = instance_dir / "graph_views"
    else:
        out_dir_path = Path.cwd() / "graph_views"

    out_dir_path.mkdir(parents=True, exist_ok=True)

    if log_jsonl is not None:
        log_path = log_jsonl
    elif instance_dir is not None:
        log_path = instance_dir / "tool_calls.jsonl"
    else:
        log_path = None

    return RunContext(instance_dir=instance_dir, pkl_path=pkl_path, out_dir=out_dir_path, log_jsonl=log_path)


def _render_text_graph(
    H: "nx.DiGraph",
    center: Any,
    up_depth: int,
    down_depth: int,
    edge_type: str,
) -> str:
    """Return a plain-text representation of the bidirectional neighborhood of center."""
    nodes, dist = _collect_bidir_nodes(H, center, up_depth=up_depth, down_depth=down_depth)

    def node_type_str(n: Any) -> str:
        return H.nodes[n].get(NODE_TYPE_KEY, "?") if n in H.nodes else "?"

    def node_label(n: Any) -> str:
        return f"{_get_node_name(n, H.nodes.get(n, {}))} [{node_type_str(n)}]"

    upstream: list[Any] = sorted(
        [n for n, d in dist.items() if d < 0],
        key=lambda n: (dist[n], _get_node_name(n, H.nodes.get(n, {})).lower()),
    )
    downstream_by_dist: Dict[int, list[Any]] = {}
    for n, d in dist.items():
        if d > 0:
            downstream_by_dist.setdefault(d, []).append(n)

    lines: list[str] = []
    lines.append(f"[Graph: {edge_type} | center: {center} | up={up_depth} down={down_depth}]")
    lines.append("")

    if upstream:
        upstream_label = {
            EDGE_TYPE_IMPORTS: "files that import center",
            EDGE_TYPE_INVOKES: "callers of center",
            EDGE_TYPE_INHERITS: "subclasses of center",
            EDGE_TYPE_CONTAINS: "parent containers",
        }.get(edge_type, "upstream")
        lines.append(f"▲ UPSTREAM ({upstream_label}):")
        for n in upstream:
            lines.append(f"  {node_label(n)}")
        lines.append("")

    lines.append("● CENTER:")
    lines.append(f"  {node_label(center)}")

    if downstream_by_dist:
        downstream_label = {
            EDGE_TYPE_IMPORTS: "modules imported by center",
            EDGE_TYPE_INVOKES: "callees of center",
            EDGE_TYPE_INHERITS: "base classes of center",
            EDGE_TYPE_CONTAINS: "contained children",
        }.get(edge_type, "downstream")
        lines.append("")
        lines.append(f"▼ DOWNSTREAM ({downstream_label}):")
        for d in sorted(downstream_by_dist.keys()):
            ns = sorted(downstream_by_dist[d], key=lambda n: _child_sort_key(H, n))
            if len(downstream_by_dist) > 1:
                lines.append(f"  [depth={d}]")
            for n in ns:
                indent = "    " if len(downstream_by_dist) > 1 else "  "
                lines.append(f"{indent}{node_label(n)}")

    return "\n".join(lines)


@app.command()
def main(
    node: str = typer.Option(None, "--node", help="Center node id. Default '/' (repo root)."),
    up_depth: int = typer.Option(1, "--up-depth", min=0, help="Ancestor hop depth."),
    down_depth: int = typer.Option(1, "--down-depth", min=0, help="Descendant hop depth."),
    pkl: Optional[Path] = typer.Option(None, "--pkl", exists=False, help="Path to repo_graph.pkl"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Output directory for images."),
    log_jsonl: Optional[Path] = typer.Option(None, "--log-jsonl", help="Append tool call info to this JSONL."),
    quiet: bool = typer.Option(False, "--quiet", help="Only print the output image path."),
    edge_type: str = typer.Option(EDGE_TYPE_CONTAINS,"--edge-type",help="Which edge type to visualize: contains/imports/inherits/invokes",),
    text: bool = typer.Option(False, "--text", help="Output plain-text graph instead of PNG image."),
) -> None:
    """
    Visualize the repo 'contains' subgraph around a node (bidir neighborhood).
    Designed as a mini-swe-agent callable CLI tool.
    """
    t0 = time.time()
    ctx = _resolve_context(pkl=pkl, out_dir=out_dir, log_jsonl=log_jsonl)

    if not ctx.pkl_path.exists():
        raise typer.BadParameter(f"Graph pkl not found: {ctx.pkl_path}")

    with open(ctx.pkl_path, "rb") as f:
        G = pickle.load(f)

    H = _extract_edge_digraph(G, edge_type=edge_type)
    if node in G and node not in H:
        H.add_node(node, **G.nodes[node])
    # Normalize "." to "/" so users can pass "." to refer to the repo root.
    if node == ".":
        node = "/"
    center = _ensure_root(H, "/")
    # if user passed node != '/', use it as center (but keep ensure_root for '/' only)
    center = node if node else "/"
    if center == "/":
        _ensure_root(H, "/")

    '''    # auto-fix direction: want center's descendants to be "contained"
    if center in H:
        out_deg = H.out_degree(center)
        in_deg = H.in_degree(center)
        if out_deg == 0 and in_deg > 0:
            H = H.reverse(copy=False)
    '''
    safe = _sanitize_filename(center)

    if text:
        text_output = _render_text_graph(
            H, center=center, up_depth=up_depth, down_depth=down_depth, edge_type=edge_type
        )
        record = {
            "tool": "mswea_graph",
            "ts": time.time(),
            "pkl": str(ctx.pkl_path),
            "node": center,
            "edge_type": edge_type,
            "up_depth": up_depth,
            "down_depth": down_depth,
            "text_mode": True,
            "runtime_sec": round(time.time() - t0, 4),
            "contains_nodes_total": int(H.number_of_nodes()),
            "contains_edges_total": int(H.number_of_edges()),
        }
        if ctx.instance_dir is not None:
            record["instance_dir"] = str(ctx.instance_dir)
        if ctx.log_jsonl is not None:
            ctx.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(ctx.log_jsonl, "a", encoding="utf-8") as wf:
                wf.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(text_output)
        return

    out_prefix = ctx.out_dir / f"{safe}__{edge_type}__u{up_depth}__d{down_depth}"
    out_path = _render_bidir_edges(
        H, center=center, out_prefix=out_prefix,
        up_depth=up_depth, down_depth=down_depth,
        graph_name=f"{edge_type}_bidir", edge_type=edge_type,
    )

    record = {
        "tool": "mswea_graph",
        "ts": time.time(),
        "pkl": str(ctx.pkl_path),
        "node": center,
        "edge_type": edge_type,
        "up_depth": up_depth,
        "down_depth": down_depth,
        "out_png": str(out_path),
        "runtime_sec": round(time.time() - t0, 4),
        "contains_nodes_total": int(H.number_of_nodes()),
        "contains_edges_total": int(H.number_of_edges()),
    }
    if ctx.instance_dir is not None:
        record["instance_dir"] = str(ctx.instance_dir)

    if ctx.log_jsonl is not None:
        ctx.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(ctx.log_jsonl, "a", encoding="utf-8") as wf:
            wf.write(json.dumps(record, ensure_ascii=False) + "\n")

    if quiet:
        print(str(out_path))
    else:
        print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()