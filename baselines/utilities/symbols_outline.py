# utilities/symbols_outline.py
import ast
from typing import Any, Dict, List, Optional, Tuple

# ---------- AST helpers ----------

def _name_of_decorator(dec) -> str:
    def _name(n):
        if isinstance(n, ast.Name):
            return n.id
        if isinstance(n, ast.Attribute):
            base = _name(n.value)
            return f"{base}.{n.attr}" if base else n.attr
        return ""
    if isinstance(dec, ast.Call):
        return _name(dec.func)
    return _name(dec)

def _var_targets(node: ast.AST) -> List[str]:
    names: List[str] = []
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name):
                names.append(t.id)
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        names.append(node.target.id)
    return names

def _node_span(node: ast.AST) -> Tuple[int, int]:
    start = getattr(node, "lineno", None) or 0
    end = getattr(node, "end_lineno", start) or start
    return start, end

def _method_info(fn: ast.AST) -> Dict[str, Any]:
    start, end = _node_span(fn)
    decos = [_name_of_decorator(d) for d in getattr(fn, "decorator_list", [])]
    tags: List[str] = []
    if any(d.endswith("property") for d in decos): tags.append("property")
    if any(d.endswith("classmethod") for d in decos): tags.append("classmethod")
    if any(d.endswith("staticmethod") for d in decos): tags.append("staticmethod")
    return {
        "kind": "func",
        "name": fn.name,
        "start": start,
        "end": end,
        "decorators": decos,
        "tags": tags,
        "children": [],
    }

def _class_children(c: ast.ClassDef) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for n in c.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append(_method_info(n))
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            for nm in _var_targets(n):
                s, e = _node_span(n)
                items.append({"kind": "const", "name": nm, "start": s, "end": e, "children": []})
        # (Inner classes/imports omitted for GitHub-like parity)
    items.sort(key=lambda d: d["start"])
    return items

# ---------- Public API ----------

def build_outline(src: str) -> List[Dict[str, Any]]:
    """
    GitHub-style symbols outline (JSON-friendly list).
    Each item: {kind:'const'|'func'|'class', name, start, end, children:[...]}
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    outline: List[Dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            s, e = _node_span(node)
            outline.append({
                "kind": "class",
                "name": node.name,
                "start": s,
                "end": e,
                "children": _class_children(node),
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            outline.append(_method_info(node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for nm in _var_targets(node):
                s, e = _node_span(node)
                outline.append({"kind": "const", "name": nm, "start": s, "end": e, "children": []})
        # imports and others are ignored to match the sidebar
    outline.sort(key=lambda d: d["start"])
    return outline

def format_outline(outline: List[Dict[str, Any]], *, max_lines: int = 2000) -> str:
    """
    Pretty text block for prompts/logs. Truncates to ~max_lines chars.
    """
    ICON = {"class": "class", "func": "func", "const": "const"}
    lines: List[str] = ["Symbols"]
    def emit(items: List[Dict[str, Any]], indent: int = 0):
        pad = "  " * indent
        for it in items:
            tag = ICON.get(it["kind"], it["kind"])
            lines.append(f"{pad}{tag} {it['name']}  [{it['start']}–{it['end']}]")
            if it.get("children"):
                emit(it["children"], indent + 1)
    emit(outline, 0)
    text = "\n".join(lines)
    return text if len(text) <= max_lines else text[: max_lines - 3] + "..."

def filter_outline_to_range(outline: List[Dict[str, Any]], start: int, end: int) -> List[Dict[str, Any]]:
    """
    Keep only items that overlap [start, end]. For classes, keep class if it overlaps,
    but drop child items outside the range.
    """
    filt: List[Dict[str, Any]] = []
    def overlaps(a: int, b: int) -> bool:
        return not (b < start or a > end)

    for it in outline:
        a, b = it["start"], it["end"]
        if not overlaps(a, b):
            continue
        if it["kind"] == "class":
            kids = []
            for ch in it.get("children", []):
                if overlaps(ch["start"], ch["end"]):
                    kids.append(ch)
            filt.append({**it, "children": kids})
        else:
            filt.append(it)
    return filt

# ---------- File IO helpers (no caching) ----------

def get_outline_for_file(path: str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    with open(path, "r", encoding=encoding) as f:
        src = f.read()
    return build_outline(src)

def render_outline_header(src: str, chunk_range: Optional[Tuple[int, int]] = None) -> str:
    """
    One-call helper: build outline and return formatted header (no caching).
    If chunk_range is provided, the outline is filtered to that range.
    """
    outline = build_outline(src)
    if chunk_range:
        outline = filter_outline_to_range(outline, chunk_range[0], chunk_range[1])
    return format_outline(outline)

def render_outline_header_for_file(path: str, chunk_range: Optional[Tuple[int, int]] = None,
                                   encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding) as f:
        src = f.read()
    return render_outline_header(src, chunk_range)
