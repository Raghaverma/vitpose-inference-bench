"""Graph-first layout for the two box-and-arrow technical diagrams
(architecture.py's pipeline+optimization diagram, stage_breakdown.py's
methodology diagram).

Nodes and edges are declared as plain data (rank = sequential position, lane
= branch index) and coordinates are DERIVED from that -- x = rank * dx,
y = lane * dy -- rather than hand-tuned per box. This keeps both diagrams
visually consistent by construction and means inserting a real TensorRT box
later (once Stage 3 is measured) is a data edit, not a re-layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from visualizations._style import BASELINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE, UNMEASURED_GREY

BOX_W, BOX_H = 1.9, 0.62


@dataclass
class Node:
    id: str
    label: str
    rank: float
    lane: float
    sublabel: str | None = None
    style: str = "default"  # "default" | "measured" | "unmeasured" | "boundary"
    color: str | None = None
    box_w: float = BOX_W
    box_h: float = BOX_H


@dataclass
class Edge:
    src: str
    dst: str
    label: str | None = None
    style: str = "default"  # "default" | "unmeasured"


def layout(nodes: list[Node], dx: float = 2.3, dy: float = 1.05) -> dict[str, tuple[float, float]]:
    return {n.id: (n.rank * dx, n.lane * dy) for n in nodes}


def _box_style(style: str, color: str | None):
    if style == "unmeasured":
        return dict(facecolor="none", edgecolor=UNMEASURED_GREY, linestyle=(0, (4, 2)),
                    linewidth=1.4, text_color=INK_MUTED)
    if style == "boundary":
        return dict(facecolor="#fff6ee", edgecolor=color or INK_PRIMARY, linestyle="solid",
                    linewidth=2.0, text_color=INK_PRIMARY)
    if style == "measured":
        return dict(facecolor=(color or "#eef3fb"), edgecolor=color or BASELINE, linestyle="solid",
                    linewidth=1.3, text_color="white" if color else INK_PRIMARY)
    return dict(facecolor="#f4f4f2", edgecolor=BASELINE, linestyle="solid", linewidth=1.1,
                text_color=INK_PRIMARY)


def render_node(ax, node: Node, pos: dict[str, tuple[float, float]]) -> None:
    x, y = pos[node.id]
    s = _box_style(node.style, node.color)
    box = FancyBboxPatch((x - node.box_w / 2, y - node.box_h / 2), node.box_w, node.box_h,
                          boxstyle="round,pad=0.02,rounding_size=0.08",
                          facecolor=s["facecolor"], edgecolor=s["edgecolor"],
                          linewidth=s["linewidth"], linestyle=s["linestyle"], zorder=3)
    ax.add_patch(box)
    ax.text(x, y, node.label, ha="center", va="center", fontsize=9.3,
            fontweight="bold" if node.style == "boundary" else "normal",
            color=s["text_color"], zorder=4)
    if node.sublabel:
        # Opaque background, not just a higher zorder: a crossing edge line is
        # drawn UNDER the text glyphs either way, but without a solid fill
        # behind it the line still shows through the gaps between letterforms
        # (e.g. a diagonal edge landing right under this box reads as cutting
        # through the digits of a number here) -- this fully occludes it.
        ax.text(x, y - node.box_h / 2 - 0.16, node.sublabel, ha="center", va="top", fontsize=7.8,
                color=INK_SECONDARY, zorder=4,
                style="italic" if node.style == "unmeasured" else "normal",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none"))


def _rect_boundary_point(cx: float, cy: float, box_w: float, box_h: float,
                          toward_x: float, toward_y: float) -> tuple[float, float]:
    """Point where the ray from (cx, cy) toward (toward_x, toward_y) exits the
    box's rectangle -- computed in DATA coordinates, not points, so it's
    correct regardless of figure size/DPI (matplotlib's shrinkA/shrinkB are in
    points and can't track a box's actual data-space extent across diagrams
    with different scales -- this sidesteps that mismatch entirely)."""
    ddx, ddy = toward_x - cx, toward_y - cy
    if ddx == 0 and ddy == 0:
        return cx, cy
    hw, hh = box_w / 2, box_h / 2
    candidates = []
    if ddx != 0:
        candidates.append(hw / abs(ddx))
    if ddy != 0:
        candidates.append(hh / abs(ddy))
    t = min(candidates)
    return cx + ddx * t, cy + ddy * t


def render_edge(ax, edge: Edge, pos: dict[str, tuple[float, float]],
                 nodes_by_id: dict[str, Node], gap_pts: float = 3) -> None:
    x0, y0 = pos[edge.src]
    x1, y1 = pos[edge.dst]
    src_node, dst_node = nodes_by_id[edge.src], nodes_by_id[edge.dst]
    sx, sy = _rect_boundary_point(x0, y0, src_node.box_w, src_node.box_h, x1, y1)
    ex, ey = _rect_boundary_point(x1, y1, dst_node.box_w, dst_node.box_h, x0, y0)
    color = UNMEASURED_GREY if edge.style == "unmeasured" else INK_SECONDARY
    linestyle = (0, (4, 2)) if edge.style == "unmeasured" else "solid"
    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.3,
                                 linestyle=linestyle, shrinkA=gap_pts, shrinkB=gap_pts,
                                 connectionstyle="arc3,rad=0.0", mutation_scale=14),
                zorder=2)
    if edge.label:
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        ax.text(mx, my + 0.15, edge.label, ha="center", va="bottom", fontsize=8, color=INK_SECONDARY,
                zorder=4, bbox=dict(boxstyle="round,pad=0.15", facecolor="#fcfcfb",
                                     edgecolor="none"))


def render_diagram(ax, nodes: list[Node], edges: list[Edge], dx: float = 2.3, dy: float = 1.05) -> None:
    pos = layout(nodes, dx, dy)
    nodes_by_id = {n.id: n for n in nodes}
    for e in edges:
        render_edge(ax, e, pos, nodes_by_id)
    for n in nodes:
        render_node(ax, n, pos)
    ax.set_xlim(min(x for x, _ in pos.values()) - 1.4, max(x for x, _ in pos.values()) + 1.4)
    ax.set_ylim(min(y for _, y in pos.values()) - 0.9, max(y for _, y in pos.values()) + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
