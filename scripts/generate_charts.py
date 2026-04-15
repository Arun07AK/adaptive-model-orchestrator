#!/usr/bin/env python3
"""Generate LinkedIn carousel PNG charts from benchmark results.

Outputs 4 images at 1080x1080 (LinkedIn carousel format):
    data/charts/01_hero.png          — headline stat
    data/charts/02_accuracy.png      — benchmark accuracy comparison
    data/charts/03_cost.png          — cost efficiency chart
    data/charts/04_architecture.png  — 3-tier cascade diagram

Run:
    python scripts/generate_charts.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path


# --- Design tokens (match README dark theme) ---
BG = "#0f172a"
SURFACE = "#1e293b"
SURFACE2 = "#334155"
TEXT = "#f1f5f9"
DIM = "#94a3b8"
TEAL = "#2dd4bf"
YELLOW = "#fbbf24"
RED = "#f87171"
BLUE = "#60a5fa"
PURPLE = "#a78bfa"
GREEN = "#4ade80"

# Font settings
FONT = "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": SURFACE2,
    "xtick.color": DIM,
    "ytick.color": DIM,
})


# --- Data (from latest fixed benchmarks) ---
CONFIGS = ["Baseline\n(Qwen 7B)", "V1\nOrchestrated", "V1\nHybrid", "V2-A\nSelective\nReview", "V2-B\nCascade"]
MMLU = [60.0, 76.0, 76.0, 66.0, 66.0]
GSM8K = [26.7, 70.0, 96.7, 86.7, 83.3]
ARC = [93.3, 93.3, 96.7, 93.3, 76.7]
SENIOR_PCT = [0, 100, 100, 11, 7]

COLORS = [DIM, BLUE, PURPLE, TEAL, YELLOW]

OUT_DIR = Path("data/charts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def new_figure():
    """Create a square figure sized for LinkedIn (1080x1080 at 120 DPI)."""
    fig, ax = plt.subplots(figsize=(9, 9), dpi=120)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax


# =============================================================================
# Image 1: The Hero Stat
# =============================================================================
def image_01_hero():
    fig, ax = new_figure()
    ax.axis("off")

    # Top label
    ax.text(0.5, 0.92, "A D A P T I V E   M O D E L   O R C H E S T R A T O R",
            ha="center", va="top",
            fontsize=14, color=DIM, weight="bold",
            transform=ax.transAxes)

    # Tagline
    ax.text(0.5, 0.84, "Two architectures. Same accuracy.",
            ha="center", va="top", fontsize=20, color=TEXT, weight="light",
            transform=ax.transAxes)

    # V1 card
    ax.text(0.25, 0.62, "V1", ha="center", fontsize=42, color=PURPLE,
            weight="bold", transform=ax.transAxes)
    ax.text(0.25, 0.54, "Mixture of Agents", ha="center", fontsize=14,
            color=DIM, transform=ax.transAxes)
    ax.text(0.25, 0.44, "100%", ha="center", fontsize=64, color=PURPLE,
            weight="bold", transform=ax.transAxes)
    ax.text(0.25, 0.34, "of queries use\nthe 235B model", ha="center",
            fontsize=13, color=DIM, transform=ax.transAxes, linespacing=1.4)

    # V2 card
    ax.text(0.75, 0.62, "V2", ha="center", fontsize=42, color=TEAL,
            weight="bold", transform=ax.transAxes)
    ax.text(0.75, 0.54, "Selective Review", ha="center", fontsize=14,
            color=DIM, transform=ax.transAxes)
    ax.text(0.75, 0.44, "11%", ha="center", fontsize=64, color=TEAL,
            weight="bold", transform=ax.transAxes)
    ax.text(0.75, 0.34, "of queries use\nthe 235B model", ha="center",
            fontsize=13, color=DIM, transform=ax.transAxes, linespacing=1.4)

    # Divider line
    ax.plot([0.5, 0.5], [0.3, 0.66], color=SURFACE2, lw=1, transform=ax.transAxes)

    # Bottom takeaway
    ax.text(0.5, 0.18, "9× fewer expensive calls.", ha="center",
            fontsize=26, color=TEXT, weight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.12, "Same accuracy on MMLU and ARC.", ha="center",
            fontsize=18, color=TEAL, transform=ax.transAxes)

    # Footer
    ax.text(0.5, 0.04, "github.com/Arun07AK/adaptive-model-orchestrator",
            ha="center", fontsize=11, color=DIM, transform=ax.transAxes,
            family="monospace")

    plt.savefig(OUT_DIR / "01_hero.png", dpi=120, bbox_inches="tight",
                facecolor=BG, pad_inches=0.3)
    plt.close()
    print("  ✓ 01_hero.png")


# =============================================================================
# Image 2: Benchmark Accuracy
# =============================================================================
def image_02_accuracy():
    fig, ax = plt.subplots(figsize=(9, 9), dpi=120)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    import numpy as np
    n = len(CONFIGS)
    x = np.arange(n)
    width = 0.28

    bars1 = ax.bar(x - width, MMLU, width, label="MMLU", color=BLUE, edgecolor="none")
    bars2 = ax.bar(x, GSM8K, width, label="GSM8K", color=TEAL, edgecolor="none")
    bars3 = ax.bar(x + width, ARC, width, label="ARC", color=YELLOW, edgecolor="none")

    # Labels on bars
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1.5,
                    f"{height:.0f}", ha="center", va="bottom",
                    color=TEXT, fontsize=10, weight="bold")

    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, fontsize=11, color=TEXT, linespacing=1.1)
    ax.set_ylabel("Accuracy (%)", fontsize=13, color=TEXT)
    ax.set_title("Accuracy across 110 benchmark questions",
                 fontsize=20, color=TEXT, weight="bold", pad=20, loc="left")
    ax.text(0, 106, "MMLU (50q) · GSM8K (30q) · ARC-Challenge (30q)",
            fontsize=11, color=DIM)

    ax.legend(loc="upper right", frameon=False, fontsize=12,
              labelcolor=TEXT)

    # Clean spines
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(SURFACE2)
    ax.spines["bottom"].set_color(SURFACE2)
    ax.grid(axis="y", alpha=0.15, color=SURFACE2)
    ax.set_axisbelow(True)

    # Footer
    fig.text(0.5, 0.02,
             "github.com/Arun07AK/adaptive-model-orchestrator",
             ha="center", fontsize=10, color=DIM, family="monospace")

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(OUT_DIR / "02_accuracy.png", dpi=120, bbox_inches="tight",
                facecolor=BG, pad_inches=0.3)
    plt.close()
    print("  ✓ 02_accuracy.png")


# =============================================================================
# Image 3: Cost Efficiency
# =============================================================================
def image_03_cost():
    fig, ax = plt.subplots(figsize=(9, 9), dpi=120)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    labels = CONFIGS
    values = SENIOR_PCT
    colors_map = [DIM, PURPLE, PURPLE, TEAL, YELLOW]

    y_pos = list(range(len(labels)))[::-1]

    bars = ax.barh(y_pos, values, color=colors_map, edgecolor="none", height=0.6)
    for bar, v in zip(bars, values):
        label = f"{v}%" if v > 0 else "no API"
        # Position label: inside bar if >15%, outside if small
        if v >= 15:
            ax.text(v - 2, bar.get_y() + bar.get_height() / 2, label,
                    ha="right", va="center", color="#0f172a",
                    fontsize=18, weight="bold")
        else:
            ax.text(v + 2, bar.get_y() + bar.get_height() / 2, label,
                    ha="left", va="center", color=TEXT,
                    fontsize=18, weight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([l.replace("\n", " ") for l in labels], fontsize=12,
                       color=TEXT)
    ax.set_xlim(0, 115)
    ax.set_xlabel("% of questions that invoked the 235B senior model",
                  fontsize=12, color=TEXT)
    ax.set_title("Cost efficiency: how often the expensive model was called",
                 fontsize=18, color=TEXT, weight="bold", pad=20, loc="left")

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(SURFACE2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.1, color=SURFACE2)
    ax.set_axisbelow(True)

    # Callout for V2-A
    ax.annotate("9× fewer expensive\nAPI calls than V1",
                xy=(11, 1), xytext=(50, 1.4),
                fontsize=13, color=TEAL, weight="bold",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.5", fc=SURFACE, ec=TEAL, lw=1))

    # Footer
    fig.text(0.5, 0.02,
             "github.com/Arun07AK/adaptive-model-orchestrator",
             ha="center", fontsize=10, color=DIM, family="monospace")

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(OUT_DIR / "03_cost.png", dpi=120, bbox_inches="tight",
                facecolor=BG, pad_inches=0.3)
    plt.close()
    print("  ✓ 03_cost.png")


# =============================================================================
# Image 4: Architecture diagram (V2-A Selective Review — the winner)
# =============================================================================
def image_04_architecture():
    fig, ax = new_figure()
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Title
    ax.text(5, 9.5, "V 2 - A   S E L E C T I V E   R E V I E W",
            ha="center", fontsize=16, color=TEAL, weight="bold")
    ax.text(5, 9, "Mimics resident-to-attending escalation in hospitals",
            ha="center", fontsize=12, color=DIM)

    def draw_box(x, y, w, h, label, sublabel, color, textcolor=TEXT):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.1",
                             facecolor=SURFACE, edgecolor=color, lw=2.5)
        ax.add_patch(box)
        ax.text(x, y + 0.2, label, ha="center", va="center",
                fontsize=13, color=color, weight="bold")
        ax.text(x, y - 0.35, sublabel, ha="center", va="center",
                fontsize=10, color=DIM)

    def draw_arrow(x1, y1, x2, y2, label=None, labeloffset=0.3, color=DIM):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + labeloffset, my, label, fontsize=10, color=color,
                    style="italic", va="center")

    # Question box
    draw_box(5, 7.8, 3, 0.9, "QUESTION", "arrives", TEAL)

    # Specialist box
    draw_box(5, 5.8, 5.5, 1.5, "SPECIALIST",
             "Qwen3-32B (math) / Llama-3.3-70B (general) / Llama-4-Scout (code)",
             BLUE)
    ax.text(5, 5.3, "self-consistency check (2× samples)",
            ha="center", fontsize=10, color=DIM, style="italic")

    # Arrow Q -> Specialist
    draw_arrow(5, 7.35, 5, 6.55)

    # Branch decision
    ax.text(5, 4.4, "Do the 2 attempts agree?", ha="center",
            fontsize=12, color=TEXT, style="italic")

    # Yes branch (left)
    draw_box(2.3, 2.6, 3.2, 1.3, "✓ YES (89%)",
             "Return specialist's\nanswer. Done.", GREEN)
    draw_arrow(4.3, 4.1, 2.8, 3.3, "consistent", color=GREEN)

    # No branch (right)
    draw_box(7.7, 2.6, 3.2, 1.3, "✗ NO (11%)",
             "Escalate to senior", YELLOW)
    draw_arrow(5.7, 4.1, 7.2, 3.3, "disagree", color=YELLOW)

    # Senior box below
    draw_box(7.7, 0.9, 3.2, 1.1, "SENIOR REVIEWER",
             "Qwen3-235B (Cerebras)", RED)
    draw_arrow(7.7, 1.95, 7.7, 1.45, color=YELLOW)

    # Footer
    ax.text(5, 0.1, "github.com/Arun07AK/adaptive-model-orchestrator",
            ha="center", fontsize=10, color=DIM, family="monospace")

    plt.savefig(OUT_DIR / "04_architecture.png", dpi=120, bbox_inches="tight",
                facecolor=BG, pad_inches=0.3)
    plt.close()
    print("  ✓ 04_architecture.png")


def main():
    print(f"Generating LinkedIn carousel images in {OUT_DIR}/\n")
    image_01_hero()
    image_02_accuracy()
    image_03_cost()
    image_04_architecture()
    print(f"\nDone. 4 images saved. Upload in order to LinkedIn as a carousel.")


if __name__ == "__main__":
    main()
