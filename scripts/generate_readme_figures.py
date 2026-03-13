"""
Generate paper-style architecture figures for ChaoRec README.

Outputs:
  - docs/figures/chao_rec_architecture.svg
  - docs/figures/chao_rec_architecture.png
  - docs/figures/chao_rec_training_pipeline.svg
  - docs/figures/chao_rec_training_pipeline.png
  - docs/figures/sigir/chao_rec_architecture_sigir.{pdf,svg,png}
  - docs/figures/sigir/chao_rec_training_pipeline_sigir.{pdf,svg,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
SIGIR_DIR = FIG_DIR / "sigir"

SIGIR_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
}


def box(ax, x, y, w, h, text, fc="#F8FAFC", ec="#334155", lw=1.4, fs=10, fw="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=fw)
    return patch


def arrow(ax, x1, y1, x2, y2, color="#334155", lw=1.5):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=4, shrinkB=4),
    )


def fig_architecture():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background bands (labels are added separately to avoid overlap)
    box(ax, 0.03, 0.62, 0.22, 0.30, "", fc="#E6F4EA", ec="#2F855A", fs=12, fw="bold")
    box(ax, 0.29, 0.62, 0.18, 0.30, "", fc="#EAF2FF", ec="#2C5282", fs=12, fw="bold")
    box(ax, 0.03, 0.17, 0.44, 0.38, "", fc="#FFF7E6", ec="#B7791F", fs=12, fw="bold")
    box(ax, 0.52, 0.17, 0.30, 0.75, "", fc="#F5EEFF", ec="#6B46C1", fs=12, fw="bold")
    box(ax, 0.84, 0.17, 0.13, 0.75, "", fc="#FFEDED", ec="#C53030", fs=12, fw="bold")

    ax.text(0.14, 0.87, "Data Layer", ha="center", va="center", fontsize=12, fontweight="bold", color="#14532D")
    ax.text(0.38, 0.87, "Config Layer", ha="center", va="center", fontsize=12, fontweight="bold", color="#1E3A8A")
    ax.text(0.25, 0.52, "Training Orchestration", ha="center", va="center", fontsize=12, fontweight="bold", color="#7C2D12")
    ax.text(0.67, 0.88, "Model Zoo", ha="center", va="center", fontsize=12, fontweight="bold", color="#4C1D95")
    ax.text(0.905, 0.88, "Outputs", ha="center", va="center", fontsize=12, fontweight="bold", color="#7F1D1D")

    # Data blocks
    box(
        ax,
        0.05,
        0.74,
        0.18,
        0.14,
        "Data/*\n(baby, beauty,\nclothing, ...)",
        fc="#FFFFFF",
        ec="#2F855A",
    )
    box(
        ax,
        0.05,
        0.64,
        0.18,
        0.08,
        "train.npy / val.npy / test.npy\nv_feat.npy / t_feat.npy",
        fc="#FFFFFF",
        ec="#2F855A",
        fs=9,
    )

    # Config blocks
    box(ax, 0.31, 0.75, 0.14, 0.10, "arg_parser.py\n(CLI args)", fc="#FFFFFF", ec="#2C5282")
    box(ax, 0.31, 0.64, 0.14, 0.08, "Model_YAML/*.yaml\n(grid search space)", fc="#FFFFFF", ec="#2C5282", fs=9)

    # Orchestration blocks
    box(ax, 0.06, 0.42, 0.16, 0.09, "main.py\n(entrypoint)", fc="#FFFFFF", ec="#B7791F")
    box(ax, 0.24, 0.42, 0.20, 0.09, "dataload.py\n(dataset & dataloader)", fc="#FFFFFF", ec="#B7791F")
    box(ax, 0.06, 0.29, 0.38, 0.10, "train_and_evaluate.py\n(train loop / model-specific branches)", fc="#FFFFFF", ec="#B7791F")
    box(ax, 0.06, 0.20, 0.38, 0.07, "metrics.py + utils.py\nRecall / Precision / NDCG", fc="#FFFFFF", ec="#B7791F", fs=9)

    # Model zoo blocks
    box(
        ax,
        0.54,
        0.74,
        0.26,
        0.16,
        "General CF & GNN models\n(BPR, LightGCN, NCL, SimGCL, ...)",
        fc="#FFFFFF",
        ec="#6B46C1",
        fs=10,
    )
    box(
        ax,
        0.54,
        0.53,
        0.26,
        0.18,
        "Multimodal models\n(MMGCN, LGMRec, COHESION,\nDDRec, Grade, GUME, ...)",
        fc="#FFFFFF",
        ec="#6B46C1",
        fs=10,
    )
    box(
        ax,
        0.54,
        0.36,
        0.26,
        0.14,
        "Diffusion / Transformer variants\n(DiffRec, CF_Diff, DiffMM,\nGFormer, LightGT, ...)",
        fc="#FFFFFF",
        ec="#6B46C1",
        fs=10,
    )
    box(ax, 0.54, 0.22, 0.26, 0.10, "54 models total\n(aligned with Model_YAML)", fc="#FFFFFF", ec="#6B46C1", fs=10)

    # Outputs blocks
    box(ax, 0.86, 0.68, 0.09, 0.14, "log/*.log\ntraining traces", fc="#FFFFFF", ec="#C53030", fs=9)
    box(ax, 0.86, 0.49, 0.09, 0.14, "Best metrics\n@K = [5,10,20]", fc="#FFFFFF", ec="#C53030", fs=9)
    box(ax, 0.86, 0.30, 0.09, 0.14, "Best hyperparams\n(from YAML sweep)", fc="#FFFFFF", ec="#C53030", fs=9)

    # Arrows
    arrow(ax, 0.23, 0.78, 0.31, 0.80)  # data -> args
    arrow(ax, 0.23, 0.68, 0.31, 0.68)  # data -> yaml
    arrow(ax, 0.38, 0.75, 0.14, 0.50)  # args -> main
    arrow(ax, 0.38, 0.64, 0.33, 0.50)  # yaml -> main
    arrow(ax, 0.22, 0.46, 0.24, 0.46)  # main -> dataload
    arrow(ax, 0.32, 0.42, 0.25, 0.34)  # dataload -> train/eval
    arrow(ax, 0.22, 0.42, 0.22, 0.34)  # main -> train/eval
    arrow(ax, 0.44, 0.34, 0.54, 0.58)  # train/eval -> model zoo
    arrow(ax, 0.80, 0.74, 0.86, 0.75)  # model -> output logs
    arrow(ax, 0.80, 0.56, 0.86, 0.56)  # model -> output metrics
    arrow(ax, 0.80, 0.40, 0.86, 0.37)  # model -> output best hp
    arrow(ax, 0.44, 0.24, 0.86, 0.56)  # metrics -> output

    ax.text(
        0.5,
        0.97,
        "Figure 1. ChaoRec System Architecture",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.94,
        "Unified data-model-training pipeline for 54 recommendation models",
        ha="center",
        va="top",
        fontsize=10,
        color="#334155",
    )

    return fig


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(14, 6.8), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.97,
        "Figure 2. Training and Evaluation Pipeline in ChaoRec",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.935,
        "Main loop and model-specific branches used in main.py + train_and_evaluate.py",
        ha="center",
        va="top",
        fontsize=10,
        color="#334155",
    )

    # Main flow
    box(ax, 0.04, 0.72, 0.18, 0.12, "1) Parse CLI args\n(arg_parser.py)", fc="#EEF2FF", ec="#4338CA", fs=10)
    box(ax, 0.27, 0.72, 0.20, 0.12, "2) Load YAML search space\n(Model_YAML/{Model}.yaml)", fc="#EEF2FF", ec="#4338CA", fs=10)
    box(ax, 0.52, 0.72, 0.22, 0.12, "3) Load data & dataloaders\n(dataload.py)", fc="#ECFDF5", ec="#047857", fs=10)
    box(ax, 0.78, 0.72, 0.18, 0.12, "4) Build model instance\n(model_constructors)", fc="#F5F3FF", ec="#6D28D9", fs=10)

    box(ax, 0.10, 0.50, 0.32, 0.14, "5) Hyperparameter combination loop\n(cartesian product of YAML values)", fc="#FFF7ED", ec="#C2410C", fs=10)
    box(ax, 0.47, 0.50, 0.22, 0.14, "6) Train by branch\nstandard / LightGT /\nDiffMM / DiffRec / CF_Diff / MHRec", fc="#FFF7ED", ec="#C2410C", fs=9)
    box(ax, 0.73, 0.50, 0.22, 0.14, "7) Evaluate on val/test\nRecall, Precision, NDCG", fc="#FEF2F2", ec="#B91C1C", fs=10)

    box(ax, 0.12, 0.26, 0.30, 0.13, "8) Select best config\nby Recall@20", fc="#F0FDFA", ec="#0F766E", fs=10)
    box(ax, 0.48, 0.26, 0.22, 0.13, "9) Save best metrics\nand parameters", fc="#F0FDFA", ec="#0F766E", fs=10)
    box(ax, 0.74, 0.26, 0.20, 0.13, "10) Write log file\nlog/{Model}_{Dataset}.log", fc="#F0FDFA", ec="#0F766E", fs=10)

    # Side note blocks
    box(ax, 0.04, 0.06, 0.44, 0.13, "Reusable preprocessing scripts:\n- dualgnn-gen-u-u-matrix.py\n- gen_hypergraph_u_i.py", fc="#FFFFFF", ec="#64748B", fs=9)
    box(ax, 0.54, 0.06, 0.42, 0.13, "Outputs consumed by README and experiments:\n- final top-k metrics\n- selected hyperparameters\n- per-run logs", fc="#FFFFFF", ec="#64748B", fs=9)

    # Arrows
    arrow(ax, 0.22, 0.78, 0.27, 0.78)
    arrow(ax, 0.47, 0.78, 0.52, 0.78)
    arrow(ax, 0.74, 0.78, 0.78, 0.78)

    arrow(ax, 0.87, 0.72, 0.32, 0.64)
    arrow(ax, 0.26, 0.72, 0.26, 0.64)
    arrow(ax, 0.42, 0.57, 0.47, 0.57)
    arrow(ax, 0.69, 0.57, 0.73, 0.57)

    arrow(ax, 0.26, 0.50, 0.26, 0.39)
    arrow(ax, 0.84, 0.50, 0.30, 0.39)
    arrow(ax, 0.42, 0.32, 0.48, 0.32)
    arrow(ax, 0.70, 0.32, 0.74, 0.32)

    return fig


def save_fig(fig, stem):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    svg = FIG_DIR / f"{stem}.svg"
    png = FIG_DIR / f"{stem}.png"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg, png


def save_fig_sigir(fig, stem):
    SIGIR_DIR.mkdir(parents=True, exist_ok=True)
    pdf = SIGIR_DIR / f"{stem}.pdf"
    svg = SIGIR_DIR / f"{stem}.svg"
    png = SIGIR_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, svg, png


def fig_architecture_sigir():
    # ACM SIGIR two-column friendly full-width figure (~7 inches)
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    palette = {
        "data": ("#0F766E", "#ECFDF5"),
        "config": ("#1D4ED8", "#EEF2FF"),
        "exec": ("#B45309", "#FFF7ED"),
        "model": ("#6D28D9", "#F5F3FF"),
        "out": ("#BE123C", "#FFF1F2"),
        "arrow": "#334155",
    }

    def sbox(x, y, w, h, t, fs=7.2, bold=False, dashed=False, ec="#111111", fc="#FFFFFF"):
        box(
            ax,
            x,
            y,
            w,
            h,
            t,
            fc=fc,
            ec=ec,
            lw=0.9,
            fs=fs,
            fw="bold" if bold else "normal",
        )
        if dashed:
            ax.patches[-1].set_linestyle("--")

    # Layer containers
    sbox(0.02, 0.14, 0.17, 0.72, "", dashed=True, ec=palette["data"][0], fc=palette["data"][1])
    sbox(0.21, 0.14, 0.16, 0.72, "", dashed=True, ec=palette["config"][0], fc=palette["config"][1])
    sbox(0.39, 0.14, 0.25, 0.72, "", dashed=True, ec=palette["exec"][0], fc=palette["exec"][1])
    sbox(0.66, 0.14, 0.22, 0.72, "", dashed=True, ec=palette["model"][0], fc=palette["model"][1])
    sbox(0.90, 0.14, 0.08, 0.72, "", dashed=True, ec=palette["out"][0], fc=palette["out"][1])

    ax.text(0.105, 0.84, "Data", ha="center", va="center", fontsize=8, fontweight="bold", color=palette["data"][0])
    ax.text(0.29, 0.84, "Config", ha="center", va="center", fontsize=8, fontweight="bold", color=palette["config"][0])
    ax.text(0.515, 0.84, "Execution", ha="center", va="center", fontsize=8, fontweight="bold", color=palette["exec"][0])
    ax.text(0.77, 0.84, "Model Families", ha="center", va="center", fontsize=8, fontweight="bold", color=palette["model"][0])
    ax.text(0.94, 0.84, "Outputs", ha="center", va="center", fontsize=8, fontweight="bold", color=palette["out"][0])

    # Main blocks
    sbox(0.04, 0.55, 0.13, 0.24, "Data/*\n7 datasets\n+ features", ec=palette["data"][0], fc="#FFFFFF")
    sbox(0.04, 0.24, 0.13, 0.22, "train/val/test\nuser_item_dict\nv_feat/t_feat", fs=6.8, ec=palette["data"][0], fc="#FFFFFF")

    sbox(0.23, 0.55, 0.12, 0.24, "arg_parser.py\nCLI args", ec=palette["config"][0], fc="#FFFFFF")
    sbox(0.23, 0.24, 0.12, 0.22, "Model_YAML/*\nsearch space", ec=palette["config"][0], fc="#FFFFFF")

    sbox(0.41, 0.60, 0.21, 0.18, "main.py\nentrypoint", fs=7.4, ec=palette["exec"][0], fc="#FFFFFF")
    sbox(0.41, 0.40, 0.21, 0.15, "dataload.py", fs=7.4, ec=palette["exec"][0], fc="#FFFFFF")
    sbox(0.41, 0.21, 0.21, 0.14, "train_and_evaluate.py\n+ metrics.py / utils.py", fs=6.9, ec=palette["exec"][0], fc="#FFFFFF")

    sbox(0.68, 0.62, 0.18, 0.16, "General CF/GNN\n(BPR, LightGCN, NCL, ...)", fs=6.9, ec=palette["model"][0], fc="#FFFFFF")
    sbox(0.68, 0.42, 0.18, 0.16, "Multimodal\n(MMGCN, COHESION,\nDDRec, Grade, GUME, ...)", fs=6.7, ec=palette["model"][0], fc="#FFFFFF")
    sbox(0.68, 0.22, 0.18, 0.16, "Diffusion/Transformer\n(DiffRec, CF_Diff,\nDiffMM, GFormer, ...)", fs=6.6, ec=palette["model"][0], fc="#FFFFFF")

    sbox(0.915, 0.60, 0.05, 0.18, "log", ec=palette["out"][0], fc="#FFFFFF")
    sbox(0.915, 0.40, 0.05, 0.16, "best\nmetrics", fs=6.8, ec=palette["out"][0], fc="#FFFFFF")
    sbox(0.915, 0.22, 0.05, 0.14, "best\nparams", fs=6.8, ec=palette["out"][0], fc="#FFFFFF")

    # Arrows
    arrow(ax, 0.17, 0.67, 0.23, 0.67, lw=1.0, color=palette["arrow"])
    arrow(ax, 0.17, 0.35, 0.23, 0.35, lw=1.0, color=palette["arrow"])
    arrow(ax, 0.35, 0.67, 0.41, 0.69, lw=1.0, color=palette["arrow"])
    arrow(ax, 0.35, 0.35, 0.41, 0.48, lw=1.0, color=palette["arrow"])
    arrow(ax, 0.52, 0.60, 0.52, 0.55, lw=1.0, color=palette["arrow"])
    arrow(ax, 0.52, 0.40, 0.52, 0.35, lw=1.0, color=palette["arrow"])
    arrow(ax, 0.62, 0.69, 0.68, 0.69, lw=1.0, color=palette["arrow"])
    arrow(ax, 0.62, 0.48, 0.68, 0.50, lw=1.0, color=palette["arrow"])
    arrow(ax, 0.62, 0.28, 0.68, 0.30, lw=1.0, color=palette["arrow"])
    arrow(ax, 0.86, 0.70, 0.915, 0.69, lw=1.0, color=palette["arrow"])
    arrow(ax, 0.86, 0.50, 0.915, 0.48, lw=1.0, color=palette["arrow"])
    arrow(ax, 0.86, 0.30, 0.915, 0.29, lw=1.0, color=palette["arrow"])

    ax.text(0.5, 0.965, "Figure 1. ChaoRec architecture", ha="center", va="top", fontsize=8.5, fontweight="bold")
    return fig


def fig_pipeline_sigir():
    # ACM SIGIR two-column friendly full-width figure (~7 inches)
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    palette = {
        "prep": ("#1D4ED8", "#EEF2FF"),
        "train": ("#B45309", "#FFF7ED"),
        "result": ("#0F766E", "#ECFDF5"),
        "note": ("#475569", "#F8FAFC"),
        "arrow": "#334155",
    }

    def sbox(x, y, w, h, t, fs=7.0, bold=False, ec="#111111", fc="#FFFFFF"):
        box(
            ax,
            x,
            y,
            w,
            h,
            t,
            fc=fc,
            ec=ec,
            lw=0.9,
            fs=fs,
            fw="bold" if bold else "normal",
        )

    # Top row
    sbox(0.03, 0.70, 0.16, 0.16, "1) Parse args", ec=palette["prep"][0], fc="#FFFFFF")
    sbox(0.22, 0.70, 0.18, 0.16, "2) Load YAML\nsearch space", ec=palette["prep"][0], fc="#FFFFFF")
    sbox(0.43, 0.70, 0.18, 0.16, "3) Load data", ec=palette["prep"][0], fc="#FFFFFF")
    sbox(0.64, 0.70, 0.16, 0.16, "4) Build model", ec=palette["prep"][0], fc="#FFFFFF")
    sbox(0.83, 0.70, 0.14, 0.16, "5) Train & eval\nper config", ec=palette["train"][0], fc="#FFFFFF")

    # Bottom row
    sbox(0.07, 0.42, 0.28, 0.16, "6) Select best config\nby Recall@20", ec=palette["result"][0], fc="#FFFFFF")
    sbox(0.39, 0.42, 0.24, 0.16, "7) Save best metrics\nand parameters", ec=palette["result"][0], fc="#FFFFFF")
    sbox(0.67, 0.42, 0.26, 0.16, "8) Write log file\nlog/{Model}_{Dataset}.log", ec=palette["result"][0], fc="#FFFFFF")

    # Notes
    sbox(0.03, 0.12, 0.44, 0.20, "Model branches in train_and_evaluate.py:\nstandard / LightGT / DiffMM / DiffRec /\nCF_Diff / MHRec", fs=6.8, ec=palette["note"][0], fc=palette["note"][1])
    sbox(0.52, 0.12, 0.45, 0.20, "Optional preprocessing:\n- dualgnn-gen-u-u-matrix.py\n- gen_hypergraph_u_i.py", fs=6.8, ec=palette["note"][0], fc=palette["note"][1])

    # Arrows
    arrow(ax, 0.19, 0.78, 0.22, 0.78, lw=1.0, color=palette["arrow"])
    arrow(ax, 0.40, 0.78, 0.43, 0.78, lw=1.0, color=palette["arrow"])
    arrow(ax, 0.61, 0.78, 0.64, 0.78, lw=1.0, color=palette["arrow"])
    arrow(ax, 0.80, 0.78, 0.83, 0.78, lw=1.0, color=palette["arrow"])

    # stage transition from top row to bottom row (orthogonal routing for readability)
    arrow(ax, 0.90, 0.70, 0.90, 0.62, lw=1.0, color=palette["arrow"])
    arrow(ax, 0.90, 0.62, 0.20, 0.62, lw=1.0, color=palette["arrow"])
    arrow(ax, 0.20, 0.62, 0.20, 0.58, lw=1.0, color=palette["arrow"])

    arrow(ax, 0.35, 0.50, 0.39, 0.50, lw=1.0, color=palette["arrow"])
    arrow(ax, 0.63, 0.50, 0.67, 0.50, lw=1.0, color=palette["arrow"])

    ax.text(0.50, 0.885, "grid search: repeat steps (3-5) for each hyperparameter combination", ha="center", va="bottom", fontsize=6.4)

    ax.text(0.5, 0.965, "Figure 2. ChaoRec training pipeline", ha="center", va="top", fontsize=8.5, fontweight="bold")
    return fig


def main():
    arch = fig_architecture()
    p1 = save_fig(arch, "chao_rec_architecture")

    flow = fig_pipeline()
    p2 = save_fig(flow, "chao_rec_training_pipeline")

    with plt.rc_context(SIGIR_RCPARAMS):
        arch_sigir = fig_architecture_sigir()
        p3 = save_fig_sigir(arch_sigir, "chao_rec_architecture_sigir")

        flow_sigir = fig_pipeline_sigir()
        p4 = save_fig_sigir(flow_sigir, "chao_rec_training_pipeline_sigir")

    print("Generated:")
    for p in [*p1, *p2, *p3, *p4]:
        print("-", p)


if __name__ == "__main__":
    main()
