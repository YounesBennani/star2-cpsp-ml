"""Methods pipeline schematic for STAR-2 paper.
Repeated nested CV with RFE in the inner loop, refit, dev-set threshold optimisation,
held-out outer test, 50-fold aggregation, and final full-data refit for SHAP.
Arrow-free version — flow is implied by top-to-bottom and left-to-right layout."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "Data/Star 2/Figure_pipeline_schematic.png"

# ── Palette ────────────────────────────────────────────────────────────────
C_DATA   = "#E8EEF7"
C_OUTER  = "#D6E4F3"
C_TRAIN  = "#FFE9D3"
C_INNER  = "#FFD9A8"
C_REFIT  = "#FFE0BD"
C_DEV    = "#FFF4D6"
C_TEST   = "#D9F2D9"
C_AGG    = "#E2D9F2"
C_SHAP   = "#F2D9E2"
EDGE     = "#2E2E2E"

fig, ax = plt.subplots(figsize=(10.5, 14.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 16.0)
ax.set_aspect("equal"); ax.axis("off")

def box(x, y, w, h, title, body=None, fc=C_DATA, title_size=10.5, body_size=9, lw=1.4):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                       linewidth=lw, edgecolor=EDGE, facecolor=fc)
    ax.add_patch(p)
    if body is None:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=EDGE)
    else:
        ax.text(x + w/2, y + h - 0.22, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold", color=EDGE)
        ax.text(x + w/2, y + h/2 - 0.20, body, ha="center", va="center",
                fontsize=body_size, color=EDGE)

def label(x, y, text, size=8.5, italic=False, color=EDGE, ha="center", va="center", weight="normal"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size,
            fontstyle="italic" if italic else "normal", color=color, fontweight=weight)

def panel_label(x, y, letter):
    """Sub-panel marker (a, b, c, ...) — placed at top-left of each panel."""
    ax.text(x, y, letter, fontsize=15, fontweight="bold", color="#1F497D",
            ha="left", va="top")

# ════════════════════════════════════════════════════════════════════════════
# Title
ax.text(5, 15.65, "Repeated nested cross-validation pipeline",
        ha="center", va="center", fontsize=13.5, fontweight="bold", color="#1F497D")

# ── (a) DATA + OUTER LOOP ───────────────────────────────────────────────────
panel_label(0.05, 15.30, "(a)")

box(0.55, 13.85, 9.05, 1.50,
    "Analytic cohort",
    "N = 144 adolescents  ·  47 CPSP+ (32.6%) / 97 CPSP−\n"
    "14 preoperative features + 150 diary aggregates  =  164 candidate features per patient per day d",
    fc=C_DATA, title_size=11)

box(0.55, 12.55, 9.05, 0.90,
    "Outer loop  ·  RepeatedStratifiedKFold (5 splits × 10 repeats) = 50 outer folds",
    fc=C_OUTER, title_size=10.5)

# Note below the outer loop
label(5.075, 12.20, "one outer fold shown below — repeated 50 times", italic=True, size=8.7)

# Partition labels
label(3.50, 11.85, "Outer training partition  ·  80%, ≈ 115 patients",
      size=10.5, weight="bold")
label(8.10, 11.85, "Outer test partition  ·  20%, ≈ 29 patients",
      size=10.5, weight="bold")

# ── OUTER TRAINING container (orange backdrop) ──────────────────────────────
TRAIN_Y, TRAIN_H = 4.50, 7.00
p = FancyBboxPatch((0.55, TRAIN_Y), 6.05, TRAIN_H,
                   boxstyle="round,pad=0.04,rounding_size=0.10",
                   linewidth=1.4, edgecolor=EDGE, facecolor=C_TRAIN)
ax.add_patch(p)

# Vertical layout inside training container (top → bottom):
INNER_Y, INNER_H = 9.60, 1.80     # top: 11.40 → 9.60
REFIT_Y, REFIT_H = 8.10, 1.10     # top: 9.20 → 8.10
DEV_Y,   DEV_H   = 4.70, 2.95     # top: 7.65 → 4.70

# ── (b) INNER CV ────────────────────────────────────────────────────────────
panel_label(0.05, INNER_Y + INNER_H - 0.10, "(b)")

box(0.80, INNER_Y, 5.55, INNER_H,
    "Inner loop  ·  5-fold stratified CV  ·  GridSearchCV",
    "• Recursive Feature Elimination  (features ∈ {10, 15, 20})\n"
    "• Hyperparameter grid  (LogReg-ElasticNet, RF, XGBoost)\n"
    "• Scoring: AUROC  →  best configuration selected",
    fc=C_INNER, title_size=9.6, body_size=8.5)

# ── (c) REFIT ───────────────────────────────────────────────────────────────
panel_label(0.05, REFIT_Y + REFIT_H - 0.10, "(c)")

box(0.80, REFIT_Y, 5.55, REFIT_H,
    "Refit",
    "Best configuration trained on the outer training partition\n"
    "(excluding the development set)",
    fc=C_REFIT, title_size=10, body_size=8.5)

# ── (d) DEV-SET THRESHOLD OPTIMISATION ──────────────────────────────────────
panel_label(0.05, DEV_Y + DEV_H - 0.10, "(d)")

box(0.80, DEV_Y, 5.55, DEV_H,
    "Development set  ·  20% of outer train  ·  threshold optimisation",
    "Held out from the inner CV and the refit step.\n"
    "Trained model predicts daily CPSP probabilities on these patients;\n"
    "(τ_pos, τ_neg) selected on a 41 × 41 grid that minimises\n"
    "cost = C_FP · FP + C_FN · FN on dev-set early decisions.",
    fc=C_DEV, title_size=9.6, body_size=8.5)

# ── (e) HELD-OUT TEST ───────────────────────────────────────────────────────
# (e) label placed just ABOVE the test box (above box top, below partition label)
panel_label(6.75, 11.78, "(e)")

box(6.75, TRAIN_Y, 2.85, TRAIN_H,
    "HELD OUT",
    "from every step:\n• preprocessing\n• feature selection\n• hyperparameter tuning\n"
    "• classifier fitting\n• threshold optimisation",
    fc=C_TEST, title_size=12, body_size=9)

# Fold-level metrics (output of (e), placed below both training and test columns)
box(2.30, 2.80, 5.40, 1.05,
    "Fold-level metrics on held-out patients",
    "AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1,\n"
    "decision day per patient, % fallback at day 30",
    fc=C_OUTER, title_size=9.6, body_size=8.5)

# ── (f) AGGREGATION + SHAP REFIT ────────────────────────────────────────────
panel_label(1.85, 2.30, "(f)")

label(5, 2.55, "repeated across the 50 outer folds", italic=True, size=8.9)

box(2.30, 1.05, 5.40, 1.20,
    "Aggregated estimate across 50 outer folds",
    "Mean ± 95% CI  =  mean  ±  1.96 · SD / √50\n"
    "Primary metric for all reported AUROCs",
    fc=C_AGG, title_size=10.2, body_size=9)

# SHAP refit (separate step) — title + body to fit text comfortably in the box
box(2.30, 0.05, 5.40, 0.80,
    "Final refit on all 144 patients",
    "(best hyperparameter configuration · used only for SHAP-based interpretation)",
    fc=C_SHAP, title_size=10, body_size=8.5)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()

from PIL import Image
im = Image.open(OUT)
print(f"Saved -> {OUT}")
print(f"Dimensions: {im.size[0]}x{im.size[1]} px = {im.size[0]/300:.1f}x{im.size[1]/300:.1f} in @300dpi")
