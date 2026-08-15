"""Horizontal study timeline for STAR-2: T1 (preoperative), T2 (30-day postop
diary), T4 (6-month outcome). Shows when each assessment occurs and what is
collected."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "Data/Star 2/Figure_study_timeline.png"

# Palette
C_T1   = "#D6E4F3"   # preop — blue
C_T2   = "#FFE9D3"   # postop diary — orange
C_T4   = "#E2D9F2"   # outcome — purple
C_SURG = "#D9F2D9"   # surgery marker — green
EDGE   = "#2E2E2E"

fig, ax = plt.subplots(figsize=(13, 5.0))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6)
ax.set_aspect("equal")
ax.axis("off")


# --------- TIME AXIS (horizontal line at y = 2.30) ---------
AXIS_Y = 2.30
ax.plot([0.4, 12.6], [AXIS_Y, AXIS_Y], color="#444", lw=1.6, zorder=1)

# Tick marks + time labels (below the line)
def tick(x, label_top, label_bottom=""):
    ax.plot([x, x], [AXIS_Y - 0.10, AXIS_Y + 0.10], color="#444", lw=1.4, zorder=2)
    ax.text(x, AXIS_Y - 0.30, label_top, ha="center", va="top", fontsize=9.5, color="#222")
    if label_bottom:
        ax.text(x, AXIS_Y - 0.70, label_bottom, ha="center", va="top",
                fontsize=8.5, fontstyle="italic", color="#555")

# Anchors on the axis
X_T1     = 1.6     # preop visit
X_SURG   = 3.5     # surgery (day 0)
X_T2_END = 6.6     # end of 30-day diary
X_T4     = 11.0    # 6-month assessment

tick(X_T1,     "T1",  "preoperative visit")
tick(X_SURG,   "Day 0", "surgery")
tick(X_T2_END, "Day 30", "end of diary")
tick(X_T4,     "T4",  "6 months post-surgery")

# Visual break: two short slashes to indicate time compression
for cx in [(X_T2_END + X_T4)/2 - 0.25, (X_T2_END + X_T4)/2 + 0.25]:
    ax.plot([cx - 0.10, cx + 0.10], [AXIS_Y - 0.12, AXIS_Y + 0.12],
            color="#444", lw=1.4)

# --------- SURGERY MARKER (vertical line + label) ---------
ax.plot([X_SURG, X_SURG], [AXIS_Y, AXIS_Y + 0.50], color="#2E7D32", lw=1.8, zorder=3)
ax.plot(X_SURG, AXIS_Y, "v", color="#2E7D32", markersize=11, zorder=4)

# --------- T1 BOX (preop) above axis ---------
def box(x, y, w, h, title, body, fc):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                       linewidth=1.4, edgecolor=EDGE, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w/2, y + h - 0.22, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=EDGE)
    ax.text(x + w/2, y + h/2 - 0.22, body, ha="center", va="center",
            fontsize=8.7, color=EDGE)

# Connecting "leader" from each box down to its tick on the axis
def leader(x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color="#888", lw=0.9, ls=":", zorder=1)

# T1 box — minimal
box(0.6, 3.40, 2.2, 1.10,
    "T1",
    "Preoperative\nbaseline\nassessment",
    fc=C_T1)
leader(X_T1, AXIS_Y + 0.10, X_T1, 3.40)

# T2 box — minimal
box(3.70, 3.40, 2.80, 1.10,
    "T2",
    "Daily postoperative diary\n(days 1–30)",
    fc=C_T2)
# Daily-tick markers above diary region
for d in range(30):
    dx = 3.75 + (d / 29) * 2.70
    ax.plot(dx, AXIS_Y + 0.15, ".", color="#B85C00", markersize=3, alpha=0.85)

# T4 box — minimal
box(9.80, 3.40, 2.50, 1.10,
    "T4",
    "Outcome\nassessment\n(CPSP at 6 months)",
    fc=C_T4)
leader(X_T4, AXIS_Y + 0.10, X_T4, 3.40)

# Surgery label
ax.text(X_SURG, 0.70, "SURGERY", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="#2E7D32")
ax.text(X_SURG, 0.40, "posterior spinal fusion", ha="center", va="center",
        fontsize=8.5, fontstyle="italic", color="#555")

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()

from PIL import Image
im = Image.open(OUT)
print(f"Saved -> {OUT}")
print(f"Dimensions: {im.size[0]}x{im.size[1]} px = {im.size[0]/300:.1f}x{im.size[1]/300:.1f} in @300dpi")
