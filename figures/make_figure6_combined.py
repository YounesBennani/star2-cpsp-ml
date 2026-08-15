"""Build combined Figure 6: (a) SHAP beeswarm + (b) SHAP bar, side by side with
panel labels. Stitches the two existing high-res SHAP PNGs (no re-computation)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DIR = "Best_model_RF"
OUT = "Results/figures/Figure6_combined.png"
os.makedirs("Results/figures", exist_ok=True)

img_a = mpimg.imread(os.path.join(DIR, "SHAP_beeswarm.png"))
img_b = mpimg.imread(os.path.join(DIR, "SHAP_bar.png"))

# both images are ~0.83 aspect (w/h); two equal panels in a 11.5x7 frame ≈ 0.82 each → minimal whitespace
fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.5, 7.0))
for ax, img in zip((axa, axb), (img_a, img_b)):
    ax.imshow(img)
    ax.axis("off")
for ax, letter in zip((axa, axb), ["a", "b"]):
    ax.text(0.01, 1.01, letter, transform=ax.transAxes, fontsize=20,
            fontweight="bold", va="bottom", ha="left")

fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.04)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close(fig)

from PIL import Image
im = Image.open(OUT)
print(f"Saved -> {OUT}")
print(f"Dimensions: {im.size[0]}x{im.size[1]} px = {im.size[0]/300:.1f}x{im.size[1]/300:.1f} in @300dpi")
