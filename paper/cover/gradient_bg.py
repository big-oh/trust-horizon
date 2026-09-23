"""Background for gradient.tex: navy-to-orange gradient with film grain.

Writes gradient_bg.jpg (US Letter at 200 dpi). The seed is fixed, so the output is reproducible.
Run from this directory: python gradient_bg.py
"""
import numpy as np
from PIL import Image

W, H = 1700, 2200  # 8.5 x 11 in at 200 dpi


def hexrgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255


def lerp_stops(t, stops):
    """Piecewise-linear colour ramp; t has any shape, stops is [(position, hex), ...]."""
    pos = np.array([p for p, _ in stops])
    cols = np.stack([hexrgb(c) for _, c in stops])
    return np.stack([np.interp(t, pos, cols[:, k]) for k in range(3)], axis=-1)


y, x = np.mgrid[0:H, 0:W]
u, v = x / W, y / H

# Vertical ramp: deep navy at the top, steel blue in the middle, dusty lavender at the bottom.
img = lerp_stops(v, [(0.00, "0E2A55"), (0.30, "15407A"), (0.52, "3E73AB"),
                     (0.70, "7F95B8"), (1.00, "C6AEBE")])

# Warm glow rising from the lower middle of the page.
d2 = ((u - 0.60) / 0.50) ** 2 + ((v - 0.93) / 0.42) ** 2
glow = np.exp(-d2)[..., None]
warm = lerp_stops(np.clip(1 - glow[..., 0], 0, 1), [(0.0, "F08F40"), (0.45, "F29E5E"), (1.0, "DDA99C")])
img = img * (1 - 0.95 * glow) + warm * (0.95 * glow)

# Film grain: fine luminance noise plus a little chroma noise.
rng = np.random.default_rng(2026)
img += rng.normal(0, 0.042, (H, W, 1)) + rng.normal(0, 0.010, (H, W, 3))

Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
    "gradient_bg.jpg", quality=88, dpi=(200, 200), optimize=True)
