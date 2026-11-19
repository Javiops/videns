"""The V. mark as vector icons: tab favicon, touch/PWA icons and the future app icon.

Outlines come from the site's own Inter (fonts/inter-var-latin.woff2, SIL OFL) at weight 900, drawn the
way the VIDENS. wordmark is: stretched to 118% wide, then leaned left until the V's right stroke is
vertical (20.97 deg, measured from the outline here). The period is neither stretched nor leaned, as on
the site. Style B: the site's purple gradient tile with the letter and the dot in white.
Writes SVGs; render_icons.mjs rasterises them and pack_ico.py builds the .ico files.
"""
import math
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
FONT = r"C:/Users/Javier/videns.gg/videns-site/fonts/inter-var-latin.woff2"
SX = 1.18                 # the wordmark's horizontal stretch
INK, DOT = "#f6f4fb", "#b556ff"
# the site's --fill: 135deg across a square tile
GRAD = ('<linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#dcaaff"/><stop offset=".48" stop-color="#b556ff"/>'
        '<stop offset="1" stop-color="#6a1ccf"/></linearGradient>')

font = instantiateVariableFont(TTFont(FONT), {"wght": 900})
gs = font.getGlyphSet()
upem = font["head"].unitsPerEm
cap = font["OS/2"].sCapHeight


def right_slant():
    """tan of the V's right outer edge from vertical once stretched by SX (top-right vs bottom-right)."""
    rec = RecordingPen()
    gs["V"].draw(rec)
    pts = [p for _op, args in rec.value for p in args if isinstance(p, tuple)]
    top = max((p for p in pts if abs(p[1] - cap) < 2), key=lambda p: p[0])
    bottom = max((p for p in pts if abs(p[1]) < 2), key=lambda p: p[0])
    return SX * (top[0] - bottom[0]) / (top[1] - bottom[1])


SKEW = right_slant()


def outline(name, a, c, e, f, s):
    """SVG path of a glyph under x' = a*x + c*y + e, y' = -s*y + f (y-up font units -> y-down canvas)."""
    pen = SVGPathPen(gs)
    gs[name].draw(TransformPen(pen, (a, 0, c, -s, e, f)))
    bp = BoundsPen(gs)
    gs[name].draw(TransformPen(bp, (a, 0, c, -s, e, f)))
    return pen.getCommands(), bp.bounds


def mark(tile, cap_frac, max_w):
    """Paths for 'V.' fitted in a square of side `tile`: cap height cap_frac * tile, shrunk if the pair
    would be wider than max_w * tile (the stretched V is wide), optically centred on cap top..baseline."""
    # The period is tucked in after the V's vertical right stroke, as a kerned "V." pair would be: its ink
    # starts GAP em after the stroke, not after the V's full advance, which reads as a hole.
    GAP = 0.07

    def layout(s):
        _, v_b = outline("V", s * SX, -s * SKEW, 0, 0, s)
        _, p0 = outline("period", s, 0, 0, 0, s)
        dot_x = v_b[2] + GAP * upem * s - p0[0]
        return dot_x, min(v_b[0], p0[0] + dot_x), max(v_b[2], p0[2] + dot_x)

    s = tile * cap_frac / cap
    _, x0, x1 = layout(s)
    s *= min(1.0, tile * max_w / (x1 - x0))
    dot_x, x0, x1 = layout(s)
    dx = (tile - (x1 - x0)) / 2 - x0
    dy = (tile - s * cap) / 2 + s * cap
    v_d, _ = outline("V", s * SX, -s * SKEW, dx, dy, s)
    p_d, _ = outline("period", s, 0, dot_x + dx, dy, s)
    return v_d, p_d


def svg(tile, cap_frac, max_w, radius, bg=True, dot=INK):
    v, p = mark(tile, cap_frac, max_w)
    back = f'<defs>{GRAD}</defs><rect width="{tile}" height="{tile}" rx="{radius}" fill="url(#g)"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {tile} {tile}">{back}'
            f'<path fill="{INK}" d="{v}"/><path fill="{dot}" d="{p}"/></svg>\n')


# rounded gradient tile: the tab favicon, the "any" PWA icons and the desktop app icon
(OUT / "v-mark.svg").write_text(svg(512, 0.56, 0.8, 112), encoding="utf-8")
# full-bleed gradient square, the mark inside the 80% safe circle: maskable PWA icon and Apple touch icon
(OUT / "v-mark-maskable.svg").write_text(svg(512, 0.4, 0.62, 0), encoding="utf-8")
# the bare mark, transparent, in the wordmark's colours (white V, purple dot) for dark surfaces
(OUT / "v-mark-glyph.svg").write_text(svg(512, 0.62, 0.9, 0, bg=False, dot=DOT), encoding="utf-8")
print("lean", round(math.degrees(math.atan(SKEW)), 2), "deg; wrote", sorted(p.name for p in OUT.glob("*.svg")))
