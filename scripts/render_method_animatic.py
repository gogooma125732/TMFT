"""Draw an illustrative, animated token-masking storyboard for the README."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
INK = "#222222"
MUTED = "#5b5b5b"
RULE = "#bababa"
BLUE = "#315e92"
RUST = "#af5a43"
PALE_BLUE = "#e9f0f7"
PALE_RUST = "#f8eee9"
PAPER = "#ffffff"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Times New Roman Bold.ttf" if bold else "Times New Roman.ttf"
    path = FONT_DIR / name
    if not path.exists():
        path = Path(font_manager.findfont("DejaVu Serif:style=bold" if bold else "DejaVu Serif"))
    return ImageFont.truetype(str(path), size)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = RULE) -> None:
    draw.line([start, end], fill=color, width=3)
    x, y = end
    draw.polygon([(x, y), (x - 5, y - 9), (x + 5, y - 9)], fill=color)


def card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], active: bool,
         accent: str, title: str, detail: str, selected: str) -> None:
    draw.rounded_rectangle(box, radius=13, fill=PALE_BLUE if active and accent == BLUE else
                           PALE_RUST if active else PAPER,
                           outline=accent if active else RULE, width=3 if active else 2)
    x, y, _, _ = box
    draw.text((x + 20, y + 14), title, font=font(27, True), fill=INK)
    draw.text((x + 20, y + 53), detail, font=font(22), fill=MUTED)
    draw.text((x + 20, y + 88), selected, font=font(22, True), fill=accent)


def frame(stage: int) -> Image.Image:
    im = Image.new("RGB", (1400, 760), PAPER)
    d = ImageDraw.Draw(im)
    d.text((82, 35), "FIGURE 1", font=font(19, True), fill=BLUE)
    d.text((82, 66), "From sensitive spans to masked loss", font=font(42, True), fill=INK)
    d.text((82, 117), "One synthetic example, animated from input to masked labels", font=font(23), fill=MUTED)
    d.text((1170, 54), f"0{stage + 1} / 05", font=font(23, True), fill=BLUE if stage < 4 else RUST)
    d.line([(82, 155), (1318, 155)], fill=INK, width=2)

    d.text((82, 179), "1  INPUT TOKENS", font=font(22, True), fill=INK)
    tokens = ["Please", "email", "Ada", "at", "ada@example.test", "today"]
    widths = [175, 175, 130, 100, 360, 175]
    x = 82
    token_boxes = []
    for token, width in zip(tokens, widths):
        box = (x, 217, x + width, 275)
        selected = token in {"Ada", "ada@example.test"}
        active = (stage in {1, 3, 4} and selected) or stage == 0
        fill = PALE_BLUE if active and selected else "#f6f6f6" if active else PAPER
        edge = BLUE if active and selected else RULE
        d.rounded_rectangle(box, radius=9, fill=fill, outline=edge, width=3 if active and selected else 2)
        d.text((x + width / 2, 246), token, anchor="mm", font=font(23, selected and active), fill=INK)
        token_boxes.append(box)
        x += width + 14
    d.text((82, 294), "2  SENSITIVE-TOKEN SIGNALS", font=font(22, True), fill=INK)
    card(d, (82, 329, 683, 466), stage == 1, BLUE,
         "NER + regular expressions", "Find names and email/phone spans", "Selected: Ada · ada@example.test")
    card(d, (717, 329, 1318, 466), stage == 2, RUST,
         "Online MIA signal", "Compare current vs. frozen base model", "Selected: ada@example.test")

    # The two signals converge before loss masking.
    d.line([(382, 466), (382, 493), (700, 493)], fill=RULE, width=3)
    d.line([(1018, 466), (1018, 493), (700, 493)], fill=RULE, width=3)
    arrow(d, (700, 493), (700, 516), BLUE if stage == 3 else RULE)
    d.rounded_rectangle((342, 518, 1058, 579), radius=10,
                        fill=PALE_BLUE if stage == 3 else PAPER,
                        outline=BLUE if stage == 3 else RULE,
                        width=3 if stage == 3 else 2)
    d.text((700, 549), "3  UNION + MASK-RATIO CAP  →  {Ada, email address}",
           anchor="mm", font=font(24, True), fill=INK)
    arrow(d, (700, 579), (700, 605), RUST if stage == 4 else RULE)

    d.text((82, 606), "4  TRAINING LABELS", font=font(22, True), fill=INK)
    x = 82
    for token, width in zip(tokens, widths):
        selected = token in {"Ada", "ada@example.test"}
        label = "−100" if selected else token
        fill = PALE_RUST if selected else "#f7f7f7"
        edge = RUST if selected and stage == 4 else RULE
        d.rounded_rectangle((x, 641, x + width, 695), radius=8, fill=fill,
                            outline=edge, width=3 if selected and stage == 4 else 2)
        d.text((x + width / 2, 668), label, anchor="mm", font=font(23, selected),
               fill=RUST if selected else INK)
        x += width + 14

    d.line([(82, 714), (1318, 714)], fill=RULE, width=2)
    d.text((82, 727), "Masked labels are excluded from loss; evaluation tracks TER, PPL, and MIA AUC.",
           font=font(19), fill=MUTED)
    return im


def render_method_animatic() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [frame(step) for step in range(5)]
    frames[-1].save(OUT / "method_overview_animatic_v3.png", optimize=True)
    palette_frames = [im.quantize(colors=96) for im in frames]
    palette_frames[0].save(
        OUT / "method_overview_animatic_v3.gif",
        save_all=True,
        append_images=palette_frames[1:],
        duration=[850, 850, 850, 850, 1500],
        loop=0,
        optimize=True,
        disposal=2,
    )


if __name__ == "__main__":
    render_method_animatic()
