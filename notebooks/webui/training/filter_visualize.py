"""
Утилиты визуализации фильтрации: маски удалённых регионов на скриншоте, side-by-side comparison.
"""

from PIL import Image, ImageDraw, ImageFont


def mask_regions(img, bboxes, fill_color=(255, 255, 255), alpha=255):
    """Возвращает копию img с залитыми bbox-регионами (white по умолчанию).

    bboxes: list of dicts {'x', 'y', 'width', 'height'}
    alpha: 0-255 (255 = полная заливка, ниже — semi-transparent)
    """
    img = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for bbox in bboxes:
        if not bbox:
            continue
        x = float(bbox.get("x", 0))
        y = float(bbox.get("y", 0))
        w = float(bbox.get("width", 0))
        h = float(bbox.get("height", 0))
        if w <= 0 or h <= 0:
            continue
        draw.rectangle(
            [x, y, x + w, y + h],
            fill=(*fill_color, alpha),
        )
    out = Image.alpha_composite(img, overlay).convert("RGB")
    return out


def draw_outlines(img, bboxes, color=(255, 0, 0), width=3):
    """Возвращает копию img с нарисованными outline'ами bbox'ов."""
    img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        if not bbox:
            continue
        x = float(bbox.get("x", 0))
        y = float(bbox.get("y", 0))
        w = float(bbox.get("width", 0))
        h = float(bbox.get("height", 0))
        if w <= 0 or h <= 0:
            continue
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
    return img


def shade_regions(img, bboxes, color=(255, 0, 0), alpha=80):
    """Полупрозрачная заливка регионов (для подсветки 'will be removed')."""
    return mask_regions(img, bboxes, fill_color=color, alpha=alpha)


def side_by_side(img1, img2, label1="Original", label2="Filtered", padding=10, label_height=30):
    """Compose two images side-by-side с подписями."""
    h = max(img1.height, img2.height) + label_height
    w = img1.width + img2.width + padding
    combined = Image.new("RGB", (w, h), (240, 240, 240))
    combined.paste(img1, (0, label_height))
    combined.paste(img2, (img1.width + padding, label_height))
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((10, 5), label1, fill=(0, 0, 0), font=font)
    draw.text((img1.width + padding + 10, 5), label2, fill=(0, 0, 0), font=font)
    return combined


def resize_to_height(img, target_height):
    """Resize image сохраняя aspect ratio до целевой высоты."""
    if img.height == target_height:
        return img
    ratio = target_height / img.height
    new_w = int(img.width * ratio)
    return img.resize((new_w, target_height), Image.LANCZOS)
