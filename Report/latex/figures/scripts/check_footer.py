"""Report any text or image that reaches into the bottom margin / page-number area of a PDF page."""
import sys

import pymupdf

MARGIN_PT = 0.9 * 72  # bottom margin of the geometry settings


def check(path):
    doc = pymupdf.open(path)
    problems = []
    for number, page in enumerate(doc, start=1):
        height = page.rect.height
        limit = height - MARGIN_PT + 2
        blocks = page.get_text("dict")["blocks"]
        footer = [b for b in blocks if b["type"] == 0 and b["bbox"][1] > limit
                  and "".join(s["text"] for l in b["lines"] for s in l["spans"]).strip().isalnum()
                  and len("".join(s["text"] for l in b["lines"] for s in l["spans"]).strip()) <= 5]
        for b in blocks:
            if b in footer:
                continue
            if b["bbox"][3] > limit:
                kind = "image" if b["type"] == 1 else "text"
                text = "" if b["type"] == 1 else "".join(s["text"] for l in b["lines"] for s in l["spans"])[:50]
                problems.append((number, kind, round(b["bbox"][3] - limit, 1), text))
        for image in page.get_image_info():
            if image["bbox"][3] > limit:
                problems.append((number, "image", round(image["bbox"][3] - limit, 1), ""))
    print(f"{path}: {len(doc)} pages, {len(problems)} intrusions into the bottom margin")
    for p in problems:
        print("  page", p)
    return problems


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        check(arg)
