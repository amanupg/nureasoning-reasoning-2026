"""Contact sheet of question boxes drawn on key frames (inputs only: question text + images), to check that
the stated normalized 2D boxes sit on objects in the same way on test and validation.
usage: python src/box_sheet.py <split> <subcategory> <n> <out.jpg>"""
import glob, json, os, random, re, sys
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camgeo
from vlm_data import load_questions, clip_images
split, sub, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
Q = load_questions(split); items = [(c, q) for c, qs in Q.items() for q in qs if q["subcategory"] == sub]
random.Random(0).shuffle(items); tiles = []
for c, q in items:
    m = re.search(r"in the (.+?) camera with normalized 2D box \[?(\d+), (\d+), (\d+), (\d+)", q["question"])
    if not m: continue
    key = camgeo.NAME2KEY.get(m.group(1).replace(" ", "_")); ims = clip_images(split, c)
    if not key or not ims: continue
    p = [k for name, _, k in ims if f"/{key}/" in k]
    if not p: continue
    im = Image.open(p[0]).convert("RGB"); W, H = im.size
    x1, y1, x2, y2 = [int(v) / 1000 for v in m.groups()[1:]]
    d = ImageDraw.Draw(im); d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=(255, 0, 0), width=max(2, W // 300))
    cx, cy = (x1 + x2) / 2 * W, (y1 + y2) / 2 * H; s = max((x2 - x1) * W, (y2 - y1) * H, 48) * 2.5; s = min(s, max(W, H))
    tile = im.crop((int(cx - s), int(cy - s), int(cx + s), int(cy + s))).resize((256, 256))
    cat = (re.search(r"the ([\w ]+?) (?:object )?in the", q["question"]) or [None, "?"])[1]
    xy = (re.search(r"\[x, y\] = \[(-?[\d.]+), (-?[\d.]+)\]", q["question"]) or [None, "", ""])
    dd = f"{(float(xy[1])**2 + float(xy[2])**2)**0.5:.0f}m" if xy[1] else ""
    ImageDraw.Draw(tile).rectangle([0, 0, 255, 14], fill=(0, 0, 0)); ImageDraw.Draw(tile).text((3, 2), f"{len(tiles)} {m.group(1)} | {cat} {dd}", fill=(255, 255, 0)); tiles.append(tile)
    if len(tiles) == n: break
cols = 6; rows = (len(tiles) + cols - 1) // cols; sheet = Image.new("RGB", (cols * 256, rows * 256))
for i, t in enumerate(tiles): sheet.paste(t, ((i % cols) * 256, (i // cols) * 256))
sheet.save(out, quality=88); print(out, len(tiles))
