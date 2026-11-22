"""Members for the Alpamayo2-Edge zero-shot VQA check (2026-11-11): 4 frames at 10 Hz (t0-0.3 s .. t0) for all 8
cameras. t0 = key frame 100, or frame 99 when frame 100 duplicates frame 99 (85% of test clips; same files), so the
4 frames are distinct. -> src/edge_members.json {split: [{clip, tar, size, members}]} (format of image_members.json).
usage: python src/edge_members.py --val-n 250 [--test]
"""
import argparse, json, os, random

CAMS = ["front", "front_left", "front_right", "left", "right", "back", "back_left", "back_right"]


def frames(md):
    fr = {f["frame_index"]: f for f in md["frames"]}
    k = 99 if fr[100]["timestamp_us"] == fr[99]["timestamp_us"] else 100
    return [fr[i] for i in range(k - 3, k + 1)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-n", type=int, default=250); ap.add_argument("--test", action="store_true")
    ap.add_argument("--out", default="src/edge_members.json")
    a = ap.parse_args()
    base = json.load(open("src/image_members.json")); out = {}
    for split in ["validation"] + (["test"] if a.test else []):
        tar = {r["clip"]: r for r in base[split]}
        clips = sorted(c for c in tar if os.path.isfile(f"cache/{split}/{c}/metadata.json"))
        if split == "validation":
            clips = [c for c in clips if os.path.isdir(f"vqa_val/{c}")]
            random.Random(0).shuffle(clips); clips = sorted(clips[:a.val_n])
        rows = []
        for c in clips:
            fs = frames(json.load(open(f"cache/{split}/{c}/metadata.json")))
            mem = [f"{c}/{f['sensors']['cameras'][cam]}" for f in fs for cam in CAMS if f["sensors"]["cameras"].get(cam)]
            rows.append({"clip": c, "tar": tar[c]["tar"], "size": tar[c]["size"], "members": mem})
        out[split] = rows
        print(split, len(rows), "clips,", sum(len(r["members"]) for r in rows), "images")
    json.dump(out, open(a.out, "w"))
