"""Ego meta-action as a 4-option VLM question (subcategory 'ego_meta_action'), one per clip, from
artifacts/meta_labels_{split}.json (src/meta_labels.py). Ego-history text = inputs only (frames <= 100).
-> artifacts/meta_q_{split}.json {clip: [question]}; read by src/vlm_data.py when META_Q=1.
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTS = {"A": "come to a stop or stay stopped", "B": "decelerate", "C": "keep roughly constant speed", "D": "accelerate"}
C4_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def text(o):
    f = lambda x: "unknown" if x is None else f"{x:.1f}"
    hist = ", ".join(f"{f(o[f'v_m{s}'])} m/s {s} s ago" for s in range(5, 0, -1))
    yr = "unknown" if o["yaw_rate"] is None else f"{o['yaw_rate'] * 180 / 3.14159:.1f} deg over the last 1 s"
    return (f"Ego vehicle state: speed now {f(o['v_m0'])} m/s ({hist}); longitudinal acceleration {o['ax']:.2f} m/s^2; "
            f"heading change {yr}; navigation command {o['command']}. "
            f"Over the next 5 seconds, what will the ego vehicle's longitudinal behaviour be?")


def main():
    for split in ("train", "validation", "test"):
        L = json.load(open(os.path.join(ROOT, "artifacts", f"meta_labels_{split}.json")))
        Q = {}
        for c, o in L.items():
            q = {"question_id": f"meta_{c[-12:]}", "question_type": "choice", "category": "meta", "subcategory": "ego_meta_action",
                 "question": text(o), "choices": dict(OPTS)}
            if "c4" in o: q["answer"] = C4_LETTER[o["c4"]]
            Q[c] = [q]
        json.dump(Q, open(os.path.join(ROOT, "artifacts", f"meta_q_{split}.json"), "w"))
        print(split, len(Q))
    print(next(iter(Q.values()))[0]["question"])


if __name__ == "__main__":
    main()
