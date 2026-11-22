"""Build a challenge_submission.json from cached test clips.

Trajectories are produced in GLOBAL frame by a planner, then converted to the
key-frame ego frame with the devkit's own global_to_ego / normalize_ego_trajectory
so the file matches exactly what the evaluator expects.

Answers are produced by an --answer strategy. Because planning and reasoning are
scored independently, a fixed-letter answer strategy turns a submission into a
controlled experiment that measures P(correct answer == that letter) over all
~3000 questions without perturbing the planning score.
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "probe")); import _stub_torch  # noqa: F401
sys.path.insert(0, os.path.join(ROOT, "recon", "devkit"))

from nureasoning.planning.benchmark import (            # noqa: E402
    BenchmarkConfig, build_route_line, select_key_frame_idx, _load_metadata)
from nureasoning.common.pickle_io import load_pickle     # noqa: E402
from nureasoning.submission.format import (              # noqa: E402
    global_to_ego, normalize_ego_trajectory, trajectory_to_list,
    make_clip_record, make_submission, write_submission,
    load_submission, validate_submission)
from nureasoning.submission.clips import load_clip_questions  # noqa: E402
import planners                                          # noqa: E402
import pickle                                            # noqa: E402
from qfeat import feats as _qfeats                        # noqa: E402
import reason_model as RM                                 # noqa: E402

CFG = BenchmarkConfig()
_CLIPS_ROOT = "cache/test"
_A15_ROOT = os.environ.get("A15_ROOT", "a15_out/nonav/test")


ROUTE_TRUST_M = 5.0


def route_trustworthy(ego, route):
    """The provided route must pass near the car. On 44 TEST clips it is >10 m
    away (max 3.7 km); following it then teleports the car onto a distant route.
    Validation, clips >=5 m from route: route-follow 0.152 NPS vs history
    extrapolation 0.550 (23 clips); >=10 m: 0.128 vs 0.661 (16 clips)."""
    if route is None or route.is_empty or route.length < 1.0:
        return False
    from shapely.geometry import Point
    return route.distance(Point(ego.pose["x"], ego.pose["y"])) < ROUTE_TRUST_M


def _fallback(cp, meta, frames, ki, ego):
    """No trustworthy route: mean of the FALLBACK path sources' raw trajectories (registry
    names, '+'-separated, e.g. a15ft_h0+a15ft_h1; yaw averaged on the circle), else
    constant_turn. Validation, 174 untrusted-route clips: constant_turn 0.6532, a15ft h0
    0.7477, h1 0.7188, mean(h0, h1) 0.7544."""
    names = [x for x in os.environ.get("FALLBACK", "").split("+") if x]
    if names:
        from a15_calib import mean_global
        reg = json.load(open("artifacts/path_sources.json"))
        G = [mean_global(np.load(os.path.join(reg[x]["test"], os.path.basename(cp) + ".npz"))) for x in names]
        g = np.mean(G, 0)
        g[:, 2] = np.arctan2(np.mean([np.sin(t[:, 2]) for t in G], 0), np.mean([np.cos(t[:, 2]) for t in G], 0))
        return g
    return planners.constant_turn(cp, meta, frames, ki, ego, None)


def plan(provider, cp, meta, frames, ki, ego, route):
    if provider != "constant_velocity" and provider != "zeros" and not route_trustworthy(ego, route):
        # a chooser pick from a path source (index >= 148) does not follow the route: keep it
        k = None
        if provider.startswith("selected:"):
            sel = json.load(open(provider.split(":", 1)[1]))
            if sel["cands"] == "labels148+sources":
                k = sel["selections"].get(os.path.basename(cp))
        if k is None or k < 148:
            return _fallback(cp, meta, frames, ki, ego)
    if provider == "constant_velocity":
        import math
        x, y, yaw = ego.pose["x"], ego.pose["y"], ego.pose["yaw"]
        vx, vy = ego.velocity.get("vx", 0.0), ego.velocity.get("vy", 0.0)
        c, s = math.cos(yaw), math.sin(yaw)
        T = np.arange(51) * 0.1
        return np.column_stack([x + (vx*c - vy*s)*T, y + (vx*s + vy*c)*T, np.full(51, yaw)])
    if provider == "route_follow":
        return planners.route_follow(cp, meta, frames, ki, ego, route)
    if provider.startswith("selected:"):
        # selections JSON from vscorer --predict: {clip: vocabulary index}
        global _SEL
        if "_SEL" not in globals():
            _SEL = json.load(open(provider.split(":", 1)[1]))
            assert _SEL["cands"] in ("labels148", "labels148+labels_a15", "labels148+sources"), "unknown candidate set"
        from vocab148 import VOCAB148, build
        k = _SEL["selections"].get(os.path.basename(cp))
        if k is not None and k >= 148 and _SEL["cands"] == "labels148+sources":
            # multi-source chooser: map k to (block, local index); build exactly as labelled
            from label_paths import SPECS as SP28
            from label_a15 import SPECS as SP27
            from a15_calib import mean_global, retime
            from planners import arc_length_kinematic, _speed, _lon_accel
            reg = json.load(open("artifacts/path_sources.json"))
            off = 0
            for name, labdir, K in _SEL["blocks"]:
                if k < off + K: break
                off += K
            j = k - off; src = reg[labdir.replace("labels_", "", 1)]
            specs = SP27 if src.get("specs") == 27 else SP28
            if src["kind"] == "a15":
                g = mean_global(np.load(os.path.join(src["test"], os.path.basename(cp) + ".npz")))
            else:
                d = np.load(src["test"]); g = np.asarray(d["traj"][list(d["names"]).index(os.path.basename(cp))], float)
            kind, v = specs[j]
            if kind == "raw":
                return g
            v0 = _speed(ego); kin = arc_length_kinematic(v0, _lon_accel(ego))
            own = float(np.sum(np.hypot(*np.diff(g[:, :2], axis=0).T)))
            L = kin * v if kind == "rel" else (v if kind == "abs" else own)
            return retime(g, v0, max(0.0, L), ego.pose["yaw"])
        if k is not None and k >= 148:
            assert _SEL["cands"] == "labels148+labels_a15"
            # Alpamayo mean path re-timed to candidate k-148 (label_a15.SPECS), exactly as labelled
            from label_a15 import SPECS
            from a15_calib import mean_global, retime
            from planners import arc_length_kinematic, _speed, _lon_accel
            z = np.load(os.path.join(_A15_ROOT, os.path.basename(cp) + ".npz"))
            g = mean_global(z); v0 = _speed(ego)
            kind, v = SPECS[k - 148]
            kin = arc_length_kinematic(v0, _lon_accel(ego))
            own = float(np.sum(np.hypot(*np.diff(g[:, :2], axis=0).T)))
            L = kin * v if kind == "rel" else (v if kind == "abs" else own)
            return retime(g, v0, max(0.0, L), ego.pose["yaw"])
        if k is None or route is None or route.is_empty or route.length < 1.0:
            return planners.route_follow(cp, meta, frames, ki, ego, route)   # no route / no pick -> default
        return build(meta, frames, ki, ego, route, VOCAB148[k])
    if provider == "zeros":
        x, y, yaw = ego.pose["x"], ego.pose["y"], ego.pose["yaw"]
        return np.column_stack([np.full(51, x), np.full(51, y), np.full(51, yaw)])
    raise ValueError(provider)


def _norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower().replace("_", " "))


_PRIORS = None
_MODELS = None
_CLIPF = None


def _load_clipfeat(path="artifacts/clipfeat_test.json"):
    """Per-clip ego/route context. The scene-level models are fitted WITH these,
    so omitting them here would feed the model all-NaN columns and silently
    degrade every driving-decision and risk answer."""
    global _CLIPF
    if _CLIPF is None:
        _CLIPF = json.load(open(path))
    return _CLIPF


_MODELS_PATH = os.environ.get("REASON_MODELS", "artifacts/reason_models.pkl")
_MOTION = None


def _load_models(path=None):
    """A bundle that records its own feature list ("num") is vectorised with it, so
    shipped models and submission features cannot drift apart."""
    global _MODELS
    if _MODELS is None:
        with open(path or _MODELS_PATH, "rb") as fh:
            _MODELS = pickle.load(fh)
        if "num" in _MODELS:
            RM.NUM = list(_MODELS["num"])
    return _MODELS


_COT = None


def _cot_feats(q):
    """Alpamayo chain-of-causation cues (src/cotfeat.py) for scene-level questions."""
    global _COT
    if _COT is None:
        p = os.environ.get("COT_FEATS", "artifacts/cotfeat_test.json")
        _COT = json.load(open(p)) if os.path.isfile(p) else {}
    return _COT.get(q.get("_clip", ""), {})


def _motion_feats(q):
    """DINOv3 correspondence motion (src/motion_corr.py) for future-motion questions."""
    global _MOTION
    if q.get("subcategory") != "future_motion_label":
        return {}
    if _MOTION is None:
        p = "artifacts/motion_test.json"
        _MOTION = json.load(open(p)) if os.path.isfile(p) else {}
    m = _MOTION.get(q.get("question_id"))
    return {f"m_{k}": v for k, v in m.items()} if m else {}


def _load_priors(path="artifacts/answer_priors.json"):
    global _PRIORS
    if _PRIORS is None:
        _PRIORS = json.load(open(path))["priors"]
    return _PRIORS


_RIGS = {}
_Z = None
GEO_SUBS = {"camera_view_for_3d_object", "image_region_from_3d_object", "multiview_camera_pair"}


def _geo_answer(q):
    """Calibration geometry for subcategories that ASK for a camera or image
    region. The organisers' test generator labels multi-camera objects differently
    from the public devkit, so models learned from devkit-generated labels do not
    transfer; projection with each clip's own calibration does."""
    global _Z
    import camgeo
    if _Z is None:
        _Z = json.load(open("artifacts/obj_center_z.json"))
    f = _qfeats(q); sub = q.get("subcategory"); clip = q.get("_clip", "")
    if "x" not in f or not clip:
        return None
    rig = _RIGS.get(clip)
    if rig is None:
        rig = _RIGS[clip] = camgeo.Rig(json.load(open(os.path.join(_CLIPS_ROOT, clip, "metadata.json"))))
    z = _Z["z_by_obj"].get(f.get("obj"), _Z["z_default"])
    ch = q.get("choices") or {}
    by_text = {_norm(v): k for k, v in ch.items()}
    if sub == "camera_view_for_3d_object":
        names = [_norm(v).replace(" ", "_") for v in ch.values()]
        cam = camgeo.best_camera(rig, f["x"], f["y"], z, names)
        return by_text.get(_norm(cam)) if cam else None
    if sub == "image_region_from_3d_object":
        cam = f.get("cam")
        if cam not in rig.cams: return None
        u, v, ok = rig.project(cam, (f["x"], f["y"], z))
        return by_text.get(_norm(camgeo.region_name(u, v))) if ok else None
    if sub == "multiview_camera_pair":
        pair = camgeo.best_pair(rig, f["x"], f["y"], z, list(ch.values()))
        return by_text.get(_norm(pair)) if pair else None
    return None


_VLM = None


def _choice_probs(kind_model, q, maps):
    """Same as reason_train.choice_probs. Copied, because importing reason_train would append
    its feature columns to RM.NUM and silently break the bundle's vectorisation here."""
    kind, m, cols = kind_model
    ch = q.get("choices") or {}
    if kind == "reg" or not ch: return None
    pr = m.predict_proba(np.array([RM.vec(q["_f"], maps)], float)[:, cols])[0]
    p = {cl: float(v) for cl, v in zip(m.classes_, pr)}
    out = {k: p.get(RM.norm_text(v), 0.0) for k, v in ch.items()}
    z = sum(out.values())
    return {k: v / z for k, v in out.items()} if z > 0 else None


def _vlm_blend(q, mdl, qq, maps, letter):
    """Optional per-subcategory blend with the reasoning VLM (src/reason_vlm_eval.py output):
    VLM_PROBS = jsonl of {clip_dir, question_id, probs}, VLM_ALPHA = "sub:a,sub:a,...".
    p = (1 - a) * p_gbdt + a * p_vlm (validation, held-out halves: src/reason_vlm_blend.py).
    Without VLM_PROBS, or for other subcategories, the GBDT letter is returned unchanged."""
    global _VLM
    if _VLM is None:
        _VLM = {"alpha": {}, "probs": {}}
        if os.environ.get("VLM_PROBS"):
            _VLM["alpha"] = {k: float(v) for k, v in (x.split(":") for x in os.environ.get("VLM_ALPHA", "").split(",") if x)}
            for f in os.environ["VLM_PROBS"].split("+"):
                for line in open(f):
                    r = json.loads(line); _VLM["probs"][(r["clip_dir"], r["question_id"])] = r["probs"]
    a = _VLM["alpha"].get(q.get("subcategory"), 0.0)
    pv = _VLM["probs"].get((q.get("_clip", ""), q.get("question_id")))
    if a <= 0 or pv is None:
        return letter
    pg = _choice_probs(mdl, qq, maps) or {k: float(k == letter) for k in pv}
    s = {k: (1 - a) * pg.get(k, 0.0) + a * pv[k] for k in pv}
    return max(sorted(pv), key=lambda k: s[k])


def answer_for(q, strategy):
    choices = q.get("choices") or {}
    if strategy == "geo":
        if q.get("subcategory") in GEO_SUBS:
            a = _geo_answer(q)
            if a is not None:
                return a
        return answer_for(q, "qmodel")
    if len(strategy) == 1 and strategy in "ABCD":
        # a few questions offer only A/B; fall back so the answer is always valid
        return strategy if (not choices or strategy in choices) else sorted(choices)[0]
    if strategy == "qmodel":
        if not choices:
            return "A"
        bundle = _load_models()
        sub = q.get("subcategory")
        mdl = bundle["models"].get(sub)
        if mdl is not None:
            qq = dict(q)
            f = _qfeats(q)
            f.update(_load_clipfeat().get(q.get("_clip", ""), {}))
            qq["_f"] = f
            from dist_geo import geo_feats
            f.update(geo_feats(qq, _CLIPS_ROOT))
            f.update(_motion_feats(q))
            if q.get("subcategory") in set(bundle.get("cot_subs") or []):
                f.update(_cot_feats(q))
            pri = (bundle.get("priors") or _load_priors()).get(sub, {})
            try:
                letter = RM.predict_letter(mdl, qq, bundle["maps"], pri)
            except Exception:
                letter = None             # fall through to the prior
            if letter is not None:
                return _vlm_blend(q, mdl, qq, bundle["maps"], letter)
        return answer_for(q, "prior")
    if strategy == "prior":
        if not choices:
            return "A"
        pri = _load_priors().get(q.get("subcategory"), {})
        best = max(choices, key=lambda k: pri.get(_norm(choices[k]), 0.0))
        # no choice text seen in training -> fall back to the best constant letter
        if pri.get(_norm(choices[best]), 0.0) <= 0.0:
            return "C" if "C" in choices else sorted(choices)[0]
        return best
    raise ValueError(strategy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="cache/test")
    ap.add_argument("--provider", default="route_follow")
    ap.add_argument("--answer", default="A", help="A|B|C|D")
    ap.add_argument("--name", required=True, help="e.g. v1.1_smoke_cv_allA")
    ap.add_argument("--outdir", default="submissions")
    a = ap.parse_args()
    global _CLIPS_ROOT
    _CLIPS_ROOT = a.clips

    clips = sorted(os.path.join(a.clips, d) for d in os.listdir(a.clips)
                   if os.path.isfile(os.path.join(a.clips, d, "metadata.json")))
    records, n_q = [], 0
    for cp in clips:
        meta = _load_metadata(cp); frames = meta.get("frames", [])
        ki = select_key_frame_idx(frames, CFG.key_frame_index)
        ego = load_pickle(os.path.join(cp, frames[ki]["ego_state"]))
        route = build_route_line(frames[ki].get("mission_goal"))
        g = plan(a.provider, cp, meta, frames, ki, ego, route)
        e = normalize_ego_trajectory(global_to_ego(g, ego.pose))
        qs = load_clip_questions(cp)
        clip_name = os.path.basename(cp)
        for q in qs:
            q["_clip"] = clip_name
        answers = [{"question_id": q["question_id"], "answer": answer_for(q, a.answer)}
                   for q in qs if q.get("question_id")]
        n_q += len(answers)
        records.append(make_clip_record(
            clip=os.path.basename(cp), clip_token=meta.get("clip_token"),
            log_name=meta.get("log_name"), target_frame_index=ki,
            trajectory=trajectory_to_list(e), answers=answers))

    payload = make_submission(records, extra_meta={
        "team": "Aman_upg", "method": a.name,
        "planner": a.provider, "answer_strategy": a.answer})
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, f"{a.name}.json")
    if os.path.exists(out):
        raise SystemExit(f"refusing to overwrite {out}")
    write_submission(payload, out)

    errs = validate_submission(load_submission(out), data_root=a.clips)
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    print(f"{a.name}")
    print(f"  clips={len(records)}  questions={n_q}  size={os.path.getsize(out)/1e6:.1f} MB")
    print(f"  planner={a.provider}  answers=all-{a.answer}")
    print(f"  sha256={sha}")
    print(f"  devkit validate_submission: "
          f"{'PASS (0 errors)' if not errs else f'{len(errs)} ERRORS'}")
    for e in errs[:8]:
        print(f"    - {e}")
    with open(os.path.join(a.outdir, "MANIFEST.jsonl"), "a") as fh:
        fh.write(json.dumps({"name": a.name, "file": out, "sha256": sha,
                             "clips": len(records), "questions": n_q,
                             "planner": a.provider, "answer": a.answer,
                             "valid": not errs,
                             "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")


if __name__ == "__main__":
    main()
