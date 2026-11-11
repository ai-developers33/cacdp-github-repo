"""Generate three diagnostic plots from the archived Study 2 model results."""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "generated_figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "study2_eval_results.json"), encoding="utf-8") as f:
    ev = json.load(f)
with open(os.path.join(DATA_DIR, "study2_runs_summary.json"), encoding="utf-8") as f:
    sm = json.load(f)

rf_cv = ev["run_grouped_5fold_cv"]["RandomForest"]
lr_cv = ev["run_grouped_5fold_cv"]["LogisticRegression"]
rf_loso = ev["leave_one_scenario_out"]["RandomForest"]
lr_loso = ev["leave_one_scenario_out"]["LogisticRegression"]
pr_curves = ev["pr_curves_rf"]


def save_pr_curve_svg(path: str):
    w, h = 680, 440
    ml, mr, mt, mb = 72, 30, 42, 62
    pw, ph = w - ml - mr, h - mt - mb

    def xy(rec: float, prec: float):
        return ml + rec * pw, mt + (1.0 - (prec - 0.5) / 0.5) * ph

    styles = [
        ("Model_A_Telemetry", "#64748B", "6,4", f"Model A: Telemetry-Only (PR-AUC = {rf_cv['Model_A_Telemetry']['pr_auc']:.4f})"),
        ("Model_B_Coarse", "#D97706", "3,3", f"Model B-Coarse: Naive Deploy Flags (PR-AUC = {rf_cv['Model_B_Coarse']['pr_auc']:.4f})"),
        ("Model_B_Semantic", "#0F766E", "none", f"Model B-Semantic: Contract + Queueing Util (PR-AUC = {rf_cv['Model_B_Semantic']['pr_auc']:.4f})"),
    ]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{w}" height="{h}" fill="#FFFFFF"/>',
        f'<text x="{w/2:.1f}" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#0F172A">Live Apache Kafka Study 2: Precision-Recall Curves (5-Fold Run-Grouped CV)</text>',
    ]

    for i in range(6):
        yv = 0.5 + 0.1 * i
        _, yp = xy(0, yv)
        lines.append(f'<line x1="{ml}" y1="{yp:.1f}" x2="{ml+pw}" y2="{yp:.1f}" stroke="#E2E8F0" stroke-width="1"/>')
        lines.append(f'<text x="{ml-10}" y="{yp+4:.1f}" text-anchor="end" font-size="11" fill="#334155">{yv:.1f}</text>')
        xv = 0.2 * i
        xp, _ = xy(xv, 0.5)
        lines.append(f'<line x1="{xp:.1f}" y1="{mt}" x2="{xp:.1f}" y2="{mt+ph}" stroke="#E2E8F0" stroke-width="1"/>')
        lines.append(f'<text x="{xp:.1f}" y="{mt+ph+20}" text-anchor="middle" font-size="11" fill="#334155">{xv:.1f}</text>')

    lines.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#334155" stroke-width="1.2"/>')
    lines.append(f'<text x="{ml+pw/2:.1f}" y="{h-16}" text-anchor="middle" font-size="12" font-weight="bold" fill="#0F172A">Recall (True Degradation Windows Detected)</text>')
    lines.append(f'<text x="20" y="{mt+ph/2:.1f}" text-anchor="middle" font-size="12" font-weight="bold" fill="#0F172A" transform="rotate(-90 20 {mt+ph/2:.1f})">Precision</text>')

    for key, col, dash, _ in styles:
        pts = []
        for r_v, p_v in zip(pr_curves[key]["recall"], pr_curves[key]["precision"]):
            px, py = xy(float(r_v), max(0.5, float(p_v)))
            pts.append(f"{px:.1f},{py:.1f}")
        dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        lines.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.6"{dash_attr} points="{" ".join(pts)}"/>')

    lx, ly = ml + 18, mt + ph - 86
    lines.append(f'<rect x="{lx}" y="{ly}" width="430" height="72" rx="4" fill="#F8FAFC" stroke="#CBD5E1"/>')
    for idx, (_, col, dash, label) in enumerate(styles):
        yy = ly + 18 + idx * 21
        dash_attr = f' stroke-dasharray="{dash}"' if dash != "none" else ""
        lines.append(f'<line x1="{lx+12}" y1="{yy}" x2="{lx+42}" y2="{yy}" stroke="{col}" stroke-width="2.6"{dash_attr}/>')
        lines.append(f'<text x="{lx+50}" y="{yy+4}" font-size="11" fill="#0F172A">{label}</text>')

    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_loso_far_svg(path: str):
    w, h = 760, 420
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{w}" height="{h}" fill="#FFFFFF"/>',
        f'<text x="{w/2:.1f}" y="26" text-anchor="middle" font-size="14" font-weight="bold" fill="#0F172A">Live Study 2: Benign Rollout False-Alarm Rate vs. Zero-Shot LOSO F1</text>',
    ]

    # Left panel: Benign Rollout FAR (%)
    ml1, mt, pw1, ph = 64, 58, 280, 290
    lines.append(f'<text x="{ml1+pw1/2:.1f}" y="48" text-anchor="middle" font-size="12" font-weight="bold" fill="#1E293B">(a) Benign Rollout False-Alarm Rate (%)</text>')
    max_far = 8.0
    for i in range(5):
        val = i * 2.0
        yy = mt + ph - (val / max_far) * ph
        lines.append(f'<line x1="{ml1}" y1="{yy:.1f}" x2="{ml1+pw1}" y2="{yy:.1f}" stroke="#E2E8F0"/>')
        lines.append(f'<text x="{ml1-8}" y="{yy+4:.1f}" text-anchor="end" font-size="10.5" fill="#334155">{val:.0f}%</text>')
    lines.append(f'<rect x="{ml1}" y="{mt}" width="{pw1}" height="{ph}" fill="none" stroke="#334155"/>')

    groups_far = [
        ("Model A", rf_cv["Model_A_Telemetry"]["benign_rollout_far"] * 100, lr_cv["Model_A_Telemetry"]["benign_rollout_far"] * 100),
        ("B-Coarse", rf_cv["Model_B_Coarse"]["benign_rollout_far"] * 100, lr_cv["Model_B_Coarse"]["benign_rollout_far"] * 100),
        ("B-Semantic", rf_cv["Model_B_Semantic"]["benign_rollout_far"] * 100, lr_cv["Model_B_Semantic"]["benign_rollout_far"] * 100),
    ]
    for gi, (gname, rf_v, lr_v) in enumerate(groups_far):
        gx = ml1 + 26 + gi * 88
        for bi, (val, col) in enumerate([(rf_v, "#1D4ED8"), (lr_v, "#0F766E")]):
            bh = max(2.0, (val / max_far) * ph) if val > 0 else 2.0
            bx = gx + bi * 32
            by = mt + ph - bh
            lines.append(f'<rect x="{bx}" y="{by:.1f}" width="28" height="{bh:.1f}" fill="{col}" rx="2"/>')
            lines.append(f'<text x="{bx+14}" y="{by-6:.1f}" text-anchor="middle" font-size="10" font-weight="bold" fill="#0F172A">{val:.1f}%</text>')
        lines.append(f'<text x="{gx+30}" y="{mt+ph+20}" text-anchor="middle" font-size="11" fill="#0F172A">{gname}</text>')

    # Right panel: LOSO Mean F1
    ml2, pw2 = 425, 295
    lines.append(f'<text x="{ml2+pw2/2:.1f}" y="48" text-anchor="middle" font-size="12" font-weight="bold" fill="#1E293B">(b) Unseen Fault Class LOSO Mean F1</text>')
    for i in range(5):
        val = 0.80 + i * 0.05
        yy = mt + ph - ((val - 0.80) / 0.20) * ph
        lines.append(f'<line x1="{ml2}" y1="{yy:.1f}" x2="{ml2+pw2}" y2="{yy:.1f}" stroke="#E2E8F0"/>')
        lines.append(f'<text x="{ml2-8}" y="{yy+4:.1f}" text-anchor="end" font-size="10.5" fill="#334155">{val:.2f}</text>')
    lines.append(f'<rect x="{ml2}" y="{mt}" width="{pw2}" height="{ph}" fill="none" stroke="#334155"/>')

    groups_loso = [
        ("Model A", rf_loso["Model_A_Telemetry"]["mean_f1"], lr_loso["Model_A_Telemetry"]["mean_f1"]),
        ("B-Coarse", rf_loso["Model_B_Coarse"]["mean_f1"], lr_loso["Model_B_Coarse"]["mean_f1"]),
        ("B-Semantic", rf_loso["Model_B_Semantic"]["mean_f1"], lr_loso["Model_B_Semantic"]["mean_f1"]),
    ]
    for gi, (gname, rf_v, lr_v) in enumerate(groups_loso):
        gx = ml2 + 28 + gi * 92
        for bi, (val, col) in enumerate([(rf_v, "#1D4ED8"), (lr_v, "#0F766E")]):
            bh = max(4.0, ((val - 0.80) / 0.20) * ph)
            bx = gx + bi * 32
            by = mt + ph - bh
            lines.append(f'<rect x="{bx}" y="{by:.1f}" width="28" height="{bh:.1f}" fill="{col}" rx="2"/>')
            lines.append(f'<text x="{bx+14}" y="{by-6:.1f}" text-anchor="middle" font-size="10" font-weight="bold" fill="#0F172A">{val:.3f}</text>')
        lines.append(f'<text x="{gx+30}" y="{mt+ph+20}" text-anchor="middle" font-size="11" fill="#0F172A">{gname}</text>')

    lines.append('<rect x="255" y="382" width="14" height="14" fill="#1D4ED8" rx="2"/>')
    lines.append('<text x="275" y="393" font-size="11" fill="#0F172A">Random Forest</text>')
    lines.append('<rect x="395" y="382" width="14" height="14" fill="#0F766E" rx="2"/>')
    lines.append('<text x="415" y="393" font-size="11" fill="#0F172A">L2 Logistic Regression</text>')
    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_lead_svg(path: str):
    w, h = 720, 400
    ml, mr, mt, mb = 185, 35, 48, 55
    pw, ph = w - ml - mr, h - mt - mb
    scs = [
        ("harmful_partition_hotspot", "Partition Hotspot (88% skew)"),
        ("harmful_consumer_slowdown", "Consumer Slowdown (440 ms)"),
        ("harmful_schema_breaking", "Breaking Schema (sku removed)"),
        ("harmful_routing_unsubscribed", "Unsubscribed Topic (.v2)"),
        ("harmful_dependency_outage", "Payment Gateway Outage"),
    ]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{w}" height="{h}" fill="#FFFFFF"/>',
        f'<text x="{w/2:.1f}" y="26" text-anchor="middle" font-size="14" font-weight="bold" fill="#0F172A">Mean Early-Warning Lead Time Before 2.0 s SLA Breach (Random Forest, Live Kafka)</text>',
    ]
    max_x = 5.5
    for i in range(6):
        xx = ml + (i / max_x) * pw
        lines.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt+ph}" stroke="#E2E8F0"/>')
        lines.append(f'<text x="{xx:.1f}" y="{mt+ph+18}" text-anchor="middle" font-size="11" fill="#334155">+{i:.0f} s</text>')
    lines.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#334155"/>')
    lines.append(f'<text x="{ml+pw/2:.1f}" y="{h-12}" text-anchor="middle" font-size="12" font-weight="bold" fill="#0F172A">Seconds of Advance Warning Prior to First Cohort SLA Expiration</text>')

    row_h = ph / len(scs)
    for idx, (sc_key, sc_label) in enumerate(scs):
        ry = mt + idx * row_h
        va = rf_cv["Model_A_Telemetry"]["lead_by_scenario_s"].get(sc_key, 0.0)
        vs = rf_cv["Model_B_Semantic"]["lead_by_scenario_s"].get(sc_key, 0.0)
        lines.append(f'<text x="{ml-10}" y="{ry + row_h/2 + 4:.1f}" text-anchor="end" font-size="11" fill="#0F172A">{sc_label}</text>')
        w_a = max(2.0, (va / max_x) * pw)
        w_s = max(2.0, (vs / max_x) * pw)
        lines.append(f'<rect x="{ml}" y="{ry+10:.1f}" width="{w_a:.1f}" height="16" fill="#94A3B8" rx="2"/>')
        lines.append(f'<text x="{ml+w_a+6:.1f}" y="{ry+22:.1f}" font-size="10" fill="#475569">+{va:.1f}s (Model A)</text>')
        lines.append(f'<rect x="{ml}" y="{ry+30:.1f}" width="{w_s:.1f}" height="16" fill="#0F766E" rx="2"/>')
        lines.append(f'<text x="{ml+w_s+6:.1f}" y="{ry+42:.1f}" font-size="10" font-weight="bold" fill="#0F766E">+{vs:.1f}s (B-Semantic)</text>')

    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


save_pr_curve_svg(os.path.join(FIG_DIR, "fig_study2_pr_curves.svg"))
save_loso_far_svg(os.path.join(FIG_DIR, "fig_study2_loso_and_far.svg"))
save_lead_svg(os.path.join(FIG_DIR, "fig_study2_lead_time.svg"))
