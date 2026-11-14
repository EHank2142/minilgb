"""Builds the tables (results/*.md) and figures (results/figures/*.png) from results/*.json.

  python experiments/make_report.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

# fixed colour per method (validated categorical palette, slots 1-6, light mode)
COLOR = {"LightGBM": "#2a78d6", "lgb_baseline": "#eb6834", "EFB_only": "#1baf7a",
         "SGB": "#eda100", "xgb_his": "#e87ba4", "xgb_exa": "#008300"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "legend.frameon": False})

PAPER_T2 = {"flight": dict(xgb_exa=5.94, xgb_his=1.05, lgb_baseline=1.39, EFB_only=0.27, LightGBM=0.22),
            "mslr": dict(xgb_exa=5.55, xgb_his=0.63, lgb_baseline=0.49, EFB_only=0.46, LightGBM=0.31)}
PAPER_T3 = {"flight": dict(xgb_exa=0.7601, xgb_his=0.7840, lgb_baseline=0.7847, SGB=0.7780, LightGBM=0.7846),
            "mslr": dict(xgb_exa=0.4977, xgb_his=0.4982, lgb_baseline=0.5277, SGB=0.5239, LightGBM=0.5275)}
PAPER_T4 = {"SGB": [0.5182, 0.5216, 0.5239, 0.5249, 0.5252, 0.5263, 0.5267],
            "GOSS": [0.5224, 0.5256, 0.5275, 0.5284, 0.5289, 0.5293, 0.5296]}


def jload(name):
    p = os.path.join(RES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def iter_time(h):
    t = np.array(h["iter_time"])
    return float(t[1:].mean()) if len(t) > 1 else float(t.mean())  # skip 1st (JIT warm-up)


def best(h, key="test"):
    return float(np.nanmax(h[key]))


def at_best_vali(h):
    v = np.array(h["vali"], float)
    i = int(np.nanargmax(v))
    return float(h["test"][i])


def fmt(x, d=4):
    return "–" if x is None else f"{x:.{d}f}"


lines = []
runs = {}
for dsn in ("flight", "mslr"):
    for m in ("lgb_baseline", "EFB_only", "LightGBM", "SGB"):
        runs[(dsn, "ours", m)] = jload(f"{dsn}_ours_{m}")
        runs[(dsn, "official", m)] = jload(f"{dsn}_official_{m}")
    for m in ("xgb_exa", "xgb_his"):
        runs[(dsn, "xgb", m)] = jload(f"{dsn}_{m}")


def H(dsn, src, m):
    r = runs.get((dsn, src, m))
    return r["history"] if r else None


# ------------------------------------------------------------------ Table 2
lines.append("### Table 2 – average training time per iteration (seconds)\n")
lines.append("| Dataset | Source | xgb_exa | xgb_his | lgb_baseline | EFB_only | LightGBM | speed-up LightGBM vs baseline | EFB_only vs baseline |")
lines.append("|---|---|---|---|---|---|---|---|---|")
for dsn, label in (("flight", "Flight Delay"), ("mslr", "LETOR (MSLR-WEB10K)")):
    p = PAPER_T2[dsn]
    lines.append(f"| {label} | paper (16 thr, full data) | {p['xgb_exa']} | {p['xgb_his']} | {p['lgb_baseline']} | "
                 f"{p['EFB_only']} | {p['LightGBM']} | {p['lgb_baseline']/p['LightGBM']:.1f}x | {p['lgb_baseline']/p['EFB_only']:.1f}x |")
    for src, name in (("official", "official libs (6 thr)"), ("ours", "**ours, from scratch** (6 thr)")):
        hs = {m: H(dsn, src, m) for m in ("lgb_baseline", "EFB_only", "LightGBM")}
        xs = {m: H(dsn, "xgb", m) for m in ("xgb_exa", "xgb_his")}
        t = {m: iter_time(h) if h else None for m, h in {**hs, **xs}.items()}
        if src == "ours":
            t["xgb_exa"] = t["xgb_his"] = None
        sp1 = t["lgb_baseline"] / t["LightGBM"] if t["lgb_baseline"] and t["LightGBM"] else None
        sp2 = t["lgb_baseline"] / t["EFB_only"] if t["lgb_baseline"] and t["EFB_only"] else None
        lines.append(f"| {label} | {name} | {fmt(t['xgb_exa'],3)} | {fmt(t['xgb_his'],3)} | {fmt(t['lgb_baseline'],3)} | "
                     f"{fmt(t['EFB_only'],3)} | {fmt(t['LightGBM'],3)} | {fmt(sp1,1)}x | {fmt(sp2,1)}x |")
lines.append("")

# ------------------------------------------------------------------ Table 3
lines.append("### Table 3 – test accuracy (AUC for Flight Delay, NDCG@10 for LETOR); best iteration on test, as in the paper\n")
lines.append("| Dataset | Source | xgb_exa | xgb_his | lgb_baseline | SGB | LightGBM |")
lines.append("|---|---|---|---|---|---|---|")
for dsn, label in (("flight", "Flight Delay"), ("mslr", "LETOR")):
    p = PAPER_T3[dsn]
    lines.append(f"| {label} | paper | {p['xgb_exa']} | {p['xgb_his']} | {p['lgb_baseline']} | {p['SGB']} | {p['LightGBM']} |")
    for src, name in (("official", "official libs"), ("ours", "**ours**")):
        v = {m: (best(H(dsn, src, m)) if H(dsn, src, m) else None) for m in ("lgb_baseline", "SGB", "LightGBM")}
        for m in ("xgb_exa", "xgb_his"):
            v[m] = best(H(dsn, "xgb", m)) if (src == "official" and H(dsn, "xgb", m)) else None
        lines.append(f"| {label} | {name} | {fmt(v['xgb_exa'])} | {fmt(v['xgb_his'])} | {fmt(v['lgb_baseline'])} | "
                     f"{fmt(v['SGB'])} | {fmt(v['LightGBM'])} |")
lines.append("")
lines.append("LETOR, test NDCG@10 at the best *validation* iteration (a protocol without test-set selection):\n")
lines.append("| Source | xgb_exa | xgb_his | lgb_baseline | EFB_only | SGB | LightGBM |")
lines.append("|---|---|---|---|---|---|---|")
for src, name in (("official", "official libs"), ("ours", "**ours**")):
    row = []
    for m in ("xgb_exa", "xgb_his", "lgb_baseline", "EFB_only", "SGB", "LightGBM"):
        h = H("mslr", "xgb", m) if m.startswith("xgb") else H("mslr", src, m)
        row.append(fmt(at_best_vali(h)) if (h and (src == "official" or not m.startswith("xgb"))) else "–")
    lines.append(f"| {name} | " + " | ".join(row) + " |")
lines.append("")

# ------------------------------------------------------------------ Fig 1 / 2
for dsn, metric, fname in (("flight", "AUC", "fig1_flight_time_auc.png"),
                           ("mslr", "NDCG@10", "fig2_letor_time_ndcg.png")):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    panels = [("ours", ["LightGBM", "EFB_only", "lgb_baseline", "SGB"], "From scratch (minilgb)"),
              ("official", ["LightGBM", "lgb_baseline", "xgb_his", "xgb_exa"], "Official LightGBM / XGBoost")]
    all_vals = []
    for ax, (src, methods, title) in zip(axes, panels):
        for m in methods:
            h = H(dsn, "xgb", m) if m.startswith("xgb") else H(dsn, src, m)
            if not h:
                continue
            t, v = np.array(h["time"]), np.array(h["test"], float)
            ok = ~np.isnan(v)
            ax.plot(t[ok], v[ok], color=COLOR[m], lw=2, label=m, solid_capstyle="round")
            ax.plot(t[ok][-1], v[ok][-1], "o", ms=7, color=COLOR[m], mec=SURF, mew=2)
            all_vals.append(v[ok])
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("training time (s)")
        ax.legend(loc="lower right")
    axes[0].set_ylabel(f"test {metric}")
    if all_vals:
        hi = max(np.nanmax(v) for v in all_vals)
        lo = np.nanpercentile(np.concatenate(all_vals), 5)
        axes[0].set_ylim(lo, hi + (hi - lo) * 0.05)
    fig.suptitle(f"Time–{metric} curve on {'Flight Delay (1M rows)' if dsn == 'flight' else 'LETOR (MSLR-WEB10K)'}",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close(fig)

# ------------------------------------------------------------------ Table 4
t4 = jload("table4")
if t4:
    ratios = sorted(float(r) for r in t4["ours"]["GOSS"])
    lines.append("### Table 4 – GOSS vs SGB on LETOR at equal sampling ratio (test NDCG@10, early stopping on validation)\n")
    lines.append("| Sampling ratio | " + " | ".join(f"{r:g}" for r in ratios) + " |")
    lines.append("|---|" + "---|" * len(ratios))
    allr = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    for name, key in (("paper SGB", "SGB"), ("paper GOSS", "GOSS")):
        lines.append(f"| {name} | " + " | ".join(f"{PAPER_T4[key][allr.index(r)]:.4f}" for r in ratios) + " |")
    for src in ("ours", "official"):
        for key in ("SGB", "GOSS"):
            d = {float(k): v for k, v in t4[src][key].items()}
            lines.append(f"| {'**ours**' if src == 'ours' else 'official'} {key} | " +
                         " | ".join(fmt(d[r]["test"]) if r in d else "–" for r in ratios) + " |")
    if t4.get("ablation"):
        lines.append("\nAblations (ours, LETOR, ratio 0.2 unless stated):\n")
        lines.append("| Variant | test NDCG@10 | best iter |")
        lines.append("|---|---|---|")
        for k, v in t4["ablation"].items():
            lines.append(f"| {k} | {v['test']:.4f} | {v['best_iter']} |")
    lines.append("")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, src, title in ((axes[0], "ours", "From scratch (minilgb)"), (axes[1], "official", "Official LightGBM")):
        for key, col in (("GOSS", COLOR["LightGBM"]), ("SGB", COLOR["SGB"])):
            d = {float(k): v["test"] for k, v in t4[src][key].items()}
            rs = sorted(d)
            ax.plot(rs, [d[r] for r in rs], "-o", color=col, lw=2, ms=6, mec=SURF, mew=1.5, label=key)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("sampling ratio (a+b for GOSS)")
        ax.legend()
    axes[0].set_ylabel("test NDCG@10")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_table4_goss_vs_sgb.png"), dpi=150)
    plt.close(fig)

# ------------------------------------------------------------------ GOSS theory
gt = jload("goss_theory")
if gt:
    lines.append("### Theorem 3.2 check – error of the estimated variance gain (Flight gradients)\n")
    lines.append("Relative error = mean |~V(d) − V(d)| / max V; regret = relative gain lost by the chosen split. "
                 "Sampling ratio 0.2 for all methods; mean over random draws.\n")
    methods = list(next(iter(gt.values())).keys())
    lines.append("| boosting iter | n | " + " | ".join(f"{m} err / regret" for m in methods) + " |")
    lines.append("|---|---|" + "---|" * len(methods))
    for k, r in gt.items():
        it, n = k.split("_")
        lines.append(f"| {it[4:]} | {int(n[1:]):,} | " +
                     " | ".join(f"{r[m]['rel_err']:.4f} / {r[m]['regret']:.4f}" for m in methods) + " |")
    lines.append("")
    iters = sorted({int(k.split('_')[0][4:]) for k in gt})
    fig, axes = plt.subplots(1, len(iters), figsize=(4 * len(iters), 3.6), sharey=True)
    cols = [COLOR["LightGBM"], COLOR["EFB_only"], COLOR["SGB"]]
    for ax, it in zip(np.atleast_1d(axes), iters):
        ks = sorted([k for k in gt if int(k.split('_')[0][4:]) == it], key=lambda k: int(k.split('_n')[1]))
        ns = [int(k.split('_n')[1]) for k in ks]
        for m, c in zip(methods, cols):
            ax.plot(ns, [gt[k][m]["rel_err"] for k in ks], "-o", color=c, lw=2, ms=6, mec=SURF, mew=1.5, label=m)
        ax.plot(ns, [gt[ks[0]][methods[-1]]["rel_err"] * np.sqrt(ns[0] / x) for x in ns], color=INK2,
                lw=1, ls=(0, (1, 2)), label=r"$\propto 1/\sqrt{n}$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"gradients after {it} iterations", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("n (instances at the node)")
    np.atleast_1d(axes)[0].set_ylabel("relative error of variance gain")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_goss_theory.png"), dpi=150)
    plt.close(fig)

# ------------------------------------------------------------------ EFB
ef = jload("efb_synthetic")
if ef:
    lines.append(f"### EFB on a KDD-like synthetic sparse dataset ({ef['shape'][0]:,} x {ef['shape'][1]:,}, "
                 f"{ef['nnz_per_row']:.1f} non-zeros/row)\n")
    o, f = ef["ours"], ef["official"]
    lines.append("| | #feature groups | time / iter (s) | test AUC |")
    lines.append("|---|---|---|---|")
    lines.append(f"| ours lgb_baseline | {o['baseline']['groups']:,} | {o['baseline']['iter_time']:.3f} | {o['baseline']['auc']:.5f} |")
    lines.append(f"| ours EFB_only (gamma=0) | {o['efb']['groups']:,} | {o['efb']['iter_time']:.3f} | {o['efb']['auc']:.5f} |")
    lines.append(f"| official lgb_baseline | – | {f['baseline']['iter_time']:.3f} | {f['baseline']['auc']:.5f} |")
    lines.append(f"| official EFB_only | – | {f['efb']['iter_time']:.3f} | {f['efb']['auc']:.5f} |")
    lines.append("\nBundling order (Alg. 3 by graph degree vs. by non-zero count):\n")
    lines.append("| order | #bundles | bundling time (s) |")
    lines.append("|---|---|---|")
    for k, v in ef["order"].items():
        lines.append(f"| {k} | {v['groups']:,} | {v['bundle_time']:.2f} |")
    lines.append("\nEffect of the max conflict rate gamma:\n")
    lines.append("| gamma | #groups | conflicting rows | ours time/iter | ours AUC | official time/iter | official AUC |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in ef["gamma"]:
        lines.append(f"| {r['gamma']:g} | {r['groups']:,} | {r['conflict_rows']:,} | {r['iter_time']:.3f} | "
                     f"{r['auc']:.5f} | {r['official_iter_time']:.3f} | {r['official_auc']:.5f} |")
    lines.append("")
    g = ef["gamma"]
    xs = [max(r["gamma"], 1e-5) for r in g]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].plot(xs, [r["groups"] for r in g], "-o", color=COLOR["EFB_only"], lw=2, ms=6, mec=SURF, mew=1.5)
    axes[0].set_ylabel("#feature groups"); axes[0].set_yscale("log")
    axes[1].plot(xs, [r["iter_time"] for r in g], "-o", color=COLOR["EFB_only"], lw=2, ms=6, mec=SURF, mew=1.5, label="ours")
    axes[1].plot(xs, [r["official_iter_time"] for r in g], "-o", color=COLOR["LightGBM"], lw=2, ms=6, mec=SURF, mew=1.5, label="official")
    axes[1].set_ylabel("time / iteration (s)"); axes[1].legend()
    axes[2].plot(xs, [r["auc"] for r in g], "-o", color=COLOR["EFB_only"], lw=2, ms=6, mec=SURF, mew=1.5, label="ours")
    axes[2].plot(xs, [r["official_auc"] for r in g], "-o", color=COLOR["LightGBM"], lw=2, ms=6, mec=SURF, mew=1.5, label="official")
    axes[2].set_ylabel("test AUC"); axes[2].legend()
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("max conflict rate gamma (0 plotted at 1e-5)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_efb_gamma.png"), dpi=150)
    plt.close(fig)

# feature grouping on the real datasets
lines.append("### EFB on the real datasets (ours)\n")
lines.append("| Dataset | #features | #groups after EFB | bundling time (s) |")
lines.append("|---|---|---|---|")
for dsn in ("flight", "mslr"):
    r = runs.get((dsn, "ours", "EFB_only"))
    if r:
        lines.append(f"| {dsn} | {r['features']} | {r['groups']} | {r['bundle_time']:.2f} |")
lines.append("")

with open(os.path.join(RES, "RESULTS.md"), "w", encoding="utf-8") as fh:
    fh.write("# Reproduced results\n\n(generated by experiments/make_report.py)\n\n" + "\n".join(lines))
print("\n".join(lines))
