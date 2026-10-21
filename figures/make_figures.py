"""
Tables and figures of the paper, from committed results only.

Reads  configs/*.json and results/paper/ (the tables are
       read from results/paper/*/tables/*.csv and the fold scores in */folds.csv).
Writes figures/*.pdf (and a .png preview of each) and paper/tables/*.tex.

    python figures/make_figures.py            # everything
    python figures/make_figures.py table1 fig2

No number is typed by hand: every value is read from a result file, and each table checks
the values it recomputes from folds.csv against the committed table it reproduces.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "paper"
FIG = ROOT / "figures"
TAB = ROOT / "paper" / "tables"
LETTERS = ["S", "V", "T", "R", "M", "Q", "P", "C"]
NAME = {"stochastic_trend": "S", "volatility": "V", "trend": "T", "trend_shift": "R",
        "mean_shift": "M", "variance_shift": "Q", "point_anomaly": "P", "collective_anomaly": "C"}
CLF = {"lr": "LR", "hgb": "HGB", "ridge": "ridge"}

# one colour per feature set, used in every figure (Okabe-Ito)
COLOR = {"TDA-90": "#0072B2", "TSFresh-774": "#E69F00", "catch22": "#009E73", "MiniRocket": "#7F7F7F",
         "TSFresh-774+TDA-90": "#0072B2", "TSFresh-774+catch22": "#009E73",
         # Figure 1 only (series and windows, not feature sets): colour-blind safe, no pink,
         # no red/green pair, and none of them is a feature-set colour
         "volatility series": "#1A1A1A", "stationary series": "#8C8C8C",
         "turbulent window": "#D55E00", "calm window": "#56B4E9"}


def diverging_tda_tsfresh():
    """Diverging map for 'topological side minus TSFresh' differences: TSFresh orange below 0,
    white at 0, TDA-90 blue above 0 -- the feature-set colours of Figs. 2-4."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("tda_tsfresh", ["#8A5A00", COLOR["TSFresh-774"], "#FFFFFF",
                                                              COLOR["TDA-90"], "#003B5C"])


def style() -> None:
    mpl.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
        "axes.titlelocation": "left", "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
        "axes.linewidth": 0.6, "lines.linewidth": 1.4, "xtick.major.width": 0.6, "ytick.major.width": 0.6})


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"  -> figures/{name}.pdf")


# ------------------------------------------------------------------ parsing helpers ---
CELL = re.compile(r"^\s*([+\-]?\d+\.\d+)\s*\((\d+\.\d+)\)(?:\s*p=(\d+\.\d+))?")


def parse(cell: str):
    """'+0.048 (0.002) p=0.000**' -> (0.048, 0.002, 0.000); '0.846 (0.002)' -> (0.846, 0.002, None)."""
    m = CELL.match(str(cell))
    if not m:
        raise ValueError(f"cannot parse {cell!r}")
    return float(m.group(1)), float(m.group(2)), (float(m.group(3)) if m.group(3) else None)


def fp(p: float) -> str:
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def fs(v: float, signed: bool = True) -> str:
    """Three decimals; a signed value that rounds to zero is written 0.000, without a sign, so
    that it cannot read as a gain or a loss."""
    if signed and round(v, 3) == 0:
        return "0.000"
    s = f"{v:+.3f}" if signed else f"{v:.3f}"
    return s.replace("-", "$-$") if signed else s


def fit_width(text: str) -> str:
    """Wrap every tabular in adjustbox{max width=textwidth}: shrinks a wide appendix table to the
    text width of the journal class and leaves narrower tables unchanged."""
    text = re.sub(r"(\\begin\{tabular\})", r"\\begin{adjustbox}{max width=\\textwidth}\n\1", text)
    return text.replace("\\end{tabular}", "\\end{tabular}\n\\end{adjustbox}")


def fse(se: float) -> str:
    """Corrected standard error to three decimals; a value that rounds to 0.000 is written <0.001."""
    return r"$<$0.001" if round(se, 3) == 0 else f"{se:.3f}"


def mean_of_eight(folds: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean of the eight detectors per set, model, level and fold (levels 1-4)."""
    D = folds[folds.level.between(1, 4)]
    n = D.groupby(["set", "model", "level", "fold"]).char.nunique()
    assert (n == 8).all(), "every level 1-4 must score all eight detectors"
    return D.groupby(["set", "model", "level", "fold"])[metric].mean()


# ------------------------------------------------------------------------- Table 1 ---
def table1() -> None:
    cfg = {}
    for f in sorted(glob.glob(str(ROOT / "configs" / "*-config.json"))):
        cfg.update(json.loads(Path(f).read_text())["classes"])
    rows = []
    for cls, scens in cfg.items():
        chars = set() if cls == "stationary" else {NAME[p] for p in cls.split("+")}
        rows.append(dict(level=len(chars), n=sum(s["n"] for s in scens), **{c: int(c in chars) for c in LETTERS}))
    D = pd.DataFrame(rows)
    T = D.groupby("level").agg(classes=("n", "size"), series=("n", "sum"))
    for c in LETTERS:
        T[c] = (D[c] * D["n"]).groupby(D.level).sum()
    assert T.classes.tolist() == [1, 8, 22, 26, 13, 2] and T.series.sum() == 36000
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Classes and series per level, and positive series per detector at each "
             r"level. Level = number of causes; 500 series per class. S stochastic trend, V volatility, "
             r"T linear trend, R trend shift, M mean shift, Q variance shift, P point anomaly, "
             r"C collective anomaly.}",
             r"\label{tab:levels}",
             r"\begin{tabular}{l rr " + "r" * 8 + "}", r"\toprule",
             r"& & & \multicolumn{8}{c}{Positive series per detector} \\ \cmidrule(l){4-11}",
             r"Level & Classes & Series & " + " & ".join(LETTERS) + r" \\", r"\midrule"]
    for lv, r in T.iterrows():
        lines.append(f"{lv} & {r.classes} & {r.series:,} & ".replace(",", r"\,")
                     + " & ".join(f"{r[c]:,}".replace(",", r"\,") for c in LETTERS) + r" \\")
    tot = T.sum()
    lines += [r"\midrule", f"All & {tot.classes} & {tot.series:,} & ".replace(",", r"\,")
              + " & ".join(f"{tot[c]:,}".replace(",", r"\,") for c in LETTERS) + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "table1_levels.tex").write_text("\n".join(lines) + "\n")
    print("  -> paper/tables/table1_levels.tex"); print(T.to_string())


# ------------------------------------------------------------------------- Table 2 ---
ORDER2 = [("TDA-90", "lr"), ("TSFresh-774", "lr"), ("catch22", "lr"),
          ("TDA-90", "hgb"), ("TSFresh-774", "hgb"), ("catch22", "hgb"), ("MiniRocket", "ridge")]


def table2() -> None:
    folds = pd.read_csv(RES / "main" / "folds.csv")
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Full version of Table~\ref{tab:levels-results}. Mean of the eight detectors at levels 1--4 and its change from level~1 to level~4 "
             r"(drop; negative = accuracy lost), for every feature set and classifier. The last four columns "
             r"compare TDA-90 with the set of the row under the same classifier (TDA-90 minus row; "
             r"for MiniRocket, $^{a}$TDA-90 under LR and $^{b}$TDA-90 under HGB): a positive $\Delta$drop "
             r"means that TDA-90 loses less. Corrected standard errors in parentheses; $p$ from the Nadeau--Bengio test, 5 folds. "
             r"SESEPLACEHOLDER}",
             r"\label{tab:levels-full}",
             r"\begin{tabular}{ll cccc c cc cc}", r"\toprule",
             r"Set & Clf. & L1 & L2 & L3 & L4 & Drop & $\Delta$drop & $p$ & $\Delta$L4 & $p$ \\"]
    ses = []
    for metric, label in (("bal_acc", "Balanced accuracy"), ("auc", "ROC AUC")):
        T1 = pd.read_csv(RES / "main" / "tables" / f"T1_T2_mean_of_eight_{metric}.csv").set_index(["set", "model"])
        T23 = pd.read_csv(RES / "main" / "tables" / f"T2_T3_differences_{metric}.csv").set_index("comparison")
        # check: the committed table equals the mean over folds of the eight-detector mean
        M = mean_of_eight(folds, metric).groupby(["set", "model", "level"]).mean()
        for (s, m), r in T1.iterrows():
            for lv in range(1, 5):
                assert abs(parse(r[f"L{lv}"])[0] - M[(s, m, lv)]) < 6e-4, (s, m, lv)
        lines += [r"\midrule", rf"\multicolumn{{11}}{{l}}{{\emph{{{label}}}}} \\"]
        for s, m in ORDER2:
            r = T1.loc[(s, m)]
            lv = [parse(r[f"L{k}"]) for k in range(1, 5)]
            ses += [x[1] for x in lv]
            d = parse(r["drop L1->L4"])
            cells = [s, CLF[m]] + [fs(x[0], False) for x in lv] + [f"{fs(d[0])} ({fse(d[1])})"]
            if s == "TDA-90":
                cells += ["---"] * 4
            elif s == "MiniRocket":
                rows_mr = []
                for tm in ("lr", "hgb"):
                    c = T23.loc[f"TDA-90 ({tm}) - MiniRocket (ridge)"]
                    dd, l4 = parse(c["drop difference"]), parse(c["level 4"])
                    rows_mr.append([f"{fs(dd[0])} ({fse(dd[1])})", fp(dd[2]), f"{fs(l4[0])} ({fse(l4[1])})", fp(l4[2])])
                lines.append(" & ".join(cells + rows_mr[0]) + r"$^{a}$ \\")
                lines.append(" & ".join([""] * 7 + rows_mr[1]) + r"$^{b}$ \\")
                continue
            else:
                c = T23.loc[f"TDA-90 ({m}) - {s} ({m})"]
                dd, l4 = parse(c["drop difference"]), parse(c["level 4"])
                cells += [f"{fs(dd[0])} ({fse(dd[1])})", fp(dd[2]), f"{fs(l4[0])} ({fse(l4[1])})", fp(l4[2])]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    txt = "\n".join(lines) + "\n"
    txt = txt.replace("SESEPLACEHOLDER", f"Corrected standard errors of the level means are at most {max(ses):.3f}.")
    (TAB / "tableA3_levels_full.tex").write_text(fit_width(txt))
    print("  -> paper/tables/tableA3_levels_full.tex")


# ------------------------------------------------------------------------ Figure 2 ---
# ------------------------------------------------------------------------- Table 3 ---
def star(p: float) -> str:
    return "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def gains(metric: str) -> pd.DataFrame:
    """Fold-wise gain of each combined set over TSFresh-774 (same classifier, fold, level)."""
    base = mean_of_eight(pd.read_csv(RES / "main" / "folds.csv"), metric).xs("TSFresh-774", level="set")
    comb = mean_of_eight(pd.read_csv(RES / "complementarity" / "folds.csv"), metric)
    rows = []
    for (s, m, lv, f), v in comb.items():
        rows.append(dict(added=s.split("+")[1], model=m, level=lv, fold=f, gain=v - base[(m, lv, f)]))
    return pd.DataFrame(rows)


def table3() -> None:
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Full version of Table~\ref{tab:complement}. Gain from adding TDA-90 or catch22 to TSFresh-774: mean of the eight detectors of "
             r"the combined set minus that of TSFresh-774 alone, same classifier and fold, at each level; "
             r"the change of the gain from level~1 to level~4; and, at level~4, the gain from TDA-90 minus "
             r"the gain from catch22. Corrected standard errors in parentheses; "
             r"$^{*}p<0.05$, $^{**}p<0.01$, Nadeau--Bengio test, 5 folds.}",
             r"\label{tab:complement-full}",
             r"\begin{tabular}{ll ccccc}", r"\toprule",
             r"Clf. & Added to TSFresh-774 & L1 & L2 & L3 & L4 & L4 $-$ L1 \\"]
    for metric, label in (("bal_acc", "Balanced accuracy"), ("auc", "ROC AUC")):
        T = pd.read_csv(RES / "complementarity" / "tables" / f"T7_T8_gains_{metric}.csv").set_index(["model", "added"])
        G = gains(metric).groupby(["model", "added", "level"]).gain.mean()
        for (m, a), r in T.iterrows():
            if a in ("TDA-90", "catch22"):
                for lv in range(1, 5):
                    assert abs(parse(r[f"gain L{lv}"])[0] - G[(m, a, lv)]) < 6e-4, (metric, m, a, lv)
        lines += [r"\midrule", rf"\multicolumn{{7}}{{l}}{{\emph{{{label}}}}} \\"]
        for m in ("lr", "hgb"):
            for a, lab in (("TDA-90", "TDA-90"), ("catch22", "catch22"), ("TDA-90 minus catch22", r"TDA-90 $-$ catch22")):
                r = T.loc[(m, a)]
                cells = [CLF[m] if a == "TDA-90" else "", lab]
                for col in ["gain L1", "gain L2", "gain L3", "gain L4", "gain L4 - gain L1"]:
                    if pd.isna(r[col]):
                        cells.append("")
                    else:
                        v, se, pv = parse(r[col])
                        cells.append(f"{fs(v)} ({fse(se)}){star(pv)}")
                lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "tableA4_complement_full.tex").write_text(fit_width("\n".join(lines) + "\n"))
    print("  -> paper/tables/tableA4_complement_full.tex")


# ------------------------------------------------------------------------ Figure 3 ---
# ------------------------------------------------------------------------- Table 4 ---
ORDER4 = [("TSFresh-774", "TSFresh-774 (all)"), ("TSFresh top-100", "TSFresh top-100"),
          ("union top-100", "Union top-100"), ("TSFresh-774+TDA-90", "TSFresh-774 + TDA-90 (all)")]


def table4() -> None:
    lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Full version of Table~\ref{tab:budget}. Equal feature budget. Mean of the eight detectors at each level for the 100 "
             r"TSFresh-774 features and the 100 features of the pool TSFresh-774 $\cup$ TDA-90 ranked highest "
             r"by mutual information on each training fold, with TSFresh-774 and TSFresh-774 + TDA-90 "
             r"(all features) for reference. The row $\Delta$ is union top-100 minus TSFresh top-100, with its "
             r"change from level~1 to level~4. Corrected standard errors in parentheses; "
             r"$^{*}p<0.05$, $^{**}p<0.01$, Nadeau--Bengio test, 5 folds.}",
             r"\label{tab:budget-full}",
             r"\begin{tabular}{ll ccccc}", r"\toprule",
             r"Clf. & Features & L1 & L2 & L3 & L4 & L4 $-$ L1 \\"]
    bud = pd.read_csv(RES / "budget" / "folds.csv")
    for metric, label in (("bal_acc", "Balanced accuracy"), ("auc", "ROC AUC")):
        T10 = pd.read_csv(RES / "budget" / "tables" / f"T10_mean_of_eight_{metric}.csv").set_index(["model", "set"])
        T11 = pd.read_csv(RES / "budget" / "tables" / f"T11_union_minus_tsfresh_{metric}.csv").set_index("model")
        M = mean_of_eight(bud, metric).groupby(["set", "model", "level"]).mean()
        for (m, st), r in T10.iterrows():
            if st in ("TSFresh top-100", "union top-100"):
                for lv in range(1, 5):
                    assert abs(parse(r[f"L{lv}"])[0] - M[(st, m, lv)]) < 6e-4, (metric, m, st, lv)
        lines += [r"\midrule", rf"\multicolumn{{7}}{{l}}{{\emph{{{label}}}}} \\"]
        for m in ("lr", "hgb"):
            for i, (st, lab) in enumerate(ORDER4):
                r = T10.loc[(m, st)]
                lines.append(" & ".join([CLF[m] if i == 0 else "", lab]
                                        + [fs(parse(r[f"L{lv}"])[0], False) for lv in range(1, 5)] + [""]) + r" \\")
            r = T11.loc[m]
            cells = ["", r"$\Delta$ (H9, H10)"]
            for col in ["L1", "L2", "L3", "L4", "L4 - L1"]:
                v, se, pv = parse(r[col]); cells.append(f"{fs(v)} ({fse(se)}){star(pv)}")
            lines.append(" & ".join(cells) + r" \\")
            if m == "lr":
                lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "tableA5_budget_full.tex").write_text(fit_width("\n".join(lines) + "\n"))
    print("  -> paper/tables/tableA5_budget_full.tex")


# ------------------------------------------------------------------------ Figure 4 ---
FULL = {"S": "stochastic\ntrend", "V": "volatility", "T": "linear\ntrend", "R": "trend\nshift", "M": "mean\nshift",
        "Q": "variance\nshift", "P": "point\nanomaly", "C": "collective\nanomaly"}


def fig4() -> None:
    T = pd.read_csv(RES / "budget" / "tables" / "T12_tda_share.csv").set_index("char").reindex(LETTERS)
    S = pd.read_csv(RES / "budget" / "selected.csv")
    U = S[S.set == "union top-100"]
    chk = U.assign(t=U.family == "TDA").groupby(["char", "fold"]).t.mean().groupby("char").mean()
    assert np.allclose(chk.reindex(LETTERS).to_numpy(), T["TDA share"].to_numpy())
    pool = 90 / (774 + 90)                      # TDA-90 share of the 864-feature pool
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    x = np.arange(len(LETTERS))
    ax.axhline(pool, color="0.45", lw=0.8, ls="--", zorder=1, label=f"TDA-90 share of the 864-feature pool ({100 * pool:.1f} %)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), handlelength=2.2)
    ax.vlines(x, 0, T["TDA share"], color=COLOR["TDA-90"], lw=1.2, zorder=2)
    ax.errorbar(x, T["TDA share"], yerr=[T["TDA share"] - T["min over folds"], T["max over folds"] - T["TDA share"]],
                fmt="o", color=COLOR["TDA-90"], ms=4.5, elinewidth=0.8, capsize=0, zorder=3)
    for xi, v in zip(x, T["TDA share"]):
        ax.text(xi + 0.14, v, f"{v:.2f}", fontsize=6.5, va="center", color=COLOR["TDA-90"])
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n{FULL[c]}" for c in LETTERS])
    ax.set_ylim(0, 0.45); ax.set_xlim(-0.5, len(LETTERS) - 0.4)
    ax.set_ylabel("Share of TDA-90 features\nin the union top-100")
    ratio = T["TDA share"] / pool
    over = ratio[ratio > 1]
    print("  ratio to pool share:", ratio.round(2).to_dict())
    # Result statement lives in the caption and the text (no in-figure title).
    save(fig, "fig4_share")


# ------------------------------------------------------------------------ Figure 5 ---
def fig5() -> None:
    """Per detector at level 4, balanced accuracy: TDA-90 - TSFresh-774 (T4, H4), gain from
    adding TDA-90 (T9, H8), union top-100 - TSFresh top-100 (T13)."""
    T4 = pd.read_csv(RES / "main" / "tables" / "T4_per_detector_level4_differences.csv").set_index("comparison")
    T9 = pd.read_csv(RES / "complementarity" / "tables" / "T9_gain_per_detector_level4.csv").set_index(["metric", "model"])
    T13 = pd.read_csv(RES / "budget" / "tables" / "T13_per_detector_level4.csv").set_index(["metric", "model"])
    rows = {}
    for m in ("lr", "hgb"):
        rows[(m, "TDA-90 alone\n$-$ TSFresh-774")] = T4.loc[f"TDA-90 ({m}) - TSFresh-774 ({m})"]
        rows[(m, "TSFresh-774 + TDA-90\n$-$ TSFresh-774")] = T9.loc[("bal_acc", m)]
        rows[(m, "union top-100\n$-$ TSFresh top-100")] = T13.loc[("bal_acc", m)]
    V = {k: [parse(r[c]) for c in LETTERS] for k, r in rows.items()}
    vmax = max(abs(x[0]) for v in V.values() for x in v)
    vmax = np.ceil(vmax * 100) / 100
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.4), gridspec_kw={"hspace": 0.38})
    cmap = diverging_tda_tsfresh()
    for ax, m in zip(axes, ("lr", "hgb")):
        keys = [k for k in V if k[0] == m]
        Z = np.array([[x[0] for x in V[k]] for k in keys])
        im = ax.imshow(Z, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        for i, k in enumerate(keys):
            for j, (v, se, pv) in enumerate(V[k]):
                mark = "**" if pv < 0.01 else "*" if pv < 0.05 else ""
                txt = f"{v:+.3f}".replace("-", "\u2212") + mark
                r_, g_, b_, _ = cmap((v + vmax) / (2 * vmax))
                dark = 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_ < 0.5       # white text on dark cells
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white" if dark else "0.1",
                        fontweight="bold" if pv < 0.05 else "normal")
        ax.set_yticks(range(len(keys))); ax.set_yticklabels([k[1] for k in keys], fontsize=7.5)
        ax.set_xticks(range(8)); ax.set_xticklabels([f"{c}\n{FULL[c]}" for c in LETTERS] if m == "hgb" else LETTERS, fontsize=7.5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title({"lr": "Logistic regression", "hgb": "Gradient boosting"}[m])
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Difference in balanced accuracy, level 4\n(blue: topological side ahead; orange: TSFresh-774 ahead)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    save(fig, "fig5_per_cause")


# ------------------------------------------------------------------------- Table 5 ---
BLOCKS = [("paper H0, mean over windows", r"Rips $H_0$ summaries, mean over windows", 6),
          ("paper H0, std over windows", r"Rips $H_0$ summaries, std over windows", 6),
          ("paper H1, mean over windows", r"Rips $H_1$ summaries, mean over windows", 8),
          ("paper H1, std over windows", r"Rips $H_1$ summaries, std over windows", 8),
          ("windowed sublevel H0", r"Windowed sublevel $H_0$", 36),
          ("whole-series sublevel H0", r"Whole-series sublevel $H_0$", 26)]


def table5() -> None:
    """Exploratory. Block ablation of TDA-90 for the volatility (V) and mean-shift (M) detectors,
    logistic regression. results/paper/interpret/blocks_V.csv (2026-10-15) and blocks_M.csv
    (2026-10-19, same code; the V rerun reproduced blocks_V.csv byte for byte)."""
    Bs = {c: pd.read_csv(RES / "interpret" / f"blocks_{c}.csv").set_index("set") for c in ("V", "M")}
    allv = {c: B.loc["TDA-90 (all 90)"] for c, B in Bs.items()}
    assert sum(n for *_, n in BLOCKS) == 90
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Exploratory. Feature-group ablation for the volatility (V) and mean-shift (M) detectors: "
             r"logistic regression on one block of TDA-90 alone and on TDA-90 without that block; balanced "
             r"accuracy at levels 1 and 4, averaged over five folds; $\Delta$L4: change at level~4 from removing "
             r"the block. All 90 features: "
             rf"V {allv['V']['L1']:.3f} at level~1 and {allv['V']['L4']:.3f} at level~4; "
             rf"M {allv['M']['L1']:.3f} and {allv['M']['L4']:.3f}.}}",
             r"\label{tab:volatility}",
             r"\begin{adjustbox}{max width=\textwidth}",
             r"\begin{tabular}{l r cc cc c@{\hspace{10pt}}cc cc}", r"\toprule",
             r"& & \multicolumn{4}{c}{Volatility (V)} & & \multicolumn{4}{c}{Mean shift (M)} \\",
             r"\cmidrule(lr){3-6} \cmidrule(l){8-11}",
             r"& & \multicolumn{2}{c}{Block alone} & \multicolumn{2}{c}{Without block} & & "
             r"\multicolumn{2}{c}{Block alone} & \multicolumn{2}{c}{Without block} \\",
             r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){8-9} \cmidrule(l){10-11}",
             r"Block & Features & L1 & L4 & L4 & $\Delta$L4 & & L1 & L4 & L4 & $\Delta$L4 \\", r"\midrule"]
    for key, lab, n in BLOCKS:
        cells = [lab, str(n)]
        for k, c in enumerate(("V", "M")):
            B = Bs[c]; only = B.loc[f"only {key} ({n})"]; wo = B.loc[f"without {key}"]
            if k == 1:
                cells.append("")
            cells += [f"{only['L1']:.3f}", f"{only['L4']:.3f}", f"{wo['L4']:.3f}", fs(wo['L4'] - allv[c]['L4'])]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    (TAB / "table5_volatility.tex").write_text("\n".join(lines) + "\n")
    print("  -> paper/tables/table5_volatility.tex")


# ------------------------------------------------------------------------ Table A1 ---
def _cell(v, se, pv):
    return f"{fs(v)}{star(pv)}"


def tableA1() -> None:
    """Every hypothesis of the paper (Appendix B), its tests and their outcome. The
    outcome is computed here from the committed tables: a directional hypothesis is supported
    in a test if the difference has the expected sign and p < 0.05; H2 expects no significant
    difference."""
    R = {}
    for metric in ("bal_acc", "auc"):
        R[("T23", metric)] = pd.read_csv(RES / "main" / "tables" / f"T2_T3_differences_{metric}.csv").set_index("comparison")
        R[("T7", metric)] = pd.read_csv(RES / "complementarity" / "tables" / f"T7_T8_gains_{metric}.csv").set_index(["model", "added"])
        R[("T11", metric)] = pd.read_csv(RES / "budget" / "tables" / f"T11_union_minus_tsfresh_{metric}.csv").set_index("model")
    T4 = pd.read_csv(RES / "main" / "tables" / "T4_per_detector_level4_differences.csv").set_index("comparison")
    T9 = pd.read_csv(RES / "complementarity" / "tables" / "T9_gain_per_detector_level4.csv").set_index(["metric", "model"])
    T12 = pd.read_csv(RES / "budget" / "tables" / "T12_tda_share.csv").set_index("char")
    COLS = [("lr", "bal_acc"), ("hgb", "bal_acc"), ("lr", "auc"), ("hgb", "auc")]
    rows = []                                     # (hyp, test label, {col: (v,se,p)}, expectation)

    def four(get):
        return {c: parse(get(*c)) for c in COLS}
    rows.append(("H1", r"Drop difference, TDA-90 $-$ TSFresh-774", four(lambda m, k: R[("T23", k)].loc[f"TDA-90 ({m}) - TSFresh-774 ({m})", "drop difference"]), ">0"))
    rows.append(("H2", r"Level~4, TDA-90 $-$ TSFresh-774", four(lambda m, k: R[("T23", k)].loc[f"TDA-90 ({m}) - TSFresh-774 ({m})", "level 4"]), "ns"))
    for other in ("catch22", "MiniRocket"):
        o = (lambda m: f"{other} ({m})") if other == "catch22" else (lambda m: "MiniRocket (ridge)")
        rows.append(("H3", rf"Drop difference, TDA-90 $-$ {other}", four(lambda m, k: R[("T23", k)].loc[f"TDA-90 ({m}) - {o(m)}", "drop difference"]), "desc"))
        rows.append(("H3", rf"Level~4, TDA-90 $-$ {other}", four(lambda m, k: R[("T23", k)].loc[f"TDA-90 ({m}) - {o(m)}", "level 4"]), "desc"))
    for c, e in (("V", ">0"), ("M", ">0"), ("P", "<0")):
        rows.append(("H4", rf"Detector {c}, level~4, TDA-90 $-$ TSFresh-774",
                     {(m, "bal_acc"): parse(T4.loc[f"TDA-90 ({m}) - TSFresh-774 ({m})", c]) for m in ("lr", "hgb")}, e))
    rows.append(("H5", r"Gain from TDA-90, level~4", four(lambda m, k: R[("T7", k)].loc[(m, "TDA-90"), "gain L4"]), ">0"))
    rows.append(("H6", r"Gain L4 $-$ gain L1", four(lambda m, k: R[("T7", k)].loc[(m, "TDA-90"), "gain L4 - gain L1"]), ">0"))
    rows.append(("H7", r"Gain from TDA-90 $-$ gain from catch22, L4", four(lambda m, k: R[("T7", k)].loc[(m, "TDA-90 minus catch22"), "gain L4"]), ">0"))
    for c, e in (("V", ">0"), ("P", "desc"), ("T", "desc"), ("R", "desc")):
        rows.append(("H8", rf"Detector {c}, gain from TDA-90, level~4", four(lambda m, k: T9.loc[(k, m), c]), e))
    rows.append(("H9", r"Union $-$ TSFresh top-100, level~4", four(lambda m, k: R[("T11", k)].loc[m, "L4"]), ">0"))
    rows.append(("H10", r"Union $-$ TSFresh top-100, L4 $-$ L1", four(lambda m, k: R[("T11", k)].loc[m, "L4 - L1"]), ">0"))

    def verdict(cells, e):
        if e == "desc":
            return ""
        ok = [(v > 0 and p < 0.05) if e == ">0" else (v < 0 and p < 0.05) if e == "<0" else (p >= 0.05)
              for v, se, p in cells.values()]
        return f"{sum(ok)} of {len(ok)}"
    lines = [r"\begin{table}[p]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Every hypothesis of the analysis plan, its tests and their outcome. Differences at "
             r"the classifier and measure of the column; $^{*}p<0.05$, $^{**}p<0.01$, Nadeau--Bengio test, 5 "
             r"folds. Expected: the outcome the plan registered (``$>0$'': topology ahead; ``$<0$'': "
             r"TSFresh-774 ahead; ``n.s.'': no significant difference; ``---'': no direction registered). "
             r"As expected: number of tests whose outcome matched the registered expectation. Per-detector differences of H4 were "
             r"registered in balanced accuracy only. H11 is descriptive: shares of selected features, mean over folds, without a test.}",
             r"\label{tab:hypotheses}",
             r"\begin{tabular}{l p{5.2cm} cccc c c}", r"\toprule",
             r"& & \multicolumn{2}{c}{Balanced accuracy} & \multicolumn{2}{c}{AUC} & & \\",
             r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
             r"Hyp. & Test & LR & HGB & LR & HGB & Expected & As expected \\", r"\midrule"]
    last = None
    for h, lab, cells, e in rows:
        if last and h != last:
            lines.append(r"\addlinespace")
        vals = [_cell(*cells[c]) if c in cells else "" for c in COLS]
        exp = {">0": "$>0$", "<0": "$<0$", "ns": "n.s.", "desc": "---"}[e]
        lines.append(" & ".join([h if h != last else "", lab] + vals + [exp, verdict(cells, e)]) + r" \\")
        last = h
    share = T12["TDA share"].reindex(LETTERS)
    rank_v = int(share.rank(ascending=False, method="min")["V"])
    lines += [r"\addlinespace",
              r"H11 & Share of TDA-90 in union top-100 (pool share 10.4\,\%) & \multicolumn{4}{>{\raggedright\arraybackslash}p{5.0cm}}{"
              + ", ".join(f"{c} {share[c]:.2f}" for c in LETTERS) + r"} & V highest & "
              + ("yes" if rank_v == 1 else f"no (V {rank_v}rd)" if rank_v == 3 else f"no (V rank {rank_v})") + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "tableA1_hypotheses.tex").write_text(fit_width("\n".join(lines) + "\n"))
    print("  -> paper/tables/tableA1_hypotheses.tex")
    for h, lab, cells, e in rows:
        print(f"    {h:4s} {verdict(cells, e):7s} {lab}")


# ------------------------------------------------------------------------ Table A2 ---
def tableA2() -> None:
    ext = {"TDA-90": 0.0, "TSFresh-774": 0.0, "catch22": 0.0}
    key = {"tda90": "TDA-90", "tsfresh": "TSFresh-774", "catch22": "catch22"}
    logs = sorted(glob.glob(str(RES / "logs" / "features_all72?.log")))
    assert len(logs) == 6
    for f in logs:
        for line in Path(f).read_text().splitlines():
            m = re.match(r"\s*(tda90|tsfresh|catch22)\s+(\d+) features \((\d+)s\)", line)
            if m:
                ext[key[m.group(1)]] += float(m.group(3))
    T = pd.concat([pd.read_csv(RES / "main" / "timing.csv"), pd.read_csv(RES / "complementarity" / "timing.csv")])
    fit = T.groupby(["set", "model"]).seconds.sum() / 60
    order = [("TDA-90", "lr"), ("TDA-90", "hgb"), ("TSFresh-774", "lr"), ("TSFresh-774", "hgb"), ("catch22", "lr"),
             ("catch22", "hgb"), ("MiniRocket", "ridge"), ("TSFresh-774+TDA-90", "lr"), ("TSFresh-774+TDA-90", "hgb"),
             ("TSFresh-774+catch22", "lr"), ("TSFresh-774+catch22", "hgb")]
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{Computational cost on 12 cores, 36\,000 series of length 500. Extraction: wall time "
             r"of feature extraction summed over the six generation batches. Detectors: wall time of fitting and "
             r"scoring the eight detectors in all five folds. MiniRocket's transform is fitted inside each fold "
             r"and counted under its detectors. Extraction is listed once per feature family; the combined sets reuse "
             r"the extracted features.}",
             r"\label{tab:cost}", r"\begin{tabular}{l l r r}", r"\toprule",
             r"Feature set & Classifier & Extraction (min) & Detectors (min) \\", r"\midrule"]
    for st, m in order:
        e = f"{ext[st] / 60:.1f}" if (st in ext and m == "lr") else ""   # extraction once per family
        lines.append(" & ".join([st.replace("+", " + "), CLF[m], e, (f"{fit[(st, m)]:.1f}" if fit[(st, m)] >= 0.05 else r"$<$0.1")]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "tableA2_cost.tex").write_text("\n".join(lines) + "\n")
    print("  -> paper/tables/tableA2_cost.tex"); print("    extraction (s):", ext); print(fit.round(1).to_string())


# ------------------------------------------------- compact main-text tables 2, 3, 4 ---
def _v(cell):
    v, se, p = parse(cell); return f"{fs(v)}{star(p)}"


def table2c() -> None:
    T1 = pd.read_csv(RES / "main" / "tables" / "T1_T2_mean_of_eight_bal_acc.csv").set_index(["set", "model"])
    T23 = pd.read_csv(RES / "main" / "tables" / "T2_T3_differences_bal_acc.csv").set_index("comparison")
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Mean balanced accuracy of the eight detectors with one cause (L1) and with four "
             r"(L4), and its change (drop, L4 $-$ L1). The last two columns give TDA-90 minus the set of the row, "
             r"under the same classifier (for MiniRocket: TDA-90 under LR and under HGB, one line each); a positive $\Delta$drop "
             r"means that TDA-90 loses less. Registered comparisons H1--H3 (Table~\ref{tab:hypotheses}). $^{*}p<0.05$, $^{**}p<0.01$. Levels 2--3, standard errors, "
             r"$p$ values and AUC: Figure~\ref{fig:levels}a,b and Table~\ref{tab:levels-full}.}",
             r"\label{tab:levels-results}",
             r"\begin{tabular}{ll ccc cc}", r"\toprule",
             r"& & & & & \multicolumn{2}{c}{TDA-90 $-$ row} \\ \cmidrule(l){6-7}",
             r"Set & Classifier & L1 & L4 & Drop & $\Delta$drop & $\Delta$L4 \\", r"\midrule"]
    for i, (s_, m) in enumerate(ORDER2):
        r = T1.loc[(s_, m)]
        cells = [s_, CLF[m], f"{parse(r['L1'])[0]:.3f}", f"{parse(r['L4'])[0]:.3f}", fs(parse(r["drop L1->L4"])[0])]
        if s_ == "TDA-90":
            cells += ["---", "---"]
        elif s_ == "MiniRocket":           # one row per TDA-90 classifier it is compared with
            for k, tm in enumerate(("lr", "hgb")):
                x = T23.loc[f"TDA-90 ({tm}) - MiniRocket (ridge)"]
                row = (cells if k == 0 else ["", "", "", "", ""]) + [_v(x["drop difference"]) + f" ({CLF[tm]})",
                                                                     _v(x["level 4"]) + f" ({CLF[tm]})"]
                if k == 0 and i in (3, 6):
                    lines.append(r"\addlinespace")
                lines.append(" & ".join(row) + r" \\")
            continue
        else:
            c = T23.loc[f"TDA-90 ({m}) - {s_} ({m})"]
            cells += [_v(c["drop difference"]), _v(c["level 4"])]
        if i in (3, 6):
            lines.append(r"\addlinespace")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "table2_levels.tex").write_text("\n".join(lines) + "\n"); print("  -> paper/tables/table2_levels.tex")


def table3c() -> None:
    T = pd.read_csv(RES / "complementarity" / "tables" / "T7_T8_gains_bal_acc.csv").set_index(["model", "added"])
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Gain in mean balanced accuracy from adding TDA-90, or catch22 as a control, to "
             r"TSFresh-774, with one cause (L1) and four (L4), and the change of the gain. $^{*}p<0.05$, "
             r"$^{**}p<0.01$. All levels, standard errors, $p$ values and AUC: Figure~\ref{fig:levels}c,d "
             r"and Table~\ref{tab:complement-full}.}",
             r"\label{tab:complement}",
             r"\begin{tabular}{ll ccc}", r"\toprule",
             r"Classifier & Added to TSFresh-774 & L1 & L4 & L4 $-$ L1 \\", r"\midrule"]
    for m in ("lr", "hgb"):
        for a, lab in (("TDA-90", "TDA-90 (H5, H6)"), ("catch22", "catch22"), ("TDA-90 minus catch22", r"TDA-90 $-$ catch22 (H7)")):
            r = T.loc[(m, a)]
            cells = [CLF[m] if a == "TDA-90" else "", lab] + ["" if pd.isna(r[c]) else _v(r[c]) for c in ("gain L1", "gain L4", "gain L4 - gain L1")]
            lines.append(" & ".join(cells) + r" \\")
        if m == "lr":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "table3_complement.tex").write_text("\n".join(lines) + "\n"); print("  -> paper/tables/table3_complement.tex")


def table4c() -> None:
    T10 = pd.read_csv(RES / "budget" / "tables" / "T10_mean_of_eight_bal_acc.csv").set_index(["model", "set"])
    T11 = pd.read_csv(RES / "budget" / "tables" / "T11_union_minus_tsfresh_bal_acc.csv").set_index("model")
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Mean balanced accuracy at an equal budget of 100 features selected by mutual "
             r"information, with the full sets for reference. $\Delta$: union top-100 minus TSFresh top-100; "
             r"$^{*}p<0.05$, $^{**}p<0.01$. All levels, standard errors, $p$ values and AUC: "
             r"Table~\ref{tab:budget-full}.}",
             r"\label{tab:budget}",
             r"\begin{tabular}{l cc cc}", r"\toprule",
             r"& \multicolumn{2}{c}{Logistic regression} & \multicolumn{2}{c}{Gradient boosting} \\",
             r"\cmidrule(lr){2-3} \cmidrule(l){4-5}",
             r"Features & L1 & L4 & L1 & L4 \\", r"\midrule"]
    for st, lab in ORDER4:
        cells = [lab] + [f"{parse(T10.loc[(m, st)][f'L{k}'])[0]:.3f}" for m in ("lr", "hgb") for k in (1, 4)]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(" & ".join([r"$\Delta$ (H9)"] + [_v(T11.loc[m][f"L{k}"]) for m in ("lr", "hgb") for k in (1, 4)]) + r" \\")
    lines.append(" & ".join([r"$\Delta$ L4 $-$ L1 (H10)", "", _v(T11.loc["lr"]["L4 - L1"]), "", _v(T11.loc["hgb"]["L4 - L1"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "table4_budget.tex").write_text("\n".join(lines) + "\n"); print("  -> paper/tables/table4_budget.tex")


# ------------------------------------------------------------------------ Figure 1 ---
def fig1() -> None:
    """Construction of TDA-90 and what it measures, on two series of the paper's data, chosen by
    the rule in experiments/checks/fig1_select_2026-10-16.py: a GARCH-AR volatility series at
    the 90th percentile of the feature that carries the V detector (std over windows of the
    window Rips H0 W1) and an AR stationary series at its median. figures/data/fig1_data.npz is
    written by that script from data/paper on the compute server."""
    D = np.load(FIG / "data" / "fig1_data.npz")
    W = D["windows"]; z = D["volatility_z"]
    w1v, w1s = D["volatility_H0feats"][:, 0], D["stationary_H0feats"][:, 0]
    iT, iC = int(w1v.argmax()), int(w1v.argmin())                 # turbulent and calm windows
    CV, CS = COLOR["volatility series"], COLOR["stationary series"]
    CT, CC = COLOR["turbulent window"], COLOR["calm window"]
    fig = plt.figure(figsize=(6.8, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.6, wspace=0.5)
    # (a) the volatility series, its nine windows, the turbulent and the calm window
    ax = fig.add_subplot(gs[0, :])
    lo, hi_ = z.min(), z.max()
    for i, col in ((iT, CT), (iC, CC)):
        a_, b_ = W[i]; ax.axvspan(a_, b_ - 1, color=col, alpha=0.22, lw=0)
    ax.plot(np.arange(len(z)), z, color=CV, lw=0.6)
    for i, (a_, b_) in enumerate(W):
        y = lo - (0.8 if i % 2 == 0 else 1.5)
        col = CT if i == iT else CC if i == iC else "0.65"
        ax.plot([a_, b_ - 1], [y, y], color=col, lw=2.2 if i in (iT, iC) else 1.2, solid_capstyle="butt")
        ax.text((a_ + b_) / 2, y + (0.33 if i % 2 == 0 else -0.33), str(i + 1), ha="center", va="center", fontsize=5.5,
                color=col if i in (iT, iC) else "0.45")
    ax.set_ylim(lo - 2.2, hi_ + 0.3); ax.set_xlim(0, len(z) - 1)
    ax.set_xlabel("Time"); ax.set_ylabel("Standardised value")
    ax.set_title(f"a  Volatility series (GARCH) and its 9 windows: window {iT + 1} turbulent, window {iC + 1} calm", loc="left")
    # (b) delay embedding of the two windows, projected on (z_t, z_t+1)
    ax = fig.add_subplot(gs[1, 0])
    for i, col, lab in ((iT, CT, f"window {iT + 1} (turbulent)"), (iC, CC, f"window {iC + 1} (calm)")):
        P = D[f"volatility_w{i}_cloud"]; ax.scatter(P[:, 0], P[:, 1], s=5, color=col, lw=0, label=lab, zorder=3 if i == iC else 2)
    ax.set_aspect("equal", adjustable="datalim"); ax.set_xlabel("$z_t$"); ax.set_ylabel("$z_{t+1}$")
    ax.legend(loc="upper left", handletextpad=0.1, borderaxespad=0.1, markerscale=1.5, fontsize=6.5)
    ax.set_title("b  Delay embedding", loc="left")
    # (c) Rips H0 of the two windows: lifetimes = minimum-spanning-tree edge lengths
    ax = fig.add_subplot(gs[1, 1])
    for i, col in ((iT, CT), (iC, CC)):
        H0 = D[f"volatility_w{i}_H0"]; life = np.sort(H0[np.isfinite(H0[:, 1]), 1])[::-1]
        ax.plot(np.arange(1, len(life) + 1), life, color=col, lw=1.6, drawstyle="steps-mid",
                label=f"window {i + 1}: $W_1$ = {w1v[i]:.1f}")
    ax.legend(loc="upper right", fontsize=6.5, handlelength=1.4, borderaxespad=0.1)
    ax.set_xlabel("Rank of $H_0$ bar"); ax.set_ylabel("Lifetime (edge length)")
    ax.set_title("c  Rips $H_0$ of the two windows", loc="left")
    # (d) the summary in every window -> mean and std over windows, for both series
    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(1, len(W) + 1)
    for v, col, lab in ((w1s, CS, "stationary series"), (w1v, CV, "volatility series")):
        ax.plot(x, v, marker="o", ms=3, color=col, lw=1.2, zorder=3)
        ax.text(9.4, v.mean() + (4.0 if lab.startswith("stationary") else -5.0),
                f"{lab}\nmean {v.mean():.1f}\nstd {v.std():.1f}", color=col, fontsize=6, va="center")
    for i, col in ((iT, CT), (iC, CC)):
        ax.scatter([i + 1], [w1v[i]], s=34, facecolor="none", edgecolor=col, lw=1.4, zorder=4)
    ax.set_xticks(x); ax.set_xlim(0.6, 9.4); ax.set_xlabel("Window"); ax.set_ylabel("$W_1$ of window $H_0$")
    ax.set_title("d  Over the windows", loc="left")
    ax.margins(y=0.15)
    save(fig, "fig1_construction")

def tableA6() -> None:
    """Generator settings. Rules: betise 0.6.0 source (core/generator.py, dataset_generation.py,
    config/params.json), read on the server 2026-10-16. Realised values: metadata of all 36,000
    series, summarised by experiments/checks/generator_settings_2026-10-16.py."""
    G = json.loads((ROOT / "experiments" / "checks" / "generator_settings_2026-10-16.json").read_text())
    assert G["betise_version"] == "0.6.0" and G["n_series"] == 36000

    def iv(d, k=2):        # 5th-95th percentile of a realised quantity
        return f"${d['q05']:.{k}f}$ to ${d['q95']:.{k}f}$"

    def n(x):              # 9000 -> 9\,000
        return f"{int(x):,}".replace(",", r"\,")

    ar = G["ar_order"]; nar = sum(ar.values())
    lt = G["linear_trend"]
    rows = [
        ("Base", r"AR($p$) or ARMA($p,q$), $p,q \in \{1,\dots,5\}$, coefficients drawn from $[-0.9, 0.9]$ and redrawn until the process is stationary (ARMA: also invertible)",
         f"AR coefficients {iv(G['ar_coef'])}; $p=1$~in~{100 * ar['1'] / nar:.0f}\\,\\%"),
        ("S", r"random walk ($\sigma = 1$) or ARI($p$,1), $p$ and coefficients as for the base",
         f"{n(G['stochastic_type']['random_walk'])} random walk, {n(G['stochastic_type']['ari'])} ARI"),
        ("V", r"GARCH(1,1): $\omega \in [0.3, 0.6]$, $\alpha \in [0.4, 0.6]$, $\beta \in [0.2, 0.5]$, redrawn until $\alpha + \beta < 1$; "
              r"EGARCH(1,1): $\omega \in [0.1, 0.3]$, $\alpha \in [0.2, 0.4]$, $\gamma \in [-0.3, 0.3]$, $\beta \in [0.75, 0.9]$",
         f"{n(G['garch']['n'])} GARCH, {n(G['egarch']['n'])} EGARCH"),
        ("T", r"slope $|b| \in [0.01, 0.1]$ with random sign, intercept in $[-1, 1]$",
         f"$|b|$ {iv(lt['abs_slope'], 3)}; upward {100 * lt['share_upward']:.0f}\\,\\%"),
        ("R", r"the slope of the linear trend reverses sign at one break",
         f"break at {iv(G['trend_shift']['position'])} of the length"),
        ("M", r"one break; jump $= u \cdot s \cdot \mathrm{sd}$(segment before the break), $u \in [1.5, 3]$, $s \in [1, 3]$, random sign",
         f"break at {iv(G['mean_shift']['position'])} of the length"),
        ("Q", r"one break; the residual after it is multiplied or divided by $u \cdot s$, $u \in [1.5, 3]$, $s \in [2, 5]$",
         f"break at {iv(G['variance_shift']['position'])} of the length"),
        ("P", r"one value set to the series mean $\pm\, u \cdot s \cdot \mathrm{sd}$, $u \in [1.5, 2.5]$, $s \in [1, 2]$, sign random",
         f"at {iv(G['point_anomaly']['position'])} of the length"),
        ("C", r"rectangular offset of length 5--9\,\% of the series, level $\pm\, u \cdot s \cdot$ local sd, $u \in [1, 1.75]$, $s \in [1, 2]$",
         f"length {G['collective_anomaly']['length']['q05']:.0f} to {G['collective_anomaly']['length']['q95']:.0f} points; "
         f"start at {iv(G['collective_anomaly']['position'])}"),
    ]
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             r"\caption{Generator settings (betise~0.6.0 defaults; series of length 500). Rule: how each "
             r"cause is drawn. Realised: 5th--95th percentiles across series with the cause, taken "
             r"from the series metadata. Positions are fractions of the series length. $u$ and $s$ are "
             r"independent uniform draws; sd is a standard deviation of the series at the stage the "
             r"overlay is applied.}",
             r"\label{tab:generator}", r"\begin{tabular}{l >{\raggedright\arraybackslash}p{0.52\linewidth} >{\raggedright\arraybackslash}p{0.30\linewidth}}", r"\toprule",
             r"Cause & Rule & Realised \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "tableA6_generator.tex").write_text("\n".join(lines) + "\n")
    print("  -> paper/tables/tableA6_generator.tex")


def table6() -> None:
    """Exploratory: classical tests as detectors, against the learned
    detectors under gradient boosting. Balanced accuracy at levels 1 and 4, mean over folds; mean
    of the seven causes that have a standard test (C has none). Sources: results/paper/tests/
    folds_tests.csv (experiments/exploratory/stat_tests.py) and main/complementarity folds.csv."""
    T = pd.read_csv(RES / "tests" / "folds_tests.csv")
    F = pd.concat([pd.read_csv(RES / "main" / "folds.csv"), pd.read_csv(RES / "complementarity" / "folds.csv")]).drop_duplicates()
    F = pd.concat([F, T])
    seven = list("SVTRMQP")
    cols = [("Statistical tests", "5% level", "Tests"), ("TDA-90", "hgb", "TDA-90"),
            ("TSFresh-774", "hgb", "TSFresh-774"), ("TSFresh-774+TDA-90", "hgb", r"\shortstack{TSFresh-774\\+ TDA-90}")]
    names = {"S": "S stochastic trend", "V": "V volatility", "T": "T linear trend", "R": "R trend shift",
             "M": "M mean shift", "Q": "Q variance shift", "P": "P point anomaly", "C": "C collective anomaly"}
    tests = {"S": "ADF", "V": "ARCH-LM", "T": "slope", "R": "sup-LM", "M": "CS$_M$", "Q": "CS$_Q$",
             "P": "Grubbs", "C": "---"}

    def cell(s_, m, ch, lv, metric="bal_acc"):
        d = F[(F.set == s_) & (F.model == m) & (F.char == ch) & (F.level == lv)][metric]
        return f"{d.mean():.3f}" if len(d) else "---"

    def mean7(s_, m, lv, metric):
        d = F[(F.set == s_) & (F.model == m) & (F.level == lv) & F.char.isin(seven)]
        g = d.groupby("fold")[metric]
        assert (g.size() == 7).all()
        return f"{g.mean().mean():.3f}"

    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Exploratory. Classical tests (one per cause, 5\,\% level) and learned detectors (gradient "
             r"boosting): balanced accuracy with one cause (L1) and with four (L4), mean over five folds. Last "
             r"rows: mean of the seven causes with a test, in balanced accuracy and in AUC (tests ranked by "
             r"their statistic). Tests: Section~\ref{sec:protocol-tests} and Appendix~\ref{app:tests}.}",
             r"\label{tab:tests}",
             r"\begin{adjustbox}{max width=\textwidth}",
             r"\begin{tabular}{l l cc cc cc cc}", r"\toprule",
             r"& & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{c[2]}}}" for c in cols) + r" \\",
             r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(l){9-10}",
             r"Cause & Test & " + " & ".join(["L1 & L4"] * len(cols)) + r" \\", r"\midrule"]
    for ch in "SVTRMQPC":
        lines.append(" & ".join([names[ch], tests[ch]] + [cell(s_, m, ch, lv) for s_, m, _ in cols for lv in (1, 4)]) + r" \\")
    lines.append(r"\midrule")
    for metric, lab in (("bal_acc", "Mean of seven, balanced accuracy"), ("auc", "Mean of seven, AUC")):
        lines.append(rf"\multicolumn{{2}}{{l}}{{{lab}}} & " + " & ".join(mean7(s_, m, lv, metric) for s_, m, _ in cols for lv in (1, 4)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    (TAB / "table6_tests.tex").write_text("\n".join(lines) + "\n")
    print("  -> paper/tables/table6_tests.tex")


def table7() -> None:
    """Exploratory. Share of the series of each single-cause class (and of the stationary class)
    flagged by each classical test at the 5% level; the diagonal is the power of a test against
    its own cause. Source: results/paper/tests/r_crosscheck/cross_sensitivity_L1_python.csv
    (experiments/exploratory/compare.py, from the Python decisions)."""
    X = pd.read_csv(RES / "tests" / "r_crosscheck" / "cross_sensitivity_L1_python.csv", index_col=0)
    rows = [("stationary", "Stationary", None), ("stochastic_trend", "S stochastic trend", "S"), ("volatility", "V volatility", "V"),
            ("trend", "T linear trend", "T"), ("trend_shift", "R trend shift", "R"), ("mean_shift", "M mean shift", "M"),
            ("variance_shift", "Q variance shift", "Q"), ("point_anomaly", "P point anomaly", "P"),
            ("collective_anomaly", "C collective anomaly", None)]
    tests = [("S", "ADF"), ("V", "ARCH-LM"), ("T", "slope"), ("R", "sup-LM"), ("M", "CS$_M$"), ("Q", "CS$_Q$"), ("P", "Grubbs")]
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Exploratory. Proportion of the 500 series in each single-cause class and in the "
             r"stationary class flagged by each classical test at the 5\,\% level. Diagonal cells (bold): "
             r"the power of a test against its own cause; other cells: its response to causes it does not "
             r"target. Trend-shift series contain a linear trend by construction (Section~\ref{sec:data-causes}).}",
             r"\label{tab:tests-cross}",
             r"\begin{adjustbox}{max width=\textwidth}",
             r"\begin{tabular}{l ccccccc}", r"\toprule",
             r"& \multicolumn{7}{c}{Test (cause it targets)} \\ \cmidrule(l){2-8}",
             r"Class & " + " & ".join(rf"\shortstack{{{lab}\\({k})}}" for k, lab in tests) + r" \\", r"\midrule"]
    for key, lab, own in rows:
        cells = [lab]
        for k, _ in tests:
            v = f"{X.loc[key, k]:.2f}"
            cells.append(rf"\textbf{{{v}}}" if k == own else v)
        lines.append(" & ".join(cells) + r" \\")
        if key == "stationary":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    (TAB / "table7_tests_cross.tex").write_text("\n".join(lines) + "\n")
    print("  -> paper/tables/table7_tests_cross.tex")


def _separate_end_labels(ax, gap: float) -> None:
    ts = sorted([t for t in ax.texts], key=lambda t: t.get_position()[1])
    for a, b in zip(ts, ts[1:]):
        if b.get_position()[1] - a.get_position()[1] < gap:
            b.set_y(a.get_position()[1] + gap)


def fig23() -> None:
    """Figure 3 (merged 2026-10-19 from the former fig2_levels and fig3_complement): top row,
    mean balanced accuracy of the eight detectors at each level (H1-H3); bottom row, gain over
    TSFresh-774 from adding TDA-90 or the catch22 control (H5-H7). Columns: logistic regression,
    gradient boosting. Lines: means over the five folds; points: fold values."""
    M = mean_of_eight(pd.read_csv(RES / "main" / "folds.csv"), "bal_acc").reset_index()
    G = gains("bal_acc")
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.2), sharey="row",
                             gridspec_kw={"wspace": 0.55, "hspace": 0.45})
    lv = np.arange(1, 5)
    name = {"lr": "Logistic regression", "hgb": "Gradient boosting"}
    for k, model in enumerate(("lr", "hgb")):
        ax = axes[0, k]
        jit = np.random.default_rng(0)
        for s_, m in (("MiniRocket", "ridge"), ("catch22", model), ("TSFresh-774", model), ("TDA-90", model)):
            d = M[(M.set == s_) & (M.model == m)]
            focal = s_ == "TDA-90"
            ax.scatter(d.level + jit.uniform(-0.06, 0.06, len(d)), d.bal_acc, s=4, color=COLOR[s_], alpha=0.35, lw=0, zorder=2)
            mean = d.groupby("level").bal_acc.mean().reindex(lv)
            ax.plot(lv, mean, color=COLOR[s_], lw=2.0 if focal else 1.2, marker="o", ms=3.5 if focal else 2.8,
                    ls="--" if s_ == "MiniRocket" else "-", zorder=4 if focal else 3)
            ax.text(4.24, mean.iloc[-1], "MiniRocket (ridge)" if s_ == "MiniRocket" else s_, color=COLOR[s_],
                    fontsize=7, va="center", fontweight="bold" if focal else "normal")
        ax.set_title(f"{'ab'[k]}  {name[model]}: mean of the eight detectors", loc="left")
        ax.margins(y=0.05)
        _separate_end_labels(ax, 0.016)
        ax = axes[1, k]
        jit = np.random.default_rng(0)
        ax.axhline(0, color="0.6", lw=0.6, zorder=1)
        for a, lab in (("catch22", "+ catch22 (control)"), ("TDA-90", "+ TDA-90")):
            d = G[(G.added == a) & (G.model == model)]
            focal = a == "TDA-90"
            ax.scatter(d.level + jit.uniform(-0.06, 0.06, len(d)), d.gain, s=4, color=COLOR[a], alpha=0.35, lw=0, zorder=2)
            mean = d.groupby("level").gain.mean().reindex(lv)
            ax.plot(lv, mean, color=COLOR[a], lw=2.0 if focal else 1.2, marker="o", ms=3.5 if focal else 2.8,
                    zorder=4 if focal else 3)
            ax.text(4.24, mean.iloc[-1], lab, color=COLOR[a], fontsize=7, va="center", fontweight="bold" if focal else "normal")
        ax.set_title(f"{'cd'[k]}  {name[model]}: gain over TSFresh-774", loc="left")
        ax.margins(y=0.08)
    for ax in axes.ravel():
        ax.set_xticks(lv); ax.set_xlim(0.8, 4.15)
    for ax in axes[1]:
        ax.set_xlabel("Level (number of combined causes)")
    axes[0, 0].set_ylabel("Balanced accuracy")
    axes[1, 0].set_ylabel("Gain in balanced accuracy")
    save(fig, "fig3_levels_gain")


JOBS = {"fig1": fig1, "fig23": fig23, "table7": table7, "table6": table6, "tableA6": tableA6, "table2c": table2c, "table3c": table3c, "table4c": table4c, "tableA1": tableA1, "tableA2": tableA2, "fig5": fig5, "table5": table5, "table4": table4, "fig4": fig4, "table1": table1, "table2": table2, "table3": table3}

if __name__ == "__main__":
    style()
    FIG.mkdir(exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    for name in (sys.argv[1:] or JOBS):
        print(name); JOBS[name]()
