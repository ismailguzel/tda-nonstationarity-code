"""
Topological features of the paper: TDA-90.

The per-diagram statistics, sublevel H0, the z-score, the windows and the paper features
between the two "copied verbatim" markers come unchanged from the development code of the
study; tda90() and its helpers at the end assemble the 90 features, in a fixed order and on
a signed-log scale.

TDA-90 (one series, z-scored once as a whole):
    28  Rips H0/H1 of each window (9 windows of n/5, 50% overlap; delay 1, dimension 3):
        the 14 distinct features of Guzel & Kaygun 2023 (W1, bottleneck, Carlsson f1-f4,
        entropy, landscape L1 — H0 carl_f1/f3 are identically 0, Betti L1 = 2 x W1),
        mean and std over the windows
    36  sublevel H0 of each window: 8 lifetime statistics + carl_f1; mean, std, range and
        largest consecutive change over the windows
    26  sublevel H0 of the whole series: Carlsson f1-f5, entropy, landscape L1/L2,
        8 lifetime statistics, 10-point Betti curve
Every feature is stored on a signed-log scale, sign(v) * log1p(|v|), as in the study.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- copied verbatim ---

def carlsson_coordinates(diagram: np.ndarray) -> np.ndarray:
    """5 Carlsson coordinates for a persistence diagram.

    As defined in Carlsson et al. (see also Umeda 2017):
        p_i     = d_i - b_i          (lifetime / persistence)
        d_max   = max(d_i)

        f1 = Σ  b_i · p_i
        f2 = Σ  (d_max - d_i) · p_i
        f3 = Σ  b_i² · p_i⁴
        f4 = Σ  (d_max - d_i)² · p_i⁴
        f5 = max p_i                 (most persistent bar)

    These are provably stable under small Wasserstein perturbations of the
    diagram.  f1 encodes birth-weighted persistence (trend/level structure);
    f2 encodes how far deaths are from the maximum filtration value; f3/f4
    are higher-degree versions that emphasise long-lived bars; f5 captures
    the single most prominent topological feature.
    """
    diag = diagram[np.isfinite(diagram).all(axis=1)]
    diag = diag[diag[:, 1] > diag[:, 0]]

    if len(diag) == 0:
        return np.zeros(5)

    b, d  = diag[:, 0], diag[:, 1]
    p     = d - b
    d_max = float(d.max())

    return np.array([
        np.sum(b * p),                    # f1
        np.sum((d_max - d) * p),          # f2
        np.sum(b ** 2 * p ** 4),          # f3
        np.sum((d_max - d) ** 2 * p ** 4),# f4
        float(p.max()),                   # f5
    ])


def persistent_entropy(diagram: np.ndarray) -> float:
    """Shannon entropy of normalised bar lifetimes: H = -Σ (pᵢ/L) log(pᵢ/L).

    Low entropy  → few dominant bars (e.g. one long bar from a mean shift).
    High entropy → many bars of similar length (e.g. white noise).
    """
    diag = diagram[np.isfinite(diagram).all(axis=1)]
    diag = diag[diag[:, 1] > diag[:, 0]]
    if len(diag) == 0:
        return 0.0
    p = diag[:, 1] - diag[:, 0]
    L = p.sum()
    if L < 1e-12:
        return 0.0
    probs = p / L
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _persistence_landscape_vector(
    diagram: np.ndarray,
    n_landscapes: int,
    resolution: int,
    use_mean: bool,
) -> np.ndarray:
    """Mean persistence landscape discretised into `resolution` bins.

    Tent function per bar: λ(x) = min(x−b, d−x) if b ≤ x ≤ d, else 0.
    Top-n_landscapes values stored per grid point; averaged if use_mean=True.
    """
    diag = diagram[np.isfinite(diagram).all(axis=1)]
    diag = diag[diag[:, 1] > diag[:, 0]]

    out_len = resolution if use_mean else n_landscapes * resolution
    if len(diag) == 0:
        return np.zeros(out_len)

    min_b, max_d = diag[:, 0].min(), diag[:, 1].max()
    if max_d <= min_b:
        return np.zeros(out_len)

    grid = np.linspace(min_b, max_d, resolution)
    lands = np.zeros((n_landscapes, resolution))
    births, deaths = diag[:, 0], diag[:, 1]

    for i, x in enumerate(grid):
        active = (births <= x) & (x <= deaths)
        if active.any():
            vals = np.sort(np.minimum(x - births[active], deaths[active] - x))[::-1]
            k = min(n_landscapes, len(vals))
            lands[:k, i] = vals[:k]

    return lands.mean(axis=0) if use_mean else lands.flatten()


BETTI_RESOLUTION = 10


def persistence_statistics(diagram: np.ndarray) -> np.ndarray:
    """Order and scale statistics of the lifetime distribution.

    The original 8 scalars omit the plainest quantity a diagram has: how many
    bars it contains. That omission is not cosmetic here — measurement showed the
    number of significant sub_H0 bars equals the number of local minima minus one
    (a sine of period P over n samples gives n/P bars), so bar count is a direct
    readout of oscillation structure, and none of carl_f1..f5, entropy or the
    landscape norms carries it. Entropy comes closest, but only under equal
    lifetimes, where it degenerates to log(n).

    The second-longest bar and its ratio to the longest are included for the same
    reason: they are what distinguishes one loop from two (a torus gives a ratio
    near 1, a single closed curve near 0), and carl_f5_max keeps only the largest.

    Returns 8 values, see _PERS_STAT_SUFFIXES.
    """
    diag = diagram[np.isfinite(diagram).all(axis=1)]
    diag = diag[diag[:, 1] > diag[:, 0]]
    if len(diag) == 0:
        return np.zeros(8)

    p = np.sort(diag[:, 1] - diag[:, 0])[::-1]
    b, d = diag[:, 0], diag[:, 1]
    mean = float(p.mean())
    std = float(p.std())
    skew = float(np.mean(((p - mean) / std) ** 3)) if std > 1e-12 else 0.0
    second = float(p[1]) if len(p) > 1 else 0.0

    return np.array([
        float(len(p)),                    # n_bars
        float(p.sum()),                   # total_persistence
        mean,                             # pers_mean
        std,                              # pers_std
        skew,                             # pers_skew
        float(np.quantile(p, 0.90)),      # pers_q90
        second,                           # pers_second
        second / (p[0] + 1e-12),          # pers_ratio_21
    ])


def betti_curve(diagram: np.ndarray, resolution: int = BETTI_RESOLUTION) -> np.ndarray:
    """How many bars are alive at each of `resolution` points across the diagram.

    Sampled on the diagram's own [min birth, max death] range rather than on a
    fixed grid, so the curve describes shape and the statistics above carry the
    scale. A fixed grid would not be comparable across diagrams anyway: for
    sub/sup_H0 the filtration parameter is in the units of the series, while for
    H1 it is a distance in the delay embedding.

    Motivation: a mean shift puts the series at two levels, so its bars are born
    in two clusters and the curve is bimodal; the 8 scalars average that away.
    Measured on betise 0.5.0: the curve turns bimodal once the realised shift
    exceeds about 2 noise standard deviations, as for a two-Gaussian mixture.
    The valley is not specific to mean shifts — any series sitting at separated
    levels (trends, random walks) shows one.
    """
    diag = diagram[np.isfinite(diagram).all(axis=1)]
    diag = diag[diag[:, 1] > diag[:, 0]]
    if len(diag) == 0:
        return np.zeros(resolution)
    lo, hi = diag[:, 0].min(), diag[:, 1].max()
    if hi <= lo:
        return np.zeros(resolution)
    grid = np.linspace(lo, hi, resolution)
    births, deaths = diag[:, 0][None, :], diag[:, 1][None, :]
    alive = (births <= grid[:, None]) & (grid[:, None] < deaths)
    return alive.sum(axis=1).astype(float)


def diagram_to_stats(diagram: np.ndarray,
                     n_landscapes: int = 5,
                     resolution: int = 100) -> np.ndarray:
    """Summary vector for one persistence diagram.

    [carlsson x5, entropy, landscape_L1, landscape_L2]   the original 8
    [n_bars ... pers_ratio_21]                           8 lifetime statistics
    [betti_0 ... betti_9]                                10-point Betti curve

    Both blocks are always computed. Downstream code selects by feature name, so
    the original 8 and the extended set come from one extraction pass rather than
    two, and the comparison between them is exact.
    """
    carl    = carlsson_coordinates(diagram)
    entropy = persistent_entropy(diagram)
    lvec    = _persistence_landscape_vector(diagram, n_landscapes, resolution, use_mean=True)
    l1      = float(np.sum(np.abs(lvec)))
    l2      = float(np.sqrt(np.sum(lvec ** 2)))
    return np.concatenate([carl, [entropy, l1, l2],
                           persistence_statistics(diagram),
                           betti_curve(diagram)])


def sublevel_persistence_1d(ts: np.ndarray) -> np.ndarray:
    """H0 persistence via sublevel set filtration on a 1D path complex.

    Models the time series as f: {0,...,n-1} → R.  Union-find with the
    elder rule: the younger component (higher min = born later) dies when it
    merges with an older one.

    Returns array of shape (n_pairs, 2) with columns [birth, death].
    The essential class (global minimum) is omitted.

    What it measures: each local maximum is a barrier, and the bar it ends has
    lifetime = barrier height above the higher of the two basins it separates.
    An upward spike therefore gives one bar of lifetime equal to its height; the
    bar count is the number of local minima (incl. endpoints) minus one. A clean
    level shift leaves no bar — the second plateau merges on birth.
    """
    n = len(ts)
    if n < 2:
        return np.zeros((0, 2))

    parent      = list(range(n))
    comp_min    = ts.tolist()   # min function value per component root

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    edges = sorted((max(ts[i], ts[i + 1]), i, i + 1) for i in range(n - 1))

    pairs: list[tuple[float, float]] = []
    for fval, i, j in edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        bi, bj = comp_min[ri], comp_min[rj]
        if bi >= bj:           # ri is younger
            pairs.append((bi, fval))
            parent[ri] = rj
        else:                  # rj is younger
            pairs.append((bj, fval))
            parent[rj] = ri

    return np.array(pairs, dtype=float) if pairs else np.zeros((0, 2))


def _zscore(ts: np.ndarray) -> np.ndarray:
    """Per-series standardisation: (x - mean) / std.

    Sub- and superlevel persistence is measured in the units of the signal, so
    without this the absolute amplitude of a series flows straight into its
    topological features. Whether that is wanted depends on the question:

      On (pipeline default, ZSCORE=1 in modes.sh) — measured better for topology
                      (0.820 -> 0.873, 9-class), because the p^4 terms in carl_f3/f4
                      otherwise span up to 1e20 across series.
      Off           — reproduces raw-amplitude behaviour.

    One consequence to keep in mind when interpreting features: a shift or a
    spike inflates the series' standard deviation, so after standardising the
    noise — and every noise bar — shrinks by sigma_noise / sigma_total. Bar
    lengths therefore carry a variance-ratio signal through the normalisation
    itself, separate from the topology of the event.

    Running both ways on the same data is also the only way to see how much of a
    diagram family's discriminative power is shape and how much is scale.
    """
    ts = np.asarray(ts, dtype=float)
    std = ts.std()
    return (ts - ts.mean()) / std if std > 1e-12 else ts - ts.mean()


_CLASSIC_SUFFIXES = [
    'carl_f1', 'carl_f2', 'carl_f3', 'carl_f4', 'carl_f5_max',
    'entropy', 'landscape_l1', 'landscape_l2',
]


_PERS_STAT_SUFFIXES = [
    'n_bars', 'total_pers', 'pers_mean', 'pers_std', 'pers_skew',
    'pers_q90', 'pers_second', 'pers_ratio_21',
]


_BETTI_SUFFIXES = [f'betti_{i}' for i in range(BETTI_RESOLUTION)]


_STAT_SUFFIXES = _CLASSIC_SUFFIXES + _PERS_STAT_SUFFIXES + _BETTI_SUFFIXES


LIFETIME = ["n_bars", "total_pers", "pers_mean", "pers_std", "pers_skew", "pers_q90", "pers_second", "pers_ratio_21"]


PER_DIAGRAM = LIFETIME + ["carl_f1"]           # carl_f1 = sum(birth * lifetime): the block's one level-bearing feature


AGGREGATES = ["mean", "std", "range", "maxjump"]


def windows(n: int, width: int, step: int) -> list[tuple[int, int]]:
    return [(s, s + width) for s in range(0, n - width + 1, step)]


H1_DIM = 3          # fixed: FNN picked its maximum (10) for every noise-driven series, which


PAPER = ["W1", "bottleneck", "carl_f1", "carl_f2", "carl_f3", "carl_f4", "entropy", "betti_L1", "landscape_L1"]


def paper_features(diagram: np.ndarray) -> list[float]:
    """The per-diagram features of Guzel & Kaygun 2023 (Concurrency Computat Pract Exper 35:e7732),
    Section 3.3: Wasserstein-1 and bottleneck distance to the empty diagram, Carlsson f1-f4,
    persistent entropy, and the L1 norms of the Betti curve and of the first landscape. The
    Betti curve's L1 norm equals the total lifetime, so it is 2 x W1 — kept to match the paper."""
    d = diagram[np.isfinite(diagram).all(axis=1)]
    d = d[d[:, 1] > d[:, 0]]
    if len(d) == 0:
        return [0.0] * len(PAPER)
    life = d[:, 1] - d[:, 0]
    t = np.linspace(d[:, 0].min(), d[:, 1].max(), 256)
    lam1 = np.clip(np.minimum(t[:, None] - d[:, 0], d[:, 1] - t[:, None]), 0, None).max(axis=1)
    return ([float(life.sum() / 2), float(life.max() / 2)] + [float(v) for v in carlsson_coordinates(d)[:4]]
            + [persistent_entropy(d), float(life.sum()), float(np.trapezoid(lam1, t))])


def paper_win(x: np.ndarray, tau: int = 1) -> np.ndarray:
    """The paper's 18 features (9 per diagram, Rips H0 and H1) on each A1 window — a
    3-dimensional delay embedding with delay 1, as in rips_betti_win_t1 — then mean, std,
    range and largest consecutive change over the windows: 72 features."""
    from ripser import ripser
    z = _zscore(np.asarray(x, float))
    width = len(z) // 5
    W = []
    for a, b in windows(len(z), width, width // 2):
        w = z[a:b]
        n = len(w) - (H1_DIM - 1) * tau
        dg = ripser(np.column_stack([w[i * tau: i * tau + n] for i in range(H1_DIM)]), maxdim=1)["dgms"]
        W.append(paper_features(dg[0]) + paper_features(dg[1]))
    W = np.array(W)
    jumps = np.abs(np.diff(W, axis=0)).max(axis=0)
    return np.concatenate([W.mean(0), W.std(0), W.max(0) - W.min(0), jumps])


def paper_win_names() -> list[str]:
    cols = [f"pwin_{d}__{f}" for d in ("H0", "H1") for f in PAPER]
    return [f"{c}__{a}" for a in AGGREGATES for c in cols]


PAPER_DROP = ("pwin_H0__carl_f1__", "pwin_H0__carl_f3__", "pwin_H0__betti_L1__", "pwin_H1__betti_L1__")


def paper_columns(aggs) -> list[int]:
    """Indices into paper_win_names() of the 14 distinct features under the given aggregates."""
    return [i for i, n in enumerate(paper_win_names())
            if not n.startswith(PAPER_DROP) and n.rsplit("__", 1)[1] in aggs]

# ------------------------------------------------------------ end copied verbatim ---


def signed_log(v: np.ndarray) -> np.ndarray:
    return np.sign(v) * np.log1p(np.abs(v))


def windowed_sub_h0(x: np.ndarray) -> np.ndarray:
    """The sublevel half of blocks.windowed_h0: per window the 8 lifetime statistics and
    carl_f1 of sublevel H0; mean, std, range and largest consecutive change over windows.
    Each aggregate is per column, so this equals the win_sub_H0 columns of windowed_h0."""
    z = _zscore(np.asarray(x, float))
    width = len(z) // 5
    W = []
    for a, b in windows(len(z), width, width // 2):
        d = sublevel_persistence_1d(z[a:b])
        W.append(list(persistence_statistics(d)) + [float(carlsson_coordinates(d)[0])])
    W = np.array(W)
    jumps = np.abs(np.diff(W, axis=0)).max(axis=0) if len(W) > 1 else np.zeros(W.shape[1])
    return np.concatenate([W.mean(0), W.std(0), W.max(0) - W.min(0), jumps])


def windowed_sub_h0_names() -> list[str]:
    return [f"win_sub_H0__{f}__{a}" for a in AGGREGATES for f in PER_DIAGRAM]


def whole_sub_h0(x: np.ndarray) -> np.ndarray:
    """The 26 sub_H0 features of the study's topology-78 (z-scored, no self-normalisation)."""
    return diagram_to_stats(sublevel_persistence_1d(_zscore(np.asarray(x, float))), 5, 100)


def whole_sub_h0_names() -> list[str]:
    return [f"topo__sub_H0__{s}" for s in _STAT_SUFFIXES]


PAPER_MEAN_STD = paper_columns(("mean", "std"))


def tda90(x: np.ndarray) -> np.ndarray:
    """The 90 features of one series, signed-log scaled, in the order of tda90_names()."""
    raw = np.concatenate([paper_win(x)[PAPER_MEAN_STD], windowed_sub_h0(x), whole_sub_h0(x)])
    return signed_log(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0))


def tda90_names() -> list[str]:
    names = [paper_win_names()[i] for i in PAPER_MEAN_STD] + windowed_sub_h0_names() + whole_sub_h0_names()
    assert len(names) == 90
    return names
