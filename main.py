import os, csv, time
import numpy as np
import h5py

# ── CONFIG (you can edit these) ───────────────────────────────────────────────
MAT_PATH    = "VitalDB_AAMI_Test_Subset.mat"  # your .mat file name or full path
CHANNEL     = 0       # pick the signal channel to use
SAMPLES     = 800     # how many samples per segment to keep
MIN_CLUSTER = 20      # stop splitting when cluster size <= this
MAX_DEPTH   = 10      # maximum split depth
OUT_DIR     = "out"   # where results/plots will be saved
LIMIT_SEGMENTS = 1000 # cap the dataset to 1,000 segments

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_signals_from_mat(path, channel_hint=0, samples=800):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with h5py.File(path, "r") as f:
        data = f["Subset"]["Signals"][:]  # confirmed correct path

    arr = np.asarray(data)
    # file structure: (samples, channels, segments)
    arr = np.transpose(arr, (2, 1, 0))

    ch = channel_hint if arr.shape[1] > channel_hint else 0
    X = arr[:, ch, :samples]  # (N, S)
    return X


# 2) Basic helper functions used across clustering and analysis

def zscore_rows(X, eps=1e-8):
    # Normalize each time series so they all have mean 0 and standard deviation 1.
    # eps (a very small number) prevents division by zero in case std = 0.
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + eps
    Z = (X - mu) / sd
    # replace any NaN or infinity values with 0 so they don’t break clustering
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

def corr_distance(a, b):
    # Calculates how different two time series are based on correlation.
    # If correlation = 1 -> distance = 0 (very similar)
    # If correlation = 0 -> distance = 1 (completely different)
    a0 = (a - a.mean()) / (a.std() + 1e-8)
    b0 = (b - b.mean()) / (b.std() + 1e-8)
    return 1.0 - float(np.dot(a0, b0) / len(a0))

def farthest_pair(X):
    # Used in divide-and-conquer clustering.
    # It picks two signals that are the most different from each other
    # (based on correlation distance) to use as initial “centers” for splitting.
    if len(X) < 2:
        return (0, 0)
    a = 0
    b = max(range(len(X)), key=lambda j: corr_distance(X[a], X[j]))
    c = max(range(len(X)), key=lambda j: corr_distance(X[b], X[j]))
    return b, c



# 3) Divide-and-conquer clustering (binary splits using correlation distance)
def dnc_cluster(X, min_cluster=20, max_depth=10):
    Xz = zscore_rows(X)
    clusters, stack = [], [(list(range(len(Xz))), 0)]
    while stack:
        idxs, depth = stack.pop()
        if len(idxs) <= min_cluster or depth >= max_depth:
            clusters.append(idxs)
            continue

        Xi = Xz[idxs]
        iA, iB = farthest_pair(Xi)
        A, B = Xi[iA], Xi[iB]

        left, right = [], []
        for g in idxs:
            if corr_distance(Xz[g], A) <= corr_distance(Xz[g], B):
                left.append(g)
            else:
                right.append(g)

        if not left or not right:
            mid = len(idxs) // 2
            left, right = idxs[:mid], idxs[mid:]

        stack.append((right, depth + 1))
        stack.append((left,  depth + 1))
    return clusters


# 4) Closest pair inside a cluster (brute force)
def closest_pair_in_cluster(X, idxs):
    if len(idxs) < 2:
        return (None, None, float("inf"))
    best = (None, None, float("inf"))
    for ii in range(len(idxs) - 1):
        i = idxs[ii]; xi = X[i]
        for jj in range(ii + 1, len(idxs)):
            j = idxs[jj]
            d = corr_distance(xi, X[j])
            if d < best[2]:
                best = (i, j, d)
    return best


# 5) Kadane on first difference (most “active” interval)
def kadane_interval_on_diff(x):
    if len(x) < 2:
        return 0, 0, 0.0
    dx = np.diff(x)
    best = cur = dx[0]
    s = start = end = 0
    for k in range(1, len(dx)):
        if dx[k] > cur + dx[k]:
            cur = dx[k]
            s = k
        else:
            cur += dx[k]
        if cur > best:
            best = cur
            start = s
            end = k
    return start, end + 1, float(best)


# 6) Saving utilities
def ensure_dir(p):
    if p:
        os.makedirs(p, exist_ok=True)

def save_clusters_csv(clusters, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "member_index"])
        for cid, idxs in enumerate(clusters):
            for i in idxs:
                w.writerow([cid, i])

def save_pairs_csv(pairs, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "i", "j", "distance"])
        for cid, (i, j, d) in enumerate(pairs):
            w.writerow([cid, i, j, d])

def save_kadane_csv(X, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series_id", "start", "end", "score"])
        for i in range(len(X)):
            s, e, sc = kadane_interval_on_diff(X[i])
            w.writerow([i, s, e, sc])

def plot_series(x, title, path, window=None):
    ensure_dir(os.path.dirname(path))
    plt.figure(figsize=(8, 3))
    plt.plot(x, label="signal")
    if window:
        s, e = window
        s = max(0, int(s))
        e = min(len(x) - 1, int(e))
        if s <= e:
            plt.axvspan(s, e, alpha=0.2, label="Kadane")
    plt.title(title)
    plt.xlabel("samples")
    plt.ylabel("amplitude")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


# 7) Run
if __name__ == "__main__":
    t0 = time.perf_counter()
    X = load_signals_from_mat(MAT_PATH, CHANNEL, SAMPLES)

    # cap to 1,000 segments
    if len(X) > LIMIT_SEGMENTS:
        X = X[:LIMIT_SEGMENTS]
    t1 = time.perf_counter()

    clusters = dnc_cluster(X, MIN_CLUSTER, MAX_DEPTH)
    t2 = time.perf_counter()

    pairs = [closest_pair_in_cluster(X, idxs) for idxs in clusters]
    t3 = time.perf_counter()

    ensure_dir(OUT_DIR)
    save_clusters_csv(clusters, os.path.join(OUT_DIR, "clusters.csv"))
    save_pairs_csv(pairs,       os.path.join(OUT_DIR, "closest_pairs.csv"))
    save_kadane_csv(X,          os.path.join(OUT_DIR, "kadane_windows.csv"))

    # few example plots: representative + closest pairs for first 5 clusters
    for cid, idxs in enumerate(clusters[:5]):
        if not idxs:
            continue
        rep, rep_sum = None, float("inf")
        for i in idxs:
            s = sum(corr_distance(X[i], X[j]) for j in idxs if j != i)
            if s < rep_sum:
                rep_sum, rep = s, i
        if rep is not None:
            rs, re, _ = kadane_interval_on_diff(X[rep])
            plot_series(
                X[rep],
                f"Cluster {cid} representative (id={rep})",
                os.path.join(OUT_DIR, f"cluster{cid}_rep_id{rep}.png"),
                window=(rs, re),
            )

    for cid, (i, j, d) in enumerate(pairs[:5]):
        if i is None:
            continue
        si, ei, _ = kadane_interval_on_diff(X[i])
        sj, ej, _ = kadane_interval_on_diff(X[j])
        plot_series(
            X[i],
            f"Cluster {cid} closest A (i={i}) | d={d:.4f}",
            os.path.join(OUT_DIR, f"cluster{cid}_closest_i{i}.png"),
            window=(si, ei),
        )
        plot_series(
            X[j],
            f"Cluster {cid} closest B (j={j}) | d={d:.4f}",
            os.path.join(OUT_DIR, f"cluster{cid}_closest_j{j}.png"),
            window=(sj, ej),
        )

    with open(os.path.join(OUT_DIR, "performance.txt"), "w") as f:
        f.write(
            f"load: {t1 - t0:.3f}s\n"
            f"cluster: {t2 - t1:.3f}s\n"
            f"pairs: {t3 - t2:.3f}s\n"
            f"clusters={len(clusters)} segments={len(X)} (capped at {LIMIT_SEGMENTS})\n"
        )

    print(f"Done. Outputs in: {OUT_DIR}")
