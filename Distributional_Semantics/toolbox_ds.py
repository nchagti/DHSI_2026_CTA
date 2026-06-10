"""
Distributional Semantics toolbox.

A set of helpers for exploring trained Word2Vec models. Pedagogical:
each function is small, focused, and produces either a PrettyTable
(for textual exploration) or a matplotlib figure (for visual
exploration).

All functions take a `model` of type `gensim.models.KeyedVectors`
(i.e. `Word2Vec.wv`). Pass `model.wv` if you have a full Word2Vec
object.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
from prettytable import PrettyTable
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Type alias for clarity
# ---------------------------------------------------------------------------
# gensim returns KeyedVectors when you do `model.wv` on a Word2Vec object.
# We accept either — `_kv(m)` normalizes it.

def _kv(model):
    """Return a KeyedVectors object whether the user passed Word2Vec or wv."""
    if hasattr(model, "wv"):
        return model.wv
    return model


# ---------------------------------------------------------------------------
# 1. Load model + print info
# ---------------------------------------------------------------------------

def load_model(path: str | Path):
    """
    Load a Word2Vec model from `path` and print a one-shot summary table
    of its hyperparameters and vocabulary size.

    The path may use `~` for the user home directory and environment
    variables like `$HOME` or `%USERPROFILE%`; these are expanded
    automatically and work cross-platform.

    Returns
    -------
    model : gensim.models.Word2Vec
        The full model object (call `model.wv` to get just the vectors).
    info : dict
        Same numbers as the printed table, in case you want them later
        (e.g. `info["dim"]` for the embedding dimension).
    """
    import os
    from gensim.models import Word2Vec

    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = Word2Vec.load(str(path))
    kv = model.wv

    # Pull the hyperparameters that survive into the loaded object.
    # Some are on `model`, others on `kv`.
    info = {
        "path":          str(path),
        "vocab_size":    len(kv),
        "dim":           kv.vector_size,
        "window":        getattr(model, "window", None),
        "min_count":     getattr(model, "min_count", None),
        "sg":            getattr(model, "sg", None),
        "negative":      getattr(model, "negative", None),
        "hs":            getattr(model, "hs", None),
        "epochs":        getattr(model, "epochs", None),
        "workers":       getattr(model, "workers", None),
        "total_train_time": getattr(model, "total_train_time", None),
    }

    # Pretty-print summary.
    tbl = PrettyTable()
    tbl.field_names = ["Property", "Value"]
    tbl.align["Property"] = "l"
    tbl.align["Value"] = "l"

    def _fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    tbl.add_row(["Path", info["path"]])
    tbl.add_row(["Vocabulary size", f"{info['vocab_size']:,}"])
    tbl.add_row(["Vector dimension", info["dim"]])
    tbl.add_row(["Window", _fmt(info["window"])])
    tbl.add_row(["Min count", _fmt(info["min_count"])])
    tbl.add_row(["Architecture",
                 "skip-gram" if info["sg"] == 1
                 else ("CBOW" if info["sg"] == 0 else "—")])
    #tbl.add_row(["Negative sampling", _fmt(info["negative"])])
    #tbl.add_row(["Hierarchical softmax", _fmt(info["hs"])])
    #tbl.add_row(["Epochs", _fmt(info["epochs"])])
    #tbl.add_row(["Workers", _fmt(info["workers"])])
    if info["total_train_time"] is not None:
        tbl.add_row(["Total training time",
                     f"{info['total_train_time']:.1f} s"])

    print(tbl)
    return model, info


# ---------------------------------------------------------------------------
# Internal helpers: vocab lookup with friendly errors
# ---------------------------------------------------------------------------

def _check_in_vocab(kv, words: Iterable[str]):
    """Return (present_words, missing_words). Doesn't raise."""
    present, missing = [], []
    for w in words:
        if w in kv:
            present.append(w)
        else:
            missing.append(w)
    return present, missing


def _warn_missing(missing: Sequence[str]):
    if missing:
        joined = ", ".join(missing)
        print(f"  ! skipping (not in vocab): {joined}")


# ---------------------------------------------------------------------------
# 1b. Word counts — inspect corpus frequency of a list of words
# ---------------------------------------------------------------------------

def _rgb_to_ansi256(r, g, b):
    """sRGB (0..1) → ANSI 256-color code in the 6×6×6 cube."""
    def chan(v):
        return min(5, max(0, round(v * 5)))
    return 16 + 36 * chan(r) + 6 * chan(g) + chan(b)


def _ansi_wrap(text, rgb):
    """Wrap `text` in an ANSI 256-color escape sequence."""
    r, g, b = rgb[:3]
    code = _rgb_to_ansi256(r, g, b)
    return f"\033[38;5;{code}m{text}\033[0m"


def word_counts_table(model, words: Sequence[str],
                       colors=None):
    """
    Show the corpus frequency (training count) of each input word
    in a PrettyTable, plus a TOTAL row at the bottom. Each row is
    coloured for visual distinctness.

    Words not in the model's vocabulary are shown with count `—`
    (an em-dash) and are NOT counted toward the total.

    Parameters
    ----------
    words  : list of words to look up.
    colors : per-word colours. None | list | dict {word: colour}.
        - None → tab10 palette in order.
        - list → same length as `words`, positional.
        - dict → {"king": "blue"}; missing keys fall back to tab10.

    Returns the PrettyTable so you can `print(tbl)` again or export it.
    Colour codes are stripped from the returned object's string only if
    rendered in a non-ANSI environment (most terminals and Jupyter
    handle them natively).
    """
    from matplotlib.colors import to_rgb
    kv = _kv(model)

    # Resolve colours using the same helper used by pca_2d etc.
    rgba_per_word = _resolve_colors(words, colors)
    rgb_per_word = [c[:3] for c in rgba_per_word]

    tbl = PrettyTable()
    tbl.field_names = ["word", "count"]
    tbl.align["word"] = "l"
    tbl.align["count"] = "r"

    total = 0
    missing = []
    for w, rgb in zip(words, rgb_per_word):
        if w in kv:
            try:
                c = kv.get_vecattr(w, "count")
                c = int(c) if c is not None else 0
            except (KeyError, AttributeError):
                c = 0
            total += c
            tbl.add_row([_ansi_wrap(w, rgb),
                         _ansi_wrap(f"{c:,}", rgb)])
        else:
            missing.append(w)
            tbl.add_row([_ansi_wrap(w, rgb),
                         _ansi_wrap("—", rgb)])

    # Separator + total. PrettyTable's divider= adds a horizontal line.
    tbl.add_row(["─" * 4, "─" * 8], divider=True)
    tbl.add_row(["TOTAL", f"{total:,}"])

    print(tbl)
    if missing:
        print(f"  ({len(missing)} word(s) not in vocabulary: "
              f"{', '.join(missing)})")
    return tbl


# ---------------------------------------------------------------------------
# 2. Most similar — text table
# ---------------------------------------------------------------------------

def most_similar_table(model, words: Sequence[str], top_n: int = 10):
    """
    For each word in `words`, print the `top_n` nearest neighbours and
    their cosine similarities, side by side in a PrettyTable.

    Returns the PrettyTable object so you can `print(tbl)` again or
    export it.
    """
    kv = _kv(model)
    present, missing = _check_in_vocab(kv, words)
    _warn_missing(missing)

    if not present:
        print("No input words in vocabulary.")
        return None

    # Header: alternating "word_i", "sim_i" columns.
    field_names = ["#"]
    for w in present:
        field_names += [w, f"sim({w})"]

    tbl = PrettyTable()
    tbl.field_names = field_names
    for col in field_names:
        tbl.align[col] = "l"

    # Compute neighbours once per input.
    neighbours = {w: kv.most_similar(w, topn=top_n) for w in present}

    for i in range(top_n):
        row = [i + 1]
        for w in present:
            n_word, n_sim = neighbours[w][i]
            row += [n_word, f"{n_sim:.3f}"]
        tbl.add_row(row)

    print(tbl)
    return tbl


# ---------------------------------------------------------------------------
# 3. Most similar — plot (scatter + connecting lines)
# ---------------------------------------------------------------------------

def most_similar_plot(model, words: Sequence[str], top_n: int = 10,
                      figsize=(11, 6), ax=None):
    """
    Visualise `most_similar` results as a scatter plot with connecting
    lines. Each input word gets its own colour; the x-axis is the rank
    of the neighbour (1 = closest), the y-axis is cosine similarity.
    Neighbour words are annotated on the points.

    Reading the plot:
      - high, flat line  ->  word has many tightly-clustered neighbours
      - steeply dropping ->  word has one or two close neighbours, then
                              similarity falls off quickly
      - low overall      ->  word is in a sparse region of the vocab
    """
    kv = _kv(model)
    present, missing = _check_in_vocab(kv, words)
    _warn_missing(missing)

    if not present:
        print("No input words in vocabulary.")
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    cmap = plt.get_cmap("tab10")
    ranks = np.arange(1, top_n + 1)

    for i, w in enumerate(present):
        color = cmap(i % 10)
        neighbours = kv.most_similar(w, topn=top_n)
        sims = [s for _, s in neighbours]
        labels = [n for n, _ in neighbours]

        ax.plot(ranks, sims, "-", color=color, alpha=0.5, linewidth=1.5)
        ax.scatter(ranks, sims, color=color, s=50, zorder=3, label=w)

        # Annotate each point with the neighbour word.
        for r, s, lab in zip(ranks, sims, labels):
            ax.annotate(lab, (r, s), xytext=(0, 6),
                        textcoords="offset points", fontsize=8,
                        ha="center", color=color)

    ax.set_xlabel("Neighbour rank (1 = closest)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Top {top_n} nearest neighbours")
    ax.set_xticks(ranks)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Input word", loc="best")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Project words on two semantic axes (bias / stereotype plot)
# ---------------------------------------------------------------------------

def project_on_axes(model, words: Sequence[str],
                    x_axis: tuple[str, str],
                    y_axis: tuple[str, str],
                    figsize=(8, 8), ax=None):
    """
    Project each word onto two semantic axes defined by word pairs.
    Classical "axis-as-difference" method used in the bias-in-embeddings
    literature (Bolukbasi et al. 2016):

        axis_vector = (vec(pos_pole) - vec(neg_pole)) / ||...||      # unit
        coord(w)    = (vec(w) / ||vec(w)||) · axis_vector             # cosine

    A word that "points in the same direction" as the axis gets a
    positive coordinate; one that points the opposite way gets a
    negative one.

    Use case: reveal cultural associations encoded in the embedding —
    project ["doctor", "nurse", "teacher", ...] onto a (man, woman)
    axis and a (poor, rich) axis to see where the model places each
    profession.

    Parameters
    ----------
    x_axis, y_axis : tuple (negative_pole, positive_pole)
        e.g. ("man", "woman"), ("poor", "rich"). The negative pole
        sits on the left/bottom; the positive on the right/top.

    Methodology note
    ----------------
    Don't include the pole words themselves in `words`. The poles are
    *units of measurement* for the axis, not objects to be measured.
    Projecting "man" onto the (man, woman) axis is asking "how many
    metres is one metre?" — the answer technically exists but isn't
    meaningful, and is often visually confusing because the embedding
    space is not calibrated to put pole words at ±1.

    Returns the matplotlib figure.
    """
    kv = _kv(model)

    required = list(words) + list(x_axis) + list(y_axis)
    _present_req, missing_req = _check_in_vocab(kv, required)
    _warn_missing(missing_req)

    if any(w not in kv for w in x_axis):
        print(f"X axis poles {x_axis} not both in vocab — aborting.")
        return None
    if any(w not in kv for w in y_axis):
        print(f"Y axis poles {y_axis} not both in vocab — aborting.")
        return None

    words_present = [w for w in words if w in kv]
    if not words_present:
        print("No input words in vocabulary.")
        return None

    # Build unit-length axis vectors.
    def _axis_vec(neg, pos):
        v = kv[pos] - kv[neg]
        n = np.linalg.norm(v)
        if n == 0:
            return v
        return v / n

    ax_x = _axis_vec(x_axis[0], x_axis[1])
    ax_y = _axis_vec(y_axis[0], y_axis[1])

    def _coord(w):
        v = kv[w]
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        return float(v @ ax_x), float(v @ ax_y)

    xs, ys = [], []
    for w in words_present:
        x, y = _coord(w)
        xs.append(x)
        ys.append(y)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    cmap = plt.get_cmap("tab10")
    for i, (w, x, y) in enumerate(zip(words_present, xs, ys)):
        color = cmap(i % 10)
        ax.scatter(x, y, color=color, s=80, zorder=3)
        ax.annotate(w, (x, y), xytext=(7, 5),
                    textcoords="offset points", fontsize=11,
                    color=color)

    ax.set_xlabel(f"{x_axis[0]}   ←   →   {x_axis[1]}")
    ax.set_ylabel(f"{y_axis[0]}   ←   →   {y_axis[1]}")
    ax.set_title("Words projected on two semantic axes")

    ax.axhline(0, color="grey", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="grey", linewidth=0.8, alpha=0.6)
    ax.grid(True, alpha=0.25)

    # Symmetric limits so the origin is centred.
    lim = max(abs(min(xs + ys)), abs(max(xs + ys))) * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    return fig


def project_on_seed_axes(model, words: Sequence[str],
                          x_axis: tuple[Sequence[str], Sequence[str]],
                          y_axis: tuple[Sequence[str], Sequence[str]],
                          figsize=(8, 8), ax=None):
    """
    Like `project_on_axes`, but each pole of each axis is defined by a
    LIST of seed words instead of a single word. The pole vector is the
    centroid (mean) of the seed word vectors. The axis is then the
    direction from one centroid to the other.

    This is the "seed words" method (Turney & Littman 2003;
    Bolukbasi et al. 2016). It is the standard remedy when a candidate
    pole word is polysemous — its vector mixes multiple senses, only
    one of which you care about. Averaging vectors over several seeds
    that share the *target* sense lets the target sense dominate the
    centroid, because the unrelated senses point in different
    directions and partially cancel.

    Parameters
    ----------
    words : input words to project (NOT including any seeds).
    x_axis : tuple (neg_seeds, pos_seeds) — each is a list of seed words.
        e.g. (["poor", "destitute", "poverty"],
              ["rich", "wealthy", "wealth"])
    y_axis : same shape as x_axis.
    Missing seeds are silently skipped; a pole is invalid only if ALL
    its seeds are out of vocabulary.

    Methodology note (same as project_on_axes)
    -------------------------------------------
    Don't include the seed words in `words`. Seeds DEFINE the axis; they
    are not objects to be measured against it.

    Returns the matplotlib figure.
    """
    kv = _kv(model)

    def _centroid(seeds, label):
        present = [s for s in seeds if s in kv]
        missing = [s for s in seeds if s not in kv]
        if missing:
            print(f"  ! {label}: not in vocab: {missing}")
        if not present:
            print(f"  ! {label}: NO seeds in vocab — aborting.")
            return None, []
        vecs = np.stack([kv[s] for s in present])
        return vecs.mean(axis=0), present

    x_neg, x_neg_present = _centroid(x_axis[0], "x_axis negative pole")
    x_pos, x_pos_present = _centroid(x_axis[1], "x_axis positive pole")
    y_neg, y_neg_present = _centroid(y_axis[0], "y_axis negative pole")
    y_pos, y_pos_present = _centroid(y_axis[1], "y_axis positive pole")

    if any(v is None for v in (x_neg, x_pos, y_neg, y_pos)):
        return None

    # Build unit-length axis vectors from centroids.
    def _unit(v):
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    ax_x = _unit(x_pos - x_neg)
    ax_y = _unit(y_pos - y_neg)

    words_present = [w for w in words if w in kv]
    missing = [w for w in words if w not in kv]
    if missing:
        print(f"  ! input words not in vocab: {missing}")
    if not words_present:
        print("No input words in vocabulary.")
        return None

    def _coord(w):
        v = kv[w]
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        return float(v @ ax_x), float(v @ ax_y)

    xs, ys = [], []
    for w in words_present:
        x, y = _coord(w)
        xs.append(x)
        ys.append(y)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    cmap = plt.get_cmap("tab10")
    for i, (w, x, y) in enumerate(zip(words_present, xs, ys)):
        color = cmap(i % 10)
        ax.scatter(x, y, color=color, s=80, zorder=3)
        ax.annotate(w, (x, y), xytext=(7, 5),
                    textcoords="offset points", fontsize=11,
                    color=color)

    # Show the seed lists in the axis labels (truncated if long).
    def _fmt(seeds, limit=3):
        if len(seeds) <= limit:
            return "+".join(seeds)
        return "+".join(seeds[:limit]) + f"+{len(seeds)-limit} more"

    ax.set_xlabel(f"{_fmt(x_neg_present)}   ←   →   "
                  f"{_fmt(x_pos_present)}")
    ax.set_ylabel(f"{_fmt(y_neg_present)}   ←   →   "
                  f"{_fmt(y_pos_present)}")
    ax.set_title("Words projected on two seed-defined semantic axes")

    ax.axhline(0, color="grey", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="grey", linewidth=0.8, alpha=0.6)
    ax.grid(True, alpha=0.25)

    lim = max(abs(min(xs + ys)), abs(max(xs + ys))) * 1.25
    if lim == 0:
        lim = 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    return fig
# ---------------------------------------------------------------------------
# Internal: collect input + neighbours + user extras into one labelled set
# ---------------------------------------------------------------------------

def _collect_neighbourhood(kv, words: Sequence[str], top_n: int,
                            extra: Sequence[str] | None = None,
                            similarity_threshold: float | None = None):
    """
    Collect input words + their neighbours + optional extras for PCA.

    Returns four things:
      - labels  : list[str]      — unique words to plot (one position each)
      - vectors : np.ndarray     — embedding vector per label
      - group_membership : dict[int, list[tuple[int, float]]]
          For each input-word index gi → list of (label_index, similarity).
          The input word itself appears first with similarity 1.0. A word
          that's a neighbour of multiple inputs has entries in multiple
          groups (each with its own similarity to that group's input).
      - extras : list[int]       — indices into `labels` for extra words

    If `similarity_threshold` is given, `top_n` is ignored and we take
    every neighbour with cosine similarity ≥ threshold (capped at 200
    per input to keep things bounded).

    Missing input words are skipped silently.
    """
    labels = []
    vectors = []
    label_index = {}
    group_membership = {}
    extras = []

    def _add(word):
        if word in label_index:
            return label_index[word]
        label_index[word] = len(labels)
        labels.append(word)
        vectors.append(kv[word])
        return label_index[word]

    fetch_n = top_n if similarity_threshold is None else max(top_n, 200)

    for gi, w in enumerate(words):
        if w not in kv:
            continue
        members = [(_add(w), 1.0)]  # input itself, similarity 1.0
        for n, sim in kv.most_similar(w, topn=fetch_n):
            if similarity_threshold is not None and sim < similarity_threshold:
                break   # most_similar is sorted descending
            if n in kv:
                members.append((_add(n), float(sim)))
        if similarity_threshold is None:
            members = members[:top_n + 1]
        group_membership[gi] = members

    if extra:
        for w in extra:
            if w in kv and w not in label_index:
                extras.append(_add(w))

    return labels, np.array(vectors), group_membership, extras


def _resolve_colors(words: Sequence[str], colors):
    """
    Build a list of colours, one per input word. `colors` can be:
      - None: use tab10
      - list/tuple same length as `words`
      - dict {word: colour}
    """
    cmap = plt.get_cmap("tab10")
    if colors is None:
        return [cmap(i % 10) for i in range(len(words))]
    if isinstance(colors, dict):
        return [to_rgba(colors.get(w, cmap(i % 10)))
                for i, w in enumerate(words)]
    if isinstance(colors, (list, tuple)):
        if len(colors) != len(words):
            raise ValueError(f"colors length ({len(colors)}) "
                             f"must equal words length ({len(words)})")
        return [to_rgba(c) for c in colors]
    raise TypeError(f"Unsupported colors type: {type(colors).__name__}")


# ---------------------------------------------------------------------------
# 5. PCA 2D
# ---------------------------------------------------------------------------

def _compute_node_sizes(kv, labels, size_by_count: bool,
                        base_size: float, input_size: float):
    """
    Return two arrays:
      sizes_neighbour : size for each label when drawn as a neighbour
      sizes_input     : size for each label when drawn as the input word

    If `size_by_count=False`, returns constant arrays (base_size /
    input_size). If True, scales by sqrt of corpus frequency
    (`kv.get_vecattr(word, "count")`), normalised so the most frequent
    word in the plotted set keeps the default size, and rarer words
    shrink proportionally to sqrt(freq).
    """
    n = len(labels)
    if not size_by_count:
        return (np.full(n, base_size, dtype=float),
                np.full(n, input_size, dtype=float))

    counts = np.zeros(n, dtype=float)
    for i, w in enumerate(labels):
        try:
            c = kv.get_vecattr(w, "count")
            counts[i] = float(c) if c is not None else 0.0
        except (KeyError, AttributeError):
            counts[i] = 0.0

    max_count = counts.max() if counts.max() > 0 else 1.0
    # sqrt compresses the range so very rare words are still visible.
    # Floor at 20% of base_size so nothing disappears.
    rel = np.sqrt(counts / max_count)
    rel = np.clip(rel, 0.2, 1.0)
    return rel * base_size, rel * input_size


def pca_2d(model, words: Sequence[str], top_n: int = 10,
           extra: Sequence[str] | None = None,
           hull: bool = False,
           colors=None,
           size_by_count: bool = False,
           similarity_threshold: float | None = None,
           connect_to_input: bool = False,
           show_similarity_in_label: bool = False,
           marker_alpha_by_similarity: bool = False,
           show_centroid: bool = False,
           link_inputs: bool = False,
           density: bool = False,
           hull_layers: int = 1,
           figsize=(10, 8), ax=None):
    """
    Plot input words and their nearest neighbours in 2D via PCA.

    Core behaviour
    --------------
    Each input word gets its own colour. Its nearest neighbours share
    the colour. User-supplied `extra` words are plotted in grey. A
    word that is a neighbour of multiple inputs is drawn once but
    BELONGS to all relevant groups.

    Selection
    ---------
    top_n : neighbours per input word (default 10).
    similarity_threshold : alternative to top_n — take every neighbour
        with cosine similarity ≥ threshold. Overrides top_n when set.
    extra : optional extra words to add (no neighbours).

    Styling
    -------
    colors : per-input-word colours. None | list | dict {word: colour}.
    size_by_count : node size ∝ sqrt(corpus frequency). Default False.
    marker_alpha_by_similarity : neighbours' marker alpha ∝ similarity
        to their input word. High-sim = opaque, low-sim = faded.
    show_similarity_in_label : append "(0.87)" to neighbour labels.

    Cluster overlays
    ----------------
    hull : draw a convex hull around each cluster.
    hull_layers : if > 1, draws nested hulls (top-k closest, top-2k,
        ..., full cluster) with decreasing intensity. Reveals the core
        of the cluster vs its periphery.
    density : draw a smoothed density (KDE) inside each cluster
        instead of a flat hull fill. Heavy for large clusters.
    show_centroid : draw the centroid of each cluster as an X marker.

    Cross-cluster info
    ------------------
    connect_to_input : draw thin lines from each neighbour to its input
        word, with thickness scaled by similarity. Makes the cluster
        look like a star with rays.
    link_inputs : draw lines between input words, thickness scaled by
        the cosine similarity between them.
    """
    kv = _kv(model)
    words = [w for w in words if w in kv]
    if not words:
        print("No input words in vocabulary.")
        return None

    labels, vectors, group_membership, extras = _collect_neighbourhood(
        kv, words, top_n=top_n, extra=extra,
        similarity_threshold=similarity_threshold)

    coords = PCA(n_components=2, random_state=0).fit_transform(vectors)

    color_per_input = _resolve_colors(words, colors)

    sizes_neighbour, sizes_input = _compute_node_sizes(
        kv, labels, size_by_count, base_size=55.0, input_size=120.0)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ------------------------------------------------------------------
    # 1) Background layers (drawn first, lowest zorder)
    # ------------------------------------------------------------------

    # Density (KDE) — drawn before hulls, so hulls sit on top.
    if density:
        from scipy.stats import gaussian_kde
        from matplotlib.colors import to_rgba, LinearSegmentedColormap

        # Build a regular grid covering all points with some padding.
        all_pts = coords
        pad = 0.10 * (all_pts.max() - all_pts.min())
        xmin, xmax = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
        ymin, ymax = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad
        grid_x, grid_y = np.mgrid[xmin:xmax:120j, ymin:ymax:120j]
        grid_xy = np.vstack([grid_x.ravel(), grid_y.ravel()])

        for gi in range(len(words)):
            members = group_membership.get(gi, [])
            if len(members) < 3:
                continue
            pts = coords[[idx for idx, _ in members]]
            try:
                kde = gaussian_kde(pts.T)
                vals = kde(grid_xy).reshape(grid_x.shape)
            except Exception as e:
                print(f"  ! density failed for {words[gi]!r}: {e}")
                continue
            base = to_rgba(color_per_input[gi])
            cmap = LinearSegmentedColormap.from_list(
                f"d_{gi}",
                [(*base[:3], 0.0), (*base[:3], 0.45)])
            ax.contourf(grid_x, grid_y, vals, levels=6, cmap=cmap, zorder=0)

    # Hulls (possibly layered) — drawn behind everything else.
    if hull:
        try:
            from scipy.spatial import ConvexHull
        except ImportError:
            print("  ! scipy required for hull")
            ConvexHull = None
        if ConvexHull is not None:
            for gi in range(len(words)):
                members = group_membership.get(gi, [])
                if len(members) < 3:
                    continue
                col = color_per_input[gi]
                # Sort members by descending similarity for layered hulls.
                # (input is at sim=1.0, so it stays in every layer.)
                sorted_members = sorted(members, key=lambda x: -x[1])
                n_total = len(sorted_members)
                n_layers = max(1, int(hull_layers))

                # Build layer boundaries: cumulative counts of members
                # included in each layer (innermost first).
                # E.g. with n_total=10 and n_layers=3 → layers of size
                # 4, 7, 10.
                layer_sizes = [
                    max(3, round(n_total * (k + 1) / n_layers))
                    for k in range(n_layers)
                ]

                for layer_i, layer_size in enumerate(layer_sizes):
                    layer_members = sorted_members[:layer_size]
                    pts = coords[[idx for idx, _ in layer_members]]
                    if len(pts) < 3:
                        continue
                    # Intensity: innermost layer darkest.
                    intensity = 1.0 - (layer_i / max(1, n_layers))
                    fill_alpha = 0.05 + 0.20 * intensity
                    edge_alpha = 0.25 + 0.35 * intensity
                    try:
                        ch = ConvexHull(pts)
                        hp = pts[ch.vertices]
                        ax.fill(hp[:, 0], hp[:, 1],
                                color=col, alpha=fill_alpha, zorder=1)
                        ax.plot(np.append(hp[:, 0], hp[0, 0]),
                                np.append(hp[:, 1], hp[0, 1]),
                                color=col, alpha=edge_alpha,
                                linewidth=1, zorder=2)
                    except Exception as e:
                        print(f"  ! hull layer {layer_i} failed for "
                              f"{words[gi]!r}: {e}")
                        break  # outer layers won't work either

    # Centroid X-markers per cluster.
    if show_centroid:
        for gi in range(len(words)):
            members = group_membership.get(gi, [])
            if not members:
                continue
            pts = coords[[idx for idx, _ in members]]
            cx, cy = pts.mean(axis=0)
            ax.scatter([cx], [cy], color=color_per_input[gi],
                       marker="X", s=180, edgecolor="black",
                       linewidth=1.0, zorder=5)

    # ------------------------------------------------------------------
    # 2) Connection lines (rays from input to neighbours, links between inputs)
    # ------------------------------------------------------------------

    if connect_to_input:
        for gi in range(len(words)):
            members = group_membership.get(gi, [])
            if len(members) < 2:
                continue
            input_idx = members[0][0]
            ix, iy = coords[input_idx]
            col = color_per_input[gi]
            for nbr_idx, sim in members[1:]:
                if nbr_idx == input_idx:
                    continue
                nx, ny = coords[nbr_idx]
                # Line thickness scales with similarity (0..1).
                lw = 0.3 + 2.0 * max(0.0, min(1.0, sim))
                alpha = 0.25 + 0.45 * max(0.0, min(1.0, sim))
                ax.plot([ix, nx], [iy, ny], color=col,
                        linewidth=lw, alpha=alpha, zorder=2)

    if link_inputs and len(words) >= 2:
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if words[i] not in kv or words[j] not in kv:
                    continue
                sim = float(kv.similarity(words[i], words[j]))
                idx_i = group_membership[i][0][0]
                idx_j = group_membership[j][0][0]
                x1, y1 = coords[idx_i]
                x2, y2 = coords[idx_j]
                # Thicker lines for stronger similarity.
                lw = 0.5 + 4.0 * max(0.0, sim)
                ax.plot([x1, x2], [y1, y2], color="black",
                        linewidth=lw, alpha=0.4, linestyle="--",
                        zorder=2)
                # Label the mid-point with the similarity value.
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                ax.annotate(f"{sim:.2f}", (mx, my), fontsize=8,
                            color="black", alpha=0.7,
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="white", ec="none", alpha=0.7))

    # ------------------------------------------------------------------
    # 3) Markers (per group, with optional alpha-by-similarity)
    # ------------------------------------------------------------------

    for gi, w in enumerate(words):
        members = group_membership.get(gi, [])
        if not members:
            continue
        col = color_per_input[gi]
        input_idx = members[0][0]
        ax.scatter(coords[input_idx, 0], coords[input_idx, 1],
                   color=col, s=sizes_input[input_idx], edgecolor="black",
                   linewidth=1.2, zorder=4)
        # Neighbours.
        for nbr_idx, sim in members[1:]:
            if marker_alpha_by_similarity:
                # Map similarity to alpha. Cap range so very low sims
                # still show but aren't dominant.
                alpha = 0.20 + 0.70 * max(0.0, min(1.0, sim))
            else:
                alpha = 0.55
            ax.scatter([coords[nbr_idx, 0]], [coords[nbr_idx, 1]],
                       color=col, alpha=alpha,
                       s=sizes_neighbour[nbr_idx], zorder=3)

    # ------------------------------------------------------------------
    # 4) Annotations (one per label, coloured by first group)
    # ------------------------------------------------------------------

    # Build best-similarity-per-label map for similarity-in-label.
    best_sim = {}  # label_idx → (gi, sim)
    word_to_group = {}
    input_indices = set()
    for gi, members in group_membership.items():
        if not members:
            continue
        input_indices.add(members[0][0])
        for idx, sim in members:
            word_to_group.setdefault(idx, gi)
            if idx not in best_sim or sim > best_sim[idx][1]:
                best_sim[idx] = (gi, sim)

    for idx, label in enumerate(labels):
        if idx in word_to_group:
            col = color_per_input[word_to_group[idx]]
        else:
            col = "grey"
        is_input = idx in input_indices

        text = label
        if show_similarity_in_label and not is_input and idx in best_sim:
            sim = best_sim[idx][1]
            text = f"{label} ({sim:.2f})"

        ax.annotate(text, (coords[idx, 0], coords[idx, 1]),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=9, color=col,
                    fontweight=("bold" if is_input else "normal"))

    # ------------------------------------------------------------------
    # 5) Extras (grey squares)
    # ------------------------------------------------------------------

    if extras:
        pts = coords[extras]
        ax.scatter(pts[:, 0], pts[:, 1], color="grey",
                   s=sizes_neighbour[extras],
                   marker="s", zorder=3)

    # ------------------------------------------------------------------
    # 6) Axes, title, legend
    # ------------------------------------------------------------------

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    if similarity_threshold is not None:
        title_suffix = f"neighbours with sim ≥ {similarity_threshold}"
    else:
        title_suffix = f"top {top_n} neighbours each"
    ax.set_title(f"PCA(2D): {len(words)} input word(s) + {title_suffix}")

    ax.grid(True, alpha=0.25)
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", color="w",
                   markerfacecolor=color_per_input[i],
                   markeredgecolor="black", markersize=10, label=w)
        for i, w in enumerate(words)
    ], title="Input word", loc="best")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. PCA 3D (animated rotation)
# ---------------------------------------------------------------------------

def pca_3d(model, words: Sequence[str], top_n: int = 10,
           extra: Sequence[str] | None = None,
           colors=None,
           size_by_count: bool = False,
           similarity_threshold: float | None = None,
           connect_to_input: bool = False,
           show_similarity_in_label: bool = False,
           marker_alpha_by_similarity: bool = False,
           show_centroid: bool = False,
           link_inputs: bool = False,
           hull: bool = False,
           hull_layers: int = 1,
           frames: int = 60,
           figsize=(9, 9)):
    """
    Same idea as `pca_2d`, but in 3D, rendered as an animated rotation.

    Most parameters mirror `pca_2d` (see there for details). 3D-specific:

      frames : number of animation frames (one full 360° rotation total).

    Supported in 3D:
      hull : draws each cluster's 3D convex hull (triangulated faces).
        Hulls in 3D are translucent so points inside remain visible
        as the camera rotates.
      hull_layers : nested 3D hulls with decreasing intensity, just
        like in `pca_2d`.

    Not supported in 3D:
      density (KDE in 3D is too slow and visually opaque).

    Returns a `matplotlib.animation.FuncAnimation`. In a Jupyter notebook
    display it with `HTML(anim.to_jshtml())`. To save as a GIF:
    `anim.save("foo.gif", writer="pillow", fps=20)`.
    """
    kv = _kv(model)
    words = [w for w in words if w in kv]
    if not words:
        print("No input words in vocabulary.")
        return None

    labels, vectors, group_membership, extras = _collect_neighbourhood(
        kv, words, top_n=top_n, extra=extra,
        similarity_threshold=similarity_threshold)

    coords = PCA(n_components=3, random_state=0).fit_transform(vectors)

    color_per_input = _resolve_colors(words, colors)

    sizes_neighbour, sizes_input = _compute_node_sizes(
        kv, labels, size_by_count, base_size=55.0, input_size=120.0)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Hulls (possibly layered) — drawn first, behind everything.
    # In 3D each hull face is a triangle; we render them via
    # Poly3DCollection.
    if hull:
        try:
            from scipy.spatial import ConvexHull
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            print("  ! scipy + mpl_toolkits required for 3D hull")
            ConvexHull = None
        if ConvexHull is not None:
            for gi in range(len(words)):
                members = group_membership.get(gi, [])
                # ConvexHull in 3D needs at least 4 non-coplanar points.
                if len(members) < 4:
                    continue
                col = color_per_input[gi]
                # Sort members by descending similarity, like in 2D.
                sorted_members = sorted(members, key=lambda x: -x[1])
                n_total = len(sorted_members)
                n_layers = max(1, int(hull_layers))
                layer_sizes = [
                    max(4, round(n_total * (k + 1) / n_layers))
                    for k in range(n_layers)
                ]

                for layer_i, layer_size in enumerate(layer_sizes):
                    layer_members = sorted_members[:layer_size]
                    pts = coords[[idx for idx, _ in layer_members]]
                    if len(pts) < 4:
                        continue
                    intensity = 1.0 - (layer_i / max(1, n_layers))
                    # 3D needs lower alpha overall because faces stack.
                    face_alpha = 0.04 + 0.10 * intensity
                    edge_alpha = 0.15 + 0.25 * intensity
                    try:
                        ch = ConvexHull(pts)
                        # Each simplex is the indices of 3 points forming
                        # a triangular face.
                        triangles = [pts[simplex] for simplex in ch.simplices]
                        poly = Poly3DCollection(
                            triangles,
                            facecolor=col,
                            edgecolor=col,
                            alpha=face_alpha,
                            linewidths=0.4)
                        # alpha on Poly3DCollection sets BOTH face and
                        # edge; we want subtler edges, so override edge
                        # alpha via RGBA.
                        from matplotlib.colors import to_rgba
                        poly.set_edgecolor(to_rgba(col, edge_alpha))
                        ax.add_collection3d(poly)
                    except Exception as e:
                        print(f"  ! 3D hull layer {layer_i} failed for "
                              f"{words[gi]!r}: {e}")
                        break

    # Connection lines (drawn after hulls, before markers).
    if connect_to_input:
        for gi in range(len(words)):
            members = group_membership.get(gi, [])
            if len(members) < 2:
                continue
            input_idx = members[0][0]
            ix, iy, iz = coords[input_idx]
            col = color_per_input[gi]
            for nbr_idx, sim in members[1:]:
                if nbr_idx == input_idx:
                    continue
                nx, ny, nz = coords[nbr_idx]
                lw = 0.3 + 2.0 * max(0.0, min(1.0, sim))
                alpha = 0.25 + 0.45 * max(0.0, min(1.0, sim))
                ax.plot([ix, nx], [iy, ny], [iz, nz], color=col,
                        linewidth=lw, alpha=alpha)

    if link_inputs and len(words) >= 2:
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if words[i] not in kv or words[j] not in kv:
                    continue
                sim = float(kv.similarity(words[i], words[j]))
                idx_i = group_membership[i][0][0]
                idx_j = group_membership[j][0][0]
                x1, y1, z1 = coords[idx_i]
                x2, y2, z2 = coords[idx_j]
                lw = 0.5 + 4.0 * max(0.0, sim)
                ax.plot([x1, x2], [y1, y2], [z1, z2], color="black",
                        linewidth=lw, alpha=0.4, linestyle="--")

    # Plot points group by group.
    for gi, w in enumerate(words):
        members = group_membership.get(gi, [])
        if not members:
            continue
        col = color_per_input[gi]
        # Input word
        input_idx = members[0][0]
        ax.scatter(coords[input_idx, 0], coords[input_idx, 1],
                   coords[input_idx, 2], color=col, s=sizes_input[input_idx],
                   edgecolor="black", linewidth=1.2)
        # Neighbours
        for nbr_idx, sim in members[1:]:
            if marker_alpha_by_similarity:
                alpha = 0.20 + 0.70 * max(0.0, min(1.0, sim))
            else:
                alpha = 0.55
            ax.scatter([coords[nbr_idx, 0]], [coords[nbr_idx, 1]],
                       [coords[nbr_idx, 2]], color=col, alpha=alpha,
                       s=sizes_neighbour[nbr_idx])

    # Centroids.
    if show_centroid:
        for gi in range(len(words)):
            members = group_membership.get(gi, [])
            if not members:
                continue
            pts = coords[[idx for idx, _ in members]]
            cx, cy, cz = pts.mean(axis=0)
            ax.scatter([cx], [cy], [cz], color=color_per_input[gi],
                       marker="X", s=180, edgecolor="black", linewidth=1.0)

    # Annotate every word once, coloured by its first group.
    word_to_group = {}
    input_indices = set()
    best_sim = {}
    for gi, members in group_membership.items():
        if not members:
            continue
        input_indices.add(members[0][0])
        for idx, sim in members:
            word_to_group.setdefault(idx, gi)
            if idx not in best_sim or sim > best_sim[idx][1]:
                best_sim[idx] = (gi, sim)

    for idx, label in enumerate(labels):
        if idx in word_to_group:
            col = color_per_input[word_to_group[idx]]
        else:
            col = "grey"
        is_input = idx in input_indices
        text = label
        if show_similarity_in_label and not is_input and idx in best_sim:
            text = f"{label} ({best_sim[idx][1]:.2f})"
        ax.text(coords[idx, 0], coords[idx, 1], coords[idx, 2],
                text, fontsize=8, color=col)

    if extras:
        pts = coords[extras]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="grey",
                   marker="s", s=sizes_neighbour[extras])

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"PCA(3D): {len(words)} input(s) + top {top_n}")

    def _update(frame):
        # Full 360° rotation over the entire animation.
        ax.view_init(elev=20, azim=frame * 360 / frames)
        return []

    anim = FuncAnimation(fig, _update, frames=frames,
                         interval=50, blit=False)
    plt.close(fig)  # avoid showing the static frame in notebook
    return anim


# ---------------------------------------------------------------------------
# 7. Analogies (bonus)
# ---------------------------------------------------------------------------

def analogy(model, a: str, b: str, c: str, top_n: int = 5):
    """
    Solve `a` is to `b` as `c` is to ?, the way `king - man + woman = queen`.

    Returns a PrettyTable with the top_n candidates.

    Reading: this is the most famous Word2Vec demo. It works when the
    relevant relation (gender, country-capital, comparative-superlative)
    is encoded as a roughly-consistent direction in the embedding space.
    On small or noisy corpora it often fails — don't take a wrong answer
    as a sign of broken code; it can be a sign of insufficient data.
    """
    kv = _kv(model)
    present, missing = _check_in_vocab(kv, [a, b, c])
    _warn_missing(missing)
    if missing:
        return None

    # gensim's positive/negative interface:
    #   king - man + woman = positive=['king','woman'], negative=['man']
    # mapping for "a is to b as c is to ?":
    #   ? = b - a + c  → positive=[b, c], negative=[a]
    results = kv.most_similar(positive=[b, c], negative=[a], topn=top_n)

    tbl = PrettyTable()
    tbl.field_names = [f"{a} : {b} ≈ {c} : ?", "similarity"]
    tbl.align[tbl.field_names[0]] = "l"
    tbl.align["similarity"] = "l"
    for w, sim in results:
        tbl.add_row([w, f"{sim:.3f}"])

    print(tbl)
    return tbl


# ---------------------------------------------------------------------------
# 8. Odd one out (bonus)
# ---------------------------------------------------------------------------

def odd_one_out(model, words: Sequence[str]):
    """
    Return the word from the list whose vector is furthest from the mean
    of the others. Wrapper around `kv.doesnt_match`.

    Pedagogically useful as a one-liner that demonstrates the model has
    learned semantic categories: in a list of three breakfast foods plus
    "dinner", the model should pick out "dinner". On a small corpus it
    may fail in instructive ways.
    """
    kv = _kv(model)
    present, missing = _check_in_vocab(kv, words)
    _warn_missing(missing)
    if len(present) < 2:
        print("Need at least two in-vocabulary words.")
        return None

    odd = kv.doesnt_match(present)
    print(f"  input: {present}")
    print(f"  odd one out → {odd!r}")
    return odd


# ---------------------------------------------------------------------------
# 9. Cosine similarity matrix (bonus)
# ---------------------------------------------------------------------------

def similarity_matrix(model, words: Sequence[str],
                      figsize=(8, 7), ax=None):
    """
    Heatmap of pairwise cosine similarity between a set of words.

    Pedagogically the most direct view of the embedding space: a
    block-diagonal pattern (high values clustered) means the model
    found semantic groups; a flat/uniform matrix means it didn't.

    Use before reaching for PCA — the matrix shows the raw structure
    that PCA tries to compress into 2 axes.
    """
    kv = _kv(model)
    present, missing = _check_in_vocab(kv, words)
    _warn_missing(missing)
    if len(present) < 2:
        print("Need at least two in-vocabulary words.")
        return None

    n = len(present)
    M = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        M[i, j] = kv.similarity(present[i], present[j])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(present, rotation=45, ha="right")
    ax.set_yticklabels(present)

    # Annotate each cell with its similarity.
    for i in range(n):
        for j in range(n):
            txt_color = "white" if abs(M[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color=txt_color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="cosine similarity")
    ax.set_title("Pairwise cosine similarity")
    fig.tight_layout()
    return fig
