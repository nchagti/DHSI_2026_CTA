"""
toolbox_gs.py
=============

Stylometry and corpus-statistics toolbox for plain-text folders.

Public API
----------
corpus_stats(input_folder, sample_size=5000)
    Print a per-file table of lexical / structural statistics.

author_stats(input_folder, sample_size=5000)
    Print a per-author table. Authors are inferred from filenames of
    the form "<Author>_<Title>.txt" (the part before the first
    underscore is the author).

plot_authors(input_folder, x_metric, y_metric, size_metric=None,
             sample_size=5000, **kwargs)
    Scatter plot of authors in 2D metric space. Returns a matplotlib
    Figure.

word_frequency(input_folder, words)
    Print a per-file frequency table (count + per-mille) for a list
    of target words.

word_frequency_plot(input_folder, words, chunk_size=1000, **kwargs)
    One time-series figure per file. Returns a list of Figures.

word_frequency_combined(input_folder, words, chunk_size=1000, **kwargs)
    One figure for the whole folder, with all files concatenated and
    file boundaries marked. Returns a single Figure.

top_words_plot(input_file, n=20, **kwargs)
    Bar chart of the n most frequent words in a single file, useful
    for showing how stop words dominate raw word counts. Returns a
    matplotlib Figure.

compare_authors_heatmap(input_folder, n=30, exclude_stop_words=False, **kwargs)
    Heatmap comparing how each author uses the top-n words in the
    corpus. Rows = authors, columns = words. Cell colour and label
    show that word's percentage share of the author's text. Returns
    a matplotlib Figure.

cumulative_coverage_plot(input_file, max_n=100, **kwargs)
    Plot showing how much of the text is covered by the top-k most
    frequent words, for k from 1 to max_n. Illustrates Zipf's law.
    Returns a matplotlib Figure.

sentence_length_histogram(input_file, **kwargs)
    Histogram of sentence lengths (in words) for a single file.

sentence_length_over_text(input_file, window=20, **kwargs)
    Rolling mean of sentence length across the text. Reveals
    structural changes (dialogue vs description, register shifts).

word_length_histogram(input_file, **kwargs)
    Histogram of word lengths (in characters) for a single file.

word_length_over_text(input_file, chunk_size=1000, **kwargs)
    Rolling mean of word length across the text.

zipf_plot(input_file, **kwargs)
    Log-log plot of word rank vs frequency. The straight-ish line is
    Zipf's law in action.

ttr_curve(input_file, step=500, **kwargs)
    TTR computed on progressively larger samples of the same text.
    The curve decays monotonically -- explains why MTLD and TTR@N
    exist.

punctuation_density(input_folder)
    Per-file table showing the share of each punctuation type
    (commas, periods, semicolons, etc.) as a percentage of all
    characters.

Notes
-----
- All metrics use a custom tokeniser that lowercases, strips
  punctuation, and preserves Devanagari + a few neighbouring Indic
  scripts (Python's \\w drops viramas and matras, which would split
  Devanagari mid-word).
- Sentences are split on . ! ? and the Devanagari danda (।), with the
  whole input treated as one continuous text -- newlines are not
  sentence boundaries.
- Aggregate rows in tables are recomputed on the concatenated corpus
  rather than averaging per-row, because MTLD / TTR / Hapax do not
  average meaningfully.
"""

import os
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
from prettytable import PrettyTable


__all__ = [
    "corpus_stats",
    "author_stats",
    "plot_authors",
    "word_frequency",
    "word_frequency_plot",
    "word_frequency_combined",
    "top_words_plot",
    "compare_authors_heatmap",
    "cumulative_coverage_plot",
    "sentence_length_histogram",
    "sentence_length_over_text",
    "word_length_histogram",
    "word_length_over_text",
    "zipf_plot",
    "ttr_curve",
    "punctuation_density",
]


# ---------------------------------------------------------------------------
# ANSI colours (used by all table renderers)
# ---------------------------------------------------------------------------

_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"

_COLOURS = {
    # row identifiers
    "file":         "\033[36m",
    "author":       "\033[36m",
    # volume
    "files":        "\033[37m",
    "chars":        "\033[33m",
    "words":        "\033[32m",
    "tokens":       "\033[92m",
    "sentences":    "\033[35m",
    "avg_book_len": "\033[32m",
    # length-independent style
    "avg_word":     "\033[34m",
    "avg_sent":     "\033[94m",
    "mtld":         "\033[91m",
    "ttr_1k":       "\033[93m",
    # fair-sample
    "ttr_n":        "\033[1;93m",
    "hapax_n":      "\033[1;95m",
    # length-dependent
    "ld":           "\033[31m",
    "hapax_pct":    "\033[95m",
    # other
    "warning":      "\033[37m",
}

_PALETTE = [
    "\033[33m", "\033[32m", "\033[34m", "\033[35m", "\033[36m",
    "\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m",
]


def _colour(value, key):
    return f"{_COLOURS.get(key, '')}{value}{_ANSI_RESET}"


def _palette_colour(value, idx):
    return f"{_PALETTE[idx % len(_PALETTE)]}{value}{_ANSI_RESET}"


# ---------------------------------------------------------------------------
# Tokenisation and sentence splitting
# ---------------------------------------------------------------------------

_TOKEN_KEEP = (
    r"\u0900-\u097F"      # Devanagari
    r"\u0980-\u09FF"      # Bengali
    r"\u0A00-\u0A7F"      # Gurmukhi
    r"\u0A80-\u0AFF"      # Gujarati
    r"\u0B00-\u0B7F"      # Oriya
    r"\u0B80-\u0BFF"      # Tamil
    r"\u0C00-\u0C7F"      # Telugu
    r"\u0C80-\u0CFF"      # Kannada
    r"\u0D00-\u0D7F"      # Malayalam
    r"\u0600-\u06FF"      # Arabic (Urdu)
    r"\u0750-\u077F"      # Arabic Supplement
    r"\u200C\u200D"       # ZWNJ / ZWJ
    r"a-zA-Z0-9"
    r"'\u2019"
)
_NON_TOKEN_CHAR = re.compile(rf"[^{_TOKEN_KEEP}]")

_SENT_END = r"[.!?\u0964\u0965|]"
_CLOSING = r"[\"'\u201c\u201d\u2018\u2019\u00ab\u00bb\u2039\u203a\)\]\}]"
_SENT_SPLIT_RE = re.compile(rf"(?:(?<={_SENT_END}{_CLOSING})|(?<={_SENT_END}))\s+")


def _tokenize(text):
    """Lowercase, strip punctuation, split on whitespace.

    Keeps Devanagari + Bengali + Tamil + ... characters intact. Empty
    tokens (those that were pure punctuation) are dropped.
    """
    words = []
    for raw in text.lower().split():
        cleaned = _NON_TOKEN_CHAR.sub("", raw)
        if cleaned:
            words.append(cleaned)
    return words


def _normalise_token(word):
    """Apply the same lowercasing + punctuation stripping to a single
    query word as to corpus tokens. Used by word_frequency."""
    return _NON_TOKEN_CHAR.sub("", word.lower())


def _split_sentences(text):
    """Split text into sentences using terminal punctuation only.

    Line breaks are treated as ordinary whitespace; a paragraph that
    spans multiple lines counts as one sentence (or several, depending
    on its punctuation), not as several. A trailing fragment with no
    terminal punctuation still counts as a sentence.
    """
    full_text = re.sub(r"\s+", " ", text).strip()
    if not full_text:
        return []
    parts = _SENT_SPLIT_RE.split(full_text)
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _lexical_diversity(words):
    """Type-Token Ratio on the full word list."""
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _mtld(words, threshold=0.72):
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).

    Walks token-by-token, starting a new segment every time the running
    TTR drops to threshold. Run forward + backward, return the mean.
    Length-independent by design.
    """
    if not words:
        return 0.0

    def _pass(ws):
        segments, types, tokens = 0, set(), 0
        for w in ws:
            types.add(w)
            tokens += 1
            if len(types) / tokens <= threshold:
                segments += 1
                types, tokens = set(), 0
        if tokens > 0:
            segments += (1 - len(types) / tokens) / (1 - threshold)
        return len(ws) / segments if segments > 0 else float(len(ws))

    return (_pass(words) + _pass(words[::-1])) / 2


def _hapax_pct(words):
    """Percentage of word types appearing exactly once."""
    if not words:
        return 0.0
    counts = Counter(words)
    hapax = sum(1 for c in counts.values() if c == 1)
    return 100.0 * hapax / len(counts)


def _ttr_window(words, window=1000):
    """TTR averaged over fixed-size non-overlapping windows.

    For texts shorter than `window`, returns plain TTR over what's there.
    """
    if not words:
        return 0.0
    if len(words) <= window:
        return _lexical_diversity(words)
    ratios = []
    for i in range(0, len(words) - window + 1, window):
        chunk = words[i:i + window]
        ratios.append(len(set(chunk)) / len(chunk))
    return sum(ratios) / len(ratios) if ratios else 0.0


def _ttr_at_n(words, n):
    """TTR on the first n tokens (or fewer if the text is shorter)."""
    if not words or n <= 0:
        return 0.0
    chunks = [words[i:i+n] for i in range(0, len(words), n)]
    # Keep only full-size chunks
    #chunks = [chunk for chunk in chunks if len(chunk) == n]
    if not chunks:
        return 0.0

    ttrs = [len(set(chunk)) / n for chunk in chunks]
    return sum(ttrs) / len(ttrs)


def _hapax_pct_at_n(words, n):
    """Hapax % on the first n tokens (or fewer if the text is shorter)."""
    if not words or n <= 0:
        return 0.0
    chunks = [words[i:i+n] for i in range(0, len(words), n)]

    values = []
    for chunk in chunks:
        counts = Counter(chunk)
        types = len(counts)
        hapax = sum(1 for c in counts.values() if c == 1)
        values.append(100.0 * hapax / len(counts))
        #print(
        #f"tokens={len(chunk)}, "
        #f"types={types}, "
        #f"hapax={hapax}, "
        #f"hapax/types={100*hapax/types:.1f}%, "
        #f"hapax/tokens={100*hapax/len(chunk):.1f}%"
        #)

    return sum(values) / len(values) if values else 0.0


def _stats_for_text(text, sample_size):
    """Compute the full metric dict for a single text.

    `sample_size` is the token budget for ttr_n / hapax_n.
    """
    words = _tokenize(text)
    sentences = _split_sentences(text)
    n_words = len(words)
    n_sents = len(sentences)
    n_chars = sum(1 for c in text if not c.isspace())

    if n_words == 0:
        return {
            "chars": n_chars, "words": 0, "tokens": 0, "sentences": n_sents,
            "avg_word": 0.0, "avg_sent": 0.0,
            "ld": 0.0, "mtld": 0.0, "hapax_pct": 0.0, "ttr_1k": 0.0,
            "ttr_n": 0.0, "hapax_n": 0.0,
        }

    return {
        "chars":     n_chars,
        "words":     n_words,
        "tokens":    len(set(words)),
        "sentences": n_sents,
        "avg_word":  sum(len(w) for w in words) / n_words,
        "avg_sent":  n_words / n_sents if n_sents else 0.0,
        "ld":        _lexical_diversity(words),
        "mtld":      _mtld(words),
        "hapax_pct": _hapax_pct(words),
        "ttr_1k":    _ttr_window(words, 1000),
        "ttr_n":     _ttr_at_n(words, sample_size),
        "hapax_n":   _hapax_pct_at_n(words, sample_size),
    }


# ---------------------------------------------------------------------------
# File listing and grouping
# ---------------------------------------------------------------------------

def _list_txt_files(folder):
    """Plain alphabetical list of .txt files in `folder`."""
    paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".txt") and os.path.isfile(os.path.join(folder, f))
    ]
    paths.sort()
    return paths


def _collect_files_numeric(folder):
    """Like _list_txt_files but sort numerically if all stems are
    integers (e.g. 1.txt, 2.txt, 10.txt produced by the scrapper)."""
    paths = _list_txt_files(folder)
    try:
        paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    except ValueError:
        pass  # already alphabetical from _list_txt_files
    return paths


def _author_of(filename):
    """Author = part before the first underscore in the filename stem."""
    stem = os.path.splitext(filename)[0]
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def _group_files_by_author(folder):
    """Return {author: [path, ...]} for all .txt files in folder."""
    by_author = defaultdict(list)
    for name in os.listdir(folder):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        by_author[_author_of(name)].append(path)
    for paths in by_author.values():
        paths.sort()
    return by_author


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_sample_label(n):
    """Render 5000 as '5k', 10000 as '10k', 1500 as '1.5k', 500 as '500'."""
    if n < 1000:
        return str(n)
    if n % 1000 == 0:
        return f"{n // 1000}k"
    return f"{n / 1000:g}k"


def _fmt(value, key):
    """Format a metric value with the right precision and colour."""
    if value is None:
        return _colour("—", key)
    if isinstance(value, float):
        if key in ("ld", "ttr_1k", "ttr_n"):
            text = f"{value:.4f}"
        elif key == "mtld":
            text = f"{value:.1f}"
        elif key in ("hapax_pct", "hapax_n"):
            text = f"{value:.1f}%"
        elif key == "avg_book_len":
            text = f"{value:,.0f}"
        else:
            text = f"{value:.2f}"
    else:
        text = f"{value:,}" if isinstance(value, int) and value >= 1000 else str(value)
    return _colour(text, key)


def _bold_cyan(s):
    return f"{_ANSI_BOLD}\033[36m{s}{_ANSI_RESET}"


_SCALE_OPTIONS = {
    "permille": (1000.0, "Relative frequency (per 1000 tokens, \u2030)"),
    "percent":  (100.0,  "Relative frequency (%)"),
    "pmw":      (1_000_000.0, "Relative frequency (per million words)"),
    "fraction": (1.0,    "Relative frequency (fraction of tokens)"),
}


def _scale_factor_label(scale):
    """Validate `scale` and return (multiplier, y-axis label)."""
    if scale not in _SCALE_OPTIONS:
        valid = ", ".join(repr(k) for k in _SCALE_OPTIONS)
        raise ValueError(
            f"scale must be one of: {valid}. Got: {scale!r}"
        )
    return _SCALE_OPTIONS[scale]


# ---------------------------------------------------------------------------
# Public API: corpus_stats (per-file)
# ---------------------------------------------------------------------------

def corpus_stats(input_folder, sample_size=5000):
    """Print a per-file lexical / structural statistics table.

    Parameters
    ----------
    input_folder : str
        Folder containing .txt files.
    sample_size : int
        Token budget for the fair-sample columns (TTR@N, Hapax@N).
        Files with fewer tokens are flagged in the Warning column.
    """
    if not os.path.isdir(input_folder):
        raise NotADirectoryError(f"Not a directory: {input_folder}")

    paths = _collect_files_numeric(input_folder)
    if not paths:
        raise FileNotFoundError(f"No .txt files in {input_folder}")

    sample_label = _format_sample_label(sample_size)
    print(f"Scanning {len(paths)} .txt files in {input_folder} "
          f"(sample_size={sample_label}) ...")
    print()

    rows = []
    full_text_parts = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            text = f.read()
        s = _stats_for_text(text, sample_size)
        s["file"] = os.path.basename(p)
        s["warning"] = "OK" if s["words"] >= sample_size else f"<{sample_label}"
        rows.append(s)
        full_text_parts.append(text)

    aggregate = _stats_for_text("\n".join(full_text_parts), sample_size)
    aggregate["file"] = f"{_ANSI_BOLD}TOTAL{_ANSI_RESET}"
    aggregate["warning"] = ""

    columns = [
        ("File",                  "file"),
        ("Chars",                 "chars"),
        ("Words",                 "words"),
        ("Types",                 "tokens"),
        ("Sentences",             "sentences"),
        ("Avg word",              "avg_word"),
        ("Avg sent",              "avg_sent"),
        #("MTLD",                  "mtld"),
        #("TTR-1k",                "ttr_1k"),
        (f"TTR@{sample_label}",   "ttr_n"),
        (f"Hapax@{sample_label}", "hapax_n"),
        #("LD (TTR)",              "ld"),
        ("Hapax %",               "hapax_pct"),
        ("Warning",               "warning"),
    ]

    print(_render_metric_table(rows, aggregate, columns,
                               left_aligned={"file", "warning"}))


# ---------------------------------------------------------------------------
# Public API: author_stats (per-author)
# ---------------------------------------------------------------------------

def author_stats(input_folder, sample_size=5000):
    """Print a per-author statistics table.

    Filenames must follow the convention "<Author>_<Title>.txt"; the
    part before the first underscore is treated as the author. All of
    an author's files are concatenated before computing metrics.
    """
    if not os.path.isdir(input_folder):
        raise NotADirectoryError(f"Not a directory: {input_folder}")

    by_author = _group_files_by_author(input_folder)
    if not by_author:
        raise FileNotFoundError(f"No .txt files in {input_folder}")

    n_files = sum(len(paths) for paths in by_author.values())
    sample_label = _format_sample_label(sample_size)
    print(f"Scanning {n_files} files across {len(by_author)} authors "
          f"in {input_folder} (sample_size={sample_label}) ...")
    print()

    rows = []
    all_text_parts = []
    for author in sorted(by_author):
        paths = by_author[author]
        parts = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                parts.append(f.read())
        author_text = "\n".join(parts)
        s = _stats_for_text(author_text, sample_size)
        s["author"] = author
        s["files"] = len(paths)
        s["avg_book_len"] = s["words"] / len(paths) if paths else 0.0
        s["warning"] = "OK" if s["words"] >= sample_size else f"<{sample_label}"
        rows.append(s)
        all_text_parts.append(author_text)

    aggregate = _stats_for_text("\n".join(all_text_parts), sample_size)
    aggregate["author"] = f"{_ANSI_BOLD}TOTAL{_ANSI_RESET}"
    aggregate["files"] = n_files
    aggregate["avg_book_len"] = aggregate["words"] / n_files if n_files else 0.0
    aggregate["warning"] = ""

    columns = [
        ("Author",                "author"),
        ("Files",                 "files"),
        ("Words",                 "words"),
        ("Avg book",              "avg_book_len"),
        ("Avg word",              "avg_word"),
        ("Avg sent",              "avg_sent"),
        #("MTLD",                  "mtld"),
        #("TTR-1k",                "ttr_1k"),
        (f"TTR@{sample_label}",   "ttr_n"),
        (f"Hapax@{sample_label}", "hapax_n"),
        #("LD (TTR)",              "ld"),
        ("Hapax %",               "hapax_pct"),
        ("Warning",               "warning"),
    ]

    print(_render_metric_table(rows, aggregate, columns,
                               left_aligned={"author", "warning"}))


def _render_metric_table(rows, summary_row, columns, left_aligned):
    """Shared renderer for corpus_stats and author_stats tables."""
    table = PrettyTable()
    table.field_names = [_bold_cyan(label) for label, _ in columns]
    for label, key in columns:
        align = "l" if key in left_aligned else "r"
        table.align[_bold_cyan(label)] = align

    for r in rows:
        table.add_row([_fmt(r[key], key) for _, key in columns])

    table.add_row(["─" * 12] + ["─" * 8] * (len(columns) - 1))
    table.add_row([_fmt(summary_row[key], key) for _, key in columns])
    return table


# ---------------------------------------------------------------------------
# Public API: plot_authors (scatter plot)
# ---------------------------------------------------------------------------

_METRIC_LABELS_FIXED = {
    "chars":         "Characters (no whitespace)",
    "words":         "Words (tokens)",
    "tokens":        "Word types (unique words)",
    "sentences":     "Sentences",
    "avg_word":      "Avg word length (characters)",
    "avg_sent":      "Avg sentence length (words)",
    "ld":            "Lexical diversity (TTR, full text)",
    "mtld":          "MTLD (lexical diversity, length-independent)",
    "hapax_pct":     "Hapax % (full text)",
    "ttr_1k":        "TTR-1k (mean TTR over 1000-word windows)",
    "avg_book_len":  "Avg book length (words per file)",
    "files":         "Files (books in corpus)",
}


def _metric_labels(sample_size):
    sample_label = _format_sample_label(sample_size)
    labels = dict(_METRIC_LABELS_FIXED)
    labels["ttr_n"] = f"TTR @ {sample_label} tokens (fair sample)"
    labels["hapax_n"] = f"Hapax % @ {sample_label} tokens (fair sample)"
    return labels


def plot_authors(
    input_folder,
    x_metric="mtld",
    y_metric="avg_sent",
    size_metric=None,
    sample_size=5000,
    figsize=(10, 7),
    point_size=120,
    label_offset=(0.5, 0.5),
    title=None,
):
    """Scatter plot of authors using two stylometric metrics.

    Parameters
    ----------
    input_folder : str
        Folder of "<Author>_<Title>.txt" files.
    x_metric, y_metric : str
        Metric keys: chars, words, tokens, sentences, avg_word, avg_sent,
        ld, mtld, hapax_pct, ttr_1k, ttr_n, hapax_n, avg_book_len, files.
    size_metric : str or None
        Optional third metric mapped to point size.
    sample_size : int
        Used only when ttr_n or hapax_n is involved.
    figsize, point_size, label_offset, title :
        matplotlib styling.

    Returns
    -------
    matplotlib.figure.Figure
    """
    by_author = _group_files_by_author(input_folder)
    if not by_author:
        raise FileNotFoundError(f"No .txt files in {input_folder}")

    rows = []
    for author in sorted(by_author):
        paths = by_author[author]
        parts = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                parts.append(f.read())
        author_text = "\n".join(parts)
        s = _stats_for_text(author_text, sample_size)
        s["author"] = author
        s["files"] = len(paths)
        s["avg_book_len"] = s["words"] / len(paths) if paths else 0.0
        rows.append(s)

    xs = [r[x_metric] for r in rows]
    ys = [r[y_metric] for r in rows]
    labels = [r["author"] for r in rows]

    if size_metric:
        size_values = [r.get(size_metric, 0) or 0 for r in rows]
        lo, hi = min(size_values), max(size_values)
        if hi > lo:
            sizes = [40 + (v - lo) / (hi - lo) * (point_size - 40)
                     for v in size_values]
        else:
            sizes = [point_size] * len(size_values)
    else:
        sizes = [point_size] * len(rows)

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.scatter(
        xs, ys,
        s=sizes,
        c="#1f77b4",
        alpha=0.75,
        edgecolors="black",
        linewidths=0.7,
        zorder=3,
    )

    dx, dy = label_offset
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            zorder=4,
        )

    metric_labels = _metric_labels(sample_size)
    ax.set_xlabel(metric_labels.get(x_metric, x_metric), fontsize=11)
    ax.set_ylabel(metric_labels.get(y_metric, y_metric), fontsize=11)

    if title is None:
        title = f"Authors in stylometric space ({len(rows)} authors)"
    ax.set_title(title, fontsize=12)

    if size_metric:
        fig.text(
            0.99, 0.01,
            f"point size \u221d {metric_labels.get(size_metric, size_metric)}",
            ha="right", va="bottom",
            fontsize=8, style="italic", color="gray",
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public API: plot_authors_3d (rotating 3D scatter)
# ---------------------------------------------------------------------------

def plot_authors_3d(
    input_folder,
    x_metric="mtld",
    y_metric="avg_sent",
    z_metric="hapax_pct",
    size_metric=None,
    colors=None,
    sample_size=5000,
    figsize=(9, 9),
    point_size=120,
    title=None,
    frames=60,
    interval=80,
):
    """3D scatter of authors in stylometric space, rendered as a
    rotating animation. Each author gets a unique colour.

    Parameters
    ----------
    input_folder : str
        Folder of "<Author>_<Title>.txt" files.
    x_metric, y_metric, z_metric : str
        Metric keys: chars, words, tokens, sentences, avg_word, avg_sent,
        ld, mtld, hapax_pct, ttr_1k, ttr_n, hapax_n, avg_book_len, files.
    size_metric : str or None
        Optional fourth metric mapped to point size.
    colors : None | list | dict {author: colour}
        Per-author colours. None -> tab10 palette in author order.
        dict -> {"Austen": "blue", ...}; missing keys fall back to tab10.
        list -> same length as authors (alphabetical), positional.
    sample_size : int
        Used only when ttr_n or hapax_n is involved.
    figsize, point_size, title : matplotlib styling.
    frames : number of animation frames (one full 360 rotation).
    interval : ms per frame.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Display in Jupyter with `HTML(anim.to_jshtml())`. Save with
        `anim.save("authors_3d.gif", writer="pillow", fps=20)`.
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import to_rgba

    by_author = _group_files_by_author(input_folder)
    if not by_author:
        raise FileNotFoundError(f"No .txt files in {input_folder}")

    rows = []
    for author in sorted(by_author):
        paths = by_author[author]
        parts = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                parts.append(f.read())
        author_text = "\n".join(parts)
        s = _stats_for_text(author_text, sample_size)
        s["author"] = author
        s["files"] = len(paths)
        s["avg_book_len"] = s["words"] / len(paths) if paths else 0.0
        rows.append(s)

    authors = [r["author"] for r in rows]
    xs = [r[x_metric] for r in rows]
    ys = [r[y_metric] for r in rows]
    zs = [r[z_metric] for r in rows]

    # Per-author colours.
    cmap = plt.get_cmap("tab10")
    if colors is None:
        rgba_per_author = [cmap(i % 10) for i in range(len(authors))]
    elif isinstance(colors, dict):
        rgba_per_author = [
            to_rgba(colors[a]) if a in colors else cmap(i % 10)
            for i, a in enumerate(authors)
        ]
    elif isinstance(colors, (list, tuple)):
        if len(colors) != len(authors):
            raise ValueError(
                f"colors length ({len(colors)}) must equal "
                f"number of authors ({len(authors)})")
        rgba_per_author = [to_rgba(c) for c in colors]
    else:
        raise TypeError(f"Unsupported colors type: {type(colors).__name__}")

    # Point sizes.
    if size_metric:
        size_values = [r.get(size_metric, 0) or 0 for r in rows]
        lo, hi = min(size_values), max(size_values)
        if hi > lo:
            sizes = [40 + (v - lo) / (hi - lo) * (point_size - 40)
                     for v in size_values]
        else:
            sizes = [point_size] * len(size_values)
    else:
        sizes = [point_size] * len(rows)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    for i, author in enumerate(authors):
        ax.scatter(
            [xs[i]], [ys[i]], [zs[i]],
            s=[sizes[i]],
            color=[rgba_per_author[i]],
            edgecolors="black",
            linewidths=0.7,
            label=author,
            depthshade=True,
        )
        dx = (max(xs) - min(xs)) * 0.012 if max(xs) > min(xs) else 0
        dy = (max(ys) - min(ys)) * 0.012 if max(ys) > min(ys) else 0
        dz = (max(zs) - min(zs)) * 0.012 if max(zs) > min(zs) else 0
        ax.text(xs[i] + dx, ys[i] + dy, zs[i] + dz,
                author, fontsize=8, color=rgba_per_author[i])

    metric_labels = _metric_labels(sample_size)
    ax.set_xlabel(metric_labels.get(x_metric, x_metric), fontsize=10)
    ax.set_ylabel(metric_labels.get(y_metric, y_metric), fontsize=10)
    ax.set_zlabel(metric_labels.get(z_metric, z_metric), fontsize=10)

    if title is None:
        title = f"Authors in 3D stylometric space ({len(rows)} authors)"
    ax.set_title(title, fontsize=12)

    if size_metric:
        fig.text(
            0.99, 0.01,
            f"point size \u221d {metric_labels.get(size_metric, size_metric)}",
            ha="right", va="bottom",
            fontsize=8, style="italic", color="gray",
        )

    def _update(frame):
        ax.view_init(elev=20, azim=frame * (360.0 / frames))
        return []

    anim = FuncAnimation(fig, _update, frames=frames,
                          interval=interval, blit=False)
    fig.tight_layout()
    plt.close(fig)
    return anim


# ---------------------------------------------------------------------------
# Public API: word_frequency (per-file table)
# ---------------------------------------------------------------------------

def word_frequency(input_folder, words):
    """Print a per-file frequency table for a list of target words.

    For every file the table shows:
      Tokens           total tokens in the file
      <word> #         absolute count of that word
      <word> ‰         relative frequency, per 1000 tokens
    A TOTAL row aggregates over the whole folder.
    """
    if not os.path.isdir(input_folder):
        raise NotADirectoryError(f"Not a directory: {input_folder}")

    paths = _list_txt_files(input_folder)
    if not paths:
        raise FileNotFoundError(f"No .txt files in {input_folder}")

    targets = [_normalise_token(w) for w in words]
    if any(not t for t in targets):
        bad = [w for w, t in zip(words, targets) if not t]
        raise ValueError(f"These words tokenise to empty: {bad}")

    print(f"Scanning {len(paths)} files in {input_folder} for "
          f"{len(targets)} word(s) ...")
    print()

    rows = []
    total_counts = Counter()
    total_tokens = 0

    for path in paths:
        with open(path, encoding="utf-8") as f:
            tokens = _tokenize(f.read())
        n = len(tokens)
        counts = Counter(tokens)
        rows.append({
            "file": os.path.basename(path),
            "tokens": n,
            "counts": {t: counts.get(t, 0) for t in targets},
        })
        for t in targets:
            total_counts[t] += counts.get(t, 0)
        total_tokens += n

    table = PrettyTable()
    field_names = [_bold_cyan("File"), _bold_cyan("Tokens")]
    for w in words:
        field_names.append(_bold_cyan(f"{w} #"))
        field_names.append(_bold_cyan(f"{w} \u2030"))
    table.field_names = field_names
    table.align[field_names[0]] = "l"
    for fn in field_names[1:]:
        table.align[fn] = "r"

    for r in rows:
        cells = [_palette_colour(r["file"], 0), f"{r['tokens']:,}"]
        for i, t in enumerate(targets):
            c = r["counts"][t]
            permille = (c / r["tokens"] * 1000.0) if r["tokens"] else 0.0
            cells.append(_palette_colour(f"{c:,}", i + 1))
            cells.append(_palette_colour(f"{permille:.2f}", i + 1))
        table.add_row(cells)

    table.add_row(["─" * 12] + ["─" * 8] * (len(field_names) - 1))
    total_cells = [
        f"{_ANSI_BOLD}\033[36mTOTAL{_ANSI_RESET}",
        f"{total_tokens:,}",
    ]
    for i, t in enumerate(targets):
        c = total_counts[t]
        permille = (c / total_tokens * 1000.0) if total_tokens else 0.0
        total_cells.append(_palette_colour(f"{c:,}", i + 1))
        total_cells.append(_palette_colour(f"{permille:.2f}", i + 1))
    table.add_row(total_cells)

    print(table)


# ---------------------------------------------------------------------------
# Public API: word_frequency_plot (per-file time series)
# ---------------------------------------------------------------------------

def word_frequency_plot(
    input_folder,
    words,
    chunk_size=1000,
    scale="permille",
    figsize=(10, 5),
):
    """One time-series figure per .txt file.

    Each file is split into consecutive `chunk_size`-token chunks, and
    we plot per-chunk relative frequency of each target word as a line.
    The x-axis is the token offset within the file, so longer books get
    longer x-axes.

    Parameters
    ----------
    input_folder : str
    words : list of str
    chunk_size : int
        Tokens per chunk.
    scale : {"permille", "percent", "pmw", "fraction"}
        Unit for the y-axis. Default "permille" (per 1000 tokens, \u2030),
        the corpus-linguistics standard. "percent" (\u00d7100), "pmw"
        (per million words, \u00d71 000 000), or "fraction" (\u00d71). The
        shape of the curve is the same regardless of `scale`; only the
        units on the y-axis differ.
    figsize : (w, h)

    Returns
    -------
    list of matplotlib.figure.Figure
    """
    if not os.path.isdir(input_folder):
        raise NotADirectoryError(f"Not a directory: {input_folder}")

    paths = _list_txt_files(input_folder)
    if not paths:
        raise FileNotFoundError(f"No .txt files in {input_folder}")

    targets = [_normalise_token(w) for w in words]
    if any(not t for t in targets):
        bad = [w for w, t in zip(words, targets) if not t]
        raise ValueError(f"These words tokenise to empty: {bad}")

    multiplier, y_label = _scale_factor_label(scale)

    figures = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            tokens = _tokenize(f.read())

        if not tokens:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"{os.path.basename(path)} is empty",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            figures.append(fig)
            continue

        x_positions = []
        series = {t: [] for t in targets}
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            if not chunk:
                continue
            counts = Counter(chunk)
            x_positions.append(i + len(chunk) / 2.0)
            for t in targets:
                series[t].append(counts.get(t, 0) / len(chunk) * multiplier)

        fig, ax = plt.subplots(figsize=figsize)
        for word, target in zip(words, targets):
            ax.plot(x_positions, series[target], label=word, linewidth=1.5)

        ax.set_xlabel(f"Token position (chunk = {chunk_size} tokens)")
        ax.set_ylabel(y_label)
        ax.set_title(f"{os.path.basename(path)} \u2014 "
                     f"{len(tokens):,} tokens")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_xlim(0, len(tokens))
        fig.tight_layout()
        figures.append(fig)

    return figures


# ---------------------------------------------------------------------------
# Public API: word_frequency_combined (one figure for the whole folder)
# ---------------------------------------------------------------------------

def word_frequency_combined(
    input_folder,
    words,
    chunk_size=1000,
    scale="permille",
    figsize=(12, 6),
    title=None,
):
    """One figure showing all files concatenated as a single time series.

    Each file is split into consecutive `chunk_size`-token chunks; the
    last chunk of a file may be shorter. For every chunk we compute the
    relative frequency of each requested word. Chunks from all files
    are concatenated in alphabetical filename order, so the x-axis
    represents the corpus as a whole, file boundaries marked by light
    vertical dotted lines.

    The x-axis is the chunk index (not token offset), so files of
    different lengths take up proportional horizontal space. If there
    are 30 or fewer files, the file names are placed under the centre
    of each region instead of plain numeric ticks.

    Parameters
    ----------
    input_folder : str
    words : list of str
    chunk_size : int
        Tokens per chunk.
    scale : {"permille", "percent", "pmw", "fraction"}
        Unit for the y-axis. Default "permille" (per 1000 tokens, \u2030),
        the corpus-linguistics standard. "percent" (\u00d7100), "pmw"
        (per million words, \u00d71 000 000), or "fraction" (\u00d71). The
        shape of the curve is the same regardless of `scale`; only the
        units on the y-axis differ.
    figsize : (w, h)
    title : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not os.path.isdir(input_folder):
        raise NotADirectoryError(f"Not a directory: {input_folder}")

    paths = _list_txt_files(input_folder)
    if not paths:
        raise FileNotFoundError(f"No .txt files in {input_folder}")

    targets = [_normalise_token(w) for w in words]
    if any(not t for t in targets):
        bad = [w for w, t in zip(words, targets) if not t]
        raise ValueError(f"These words tokenise to empty: {bad}")

    multiplier, y_label = _scale_factor_label(scale)

    series = {t: [] for t in targets}
    boundaries = []
    file_labels = []

    for path in paths:
        with open(path, encoding="utf-8") as f:
            tokens = _tokenize(f.read())

        start_idx = len(series[targets[0]])
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            if not chunk:
                continue
            counts = Counter(chunk)
            for t in targets:
                series[t].append(counts.get(t, 0) / len(chunk) * multiplier)
        end_idx = len(series[targets[0]])

        if end_idx > start_idx:
            mid = (start_idx + end_idx - 1) / 2.0
            file_labels.append((mid, os.path.basename(path)))
        boundaries.append(end_idx)

    if not series[targets[0]]:
        raise ValueError("Corpus produced zero chunks. Is it empty?")

    fig, ax = plt.subplots(figsize=figsize)
    x = list(range(len(series[targets[0]])))
    for word, target in zip(words, targets):
        ax.plot(x, series[target], label=word, linewidth=1.5)

    for b in boundaries[:-1]:
        ax.axvline(b - 0.5, color="lightgray", linestyle=":",
                   linewidth=0.8, zorder=0)

    if len(file_labels) <= 30:
        positions = [m for m, _ in file_labels]
        labels = [name for _, name in file_labels]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xlabel(f"Chunk index ({chunk_size} tokens each)")

    ax.set_ylabel(y_label)
    if title is None:
        title = (f"Word frequency across {len(paths)} files "
                 f"(chunk = {chunk_size} tokens)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public API: top_words_plot (visualising stop-word dominance)
# ---------------------------------------------------------------------------

def top_words_plot(input_file, n=20, figsize=None, title=None):
    """Horizontal bar chart of the `n` most frequent words in a single file.

    Useful for showing students why stop-word removal matters: in a
    typical English text, the top 20 words are almost always
    grammatical function words (the, of, and, to, ...) that dominate
    the raw counts and obscure content words.

    Bars are sorted with the most frequent word at the **top** of the
    chart. The x-axis shows each word's share of the file as a
    percentage; the title gives the cumulative share of the top-n.

    Parameters
    ----------
    input_file : str
        Path to a single .txt file.
    n : int
        How many of the most frequent words to show.
    figsize : (w, h) or None
        Figure size. If None (default), height is scaled to fit n
        bars without crowding.
    title : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Not a file: {input_file}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    with open(input_file, encoding="utf-8") as f:
        tokens = _tokenize(f.read())

    if not tokens:
        raise ValueError(f"{input_file} contains no tokens.")

    total = len(tokens)
    counts = Counter(tokens)
    top = counts.most_common(n)
    # Reverse so that #1 ends up at the top of the chart (matplotlib's
    # y-axis grows upwards, so the first item plotted is at the bottom).
    words = [w for w, _ in top][::-1]
    percents = [c / total * 100.0 for _, c in top][::-1]

    # Adjust height so each bar gets ~0.3 inch -- prevents crowding
    # for large n while keeping small n compact.
    if figsize is None:
        figsize = (8, max(4, n * 0.3))

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(words)), percents,
            color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel("Share of all tokens (%)")

    if title is None:
        cumulative = sum(percents)
        title = (f"Top {n} words in {os.path.basename(input_file)} "
                 f"\u2014 {total:,} tokens, top-{n} = {cumulative:.1f}% of text")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4, zorder=0)

    # Annotate each bar with the percentage value.
    for i, p in enumerate(percents):
        ax.text(p, i, f" {p:.1f}%", va="center", ha="left", fontsize=8)

    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# English stop words (standard NLTK list, 179 words)
# ---------------------------------------------------------------------------

_ENGLISH_STOP_WORDS = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "you're", "you've", "you'll", "you'd",
    "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "she's", "her", "hers", "herself",
    "it", "it's", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "that'll",
    "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or",
    "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under",
    "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just",
    "don", "don't", "should", "should've", "now", "d", "ll", "m", "o",
    "re", "ve", "y",
    "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't",
    "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't",
    "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't",
    "mustn", "mustn't", "needn", "needn't",
    "shan", "shan't", "shouldn", "shouldn't",
    "wasn", "wasn't", "weren", "weren't",
    "won", "won't", "wouldn", "wouldn't",
})


# ---------------------------------------------------------------------------
# Public API: compare_authors_heatmap
# ---------------------------------------------------------------------------

def compare_authors_heatmap(
    input_folder,
    n=30,
    exclude_stop_words=False,
    cmap="YlOrRd",
    figsize=None,
    annotate=True,
    title=None,
):
    """Heatmap comparing how authors use the top-n words in the corpus.

    Rows are authors (sorted alphabetically), columns are the n most
    frequent words across the entire corpus. Each cell shows the
    percentage that word makes up of that author's total tokens, both
    as a colour and as a numeric label.

    By default the top-n includes stop words like "the" and "of",
    which actually carry stylometric signal -- different authors use
    them in different proportions, and these differences are a
    classic clue for authorship attribution (Mosteller & Wallace,
    1964). Set `exclude_stop_words=True` to skip them and reveal
    content words instead (character names, themes, distinctive
    vocabulary).

    Parameters
    ----------
    input_folder : str
        Folder of "<Author>_<Title>.txt" files.
    n : int
        How many top words to include as columns.
    exclude_stop_words : bool
        If True, drop common English stop words (~180 of them) before
        picking the top n. Default False.
    cmap : str
        Matplotlib colormap name. Default "YlOrRd". Other good
        sequential options: "viridis", "Blues", "Greens", "magma".
    figsize : (w, h) or None
        Figure size. If None, scales to fit n columns and number of
        authors.
    annotate : bool
        If True, write the percentage value inside each cell.
    title : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not os.path.isdir(input_folder):
        raise NotADirectoryError(f"Not a directory: {input_folder}")

    by_author = _group_files_by_author(input_folder)
    if not by_author:
        raise FileNotFoundError(f"No .txt files in {input_folder}")

    # ------------------------------------------------------------------
    # 1. Tokenise every author's full text once.
    # ------------------------------------------------------------------
    author_tokens = {}
    for author in sorted(by_author):
        parts = []
        for p in by_author[author]:
            with open(p, encoding="utf-8") as f:
                parts.append(f.read())
        author_tokens[author] = _tokenize("\n".join(parts))

    # ------------------------------------------------------------------
    # 2. Pick top-n words from the global corpus.
    # ------------------------------------------------------------------
    global_counts = Counter()
    for tokens in author_tokens.values():
        global_counts.update(tokens)

    if exclude_stop_words:
        for sw in _ENGLISH_STOP_WORDS:
            global_counts.pop(sw, None)

    top_words = [w for w, _ in global_counts.most_common(n)]
    if not top_words:
        raise ValueError("No words left after filtering. "
                         "Is the corpus empty?")

    # ------------------------------------------------------------------
    # 3. Build the matrix: rows = authors, cols = top_words.
    #    Values are percentages of the author's total tokens.
    # ------------------------------------------------------------------
    authors = sorted(author_tokens)
    matrix = []
    for author in authors:
        tokens = author_tokens[author]
        counts = Counter(tokens)
        total = len(tokens)
        if total == 0:
            row = [0.0] * len(top_words)
        else:
            row = [counts.get(w, 0) / total * 100.0 for w in top_words]
        matrix.append(row)

    # ------------------------------------------------------------------
    # 4. Draw the heatmap.
    # ------------------------------------------------------------------
    if figsize is None:
        figsize = (max(8, len(top_words) * 0.45),
                   max(4, len(authors) * 0.55))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(top_words)))
    ax.set_xticklabels(top_words, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(authors)))
    ax.set_yticklabels(authors, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Share of author's text (%)", fontsize=9)

    if annotate:
        # Choose label colour based on cell intensity for readability.
        max_val = max((max(row) for row in matrix), default=0)
        threshold = max_val * 0.55 if max_val > 0 else 0
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                colour = "white" if val > threshold else "black"
                if val >= 1:
                    label = f"{val:.1f}"
                elif val > 0:
                    label = f"{val:.2f}"
                else:
                    label = "·"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=7.5, color=colour)

    if title is None:
        suffix = " (stop words excluded)" if exclude_stop_words else ""
        title = (f"Top {n} words across {len(authors)} authors{suffix}")
    ax.set_title(title, fontsize=12, pad=12)

    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Public API: cumulative_coverage_plot
# ---------------------------------------------------------------------------

def cumulative_coverage_plot(
    input_file,
    max_n=100,
    figsize=(10, 6),
    title=None,
    annotate_milestones=True,
):
    """Plot cumulative text coverage by the top-k most frequent words.

    For every k from 1 to max_n, computes what percentage of the
    text's tokens are accounted for by the k most frequent words.
    The resulting curve climbs steeply at first (the top few stop
    words alone cover a huge fraction) and then flattens out — a
    classic visual demonstration of Zipf's law.

    Parameters
    ----------
    input_file : str
        Path to a single .txt file.
    max_n : int
        Plot from top-1 to top-max_n. Capped at the number of unique
        word types in the text.
    figsize : (w, h)
    title : str or None
    annotate_milestones : bool
        If True, mark and label the values at k = 10, 20, 50, 100
        (those that fall within max_n).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Not a file: {input_file}")
    if max_n < 1:
        raise ValueError(f"max_n must be >= 1, got {max_n}")

    with open(input_file, encoding="utf-8") as f:
        tokens = _tokenize(f.read())

    if not tokens:
        raise ValueError(f"{input_file} contains no tokens.")

    total = len(tokens)
    counts = Counter(tokens)
    n_types = len(counts)

    # Cap max_n at the number of available types.
    effective_max = min(max_n, n_types)
    top_counts = [c for _, c in counts.most_common(effective_max)]

    # Cumulative sum of percentages -- top-1 share, top-2 share, ...
    xs = list(range(1, effective_max + 1))
    cumulative = []
    running = 0
    for c in top_counts:
        running += c
        cumulative.append(running / total * 100.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs, cumulative, color="#1f77b4", linewidth=2, zorder=3)
    ax.fill_between(xs, cumulative, alpha=0.15, color="#1f77b4", zorder=2)

    ax.set_xlabel("Top-k most frequent words")
    ax.set_ylabel("Cumulative share of text (%)")

    if title is None:
        title = (f"Cumulative text coverage by top-k words \u2014 "
                 f"{os.path.basename(input_file)} ({total:,} tokens, "
                 f"{n_types:,} types)")
    ax.set_title(title, fontsize=11)

    ax.set_xlim(0, effective_max)
    ax.set_ylim(0, min(100, max(cumulative) * 1.05))
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    # Milestone markers at k = 10, 20, 50, 100 -- only those within range.
    if annotate_milestones:
        milestones = [k for k in (10, 20, 50, 100) if k <= effective_max]
        for k in milestones:
            pct = cumulative[k - 1]
            ax.plot(k, pct, "o", color="#d62728", markersize=7,
                    zorder=4, clip_on=False)
            ax.annotate(
                f"top {k} = {pct:.1f}%",
                xy=(k, pct),
                xytext=(8, -4),
                textcoords="offset points",
                fontsize=9,
                color="#d62728",
                fontweight="bold",
            )

    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Public API: sentence_length_histogram
# ---------------------------------------------------------------------------

def sentence_length_histogram(input_file, bins=50, figsize=(10, 5),
                               title=None):
    """Histogram of sentence lengths (in words) for a single file.

    Descriptive statistics like "Avg sent = 18" hide whether the
    distribution is symmetric, skewed, or bimodal. A histogram tells
    you: does this author write uniformly, or alternate short dialogue
    with long descriptions?

    Parameters
    ----------
    input_file : str
    bins : int
    figsize : (w, h)
    title : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Not a file: {input_file}")

    with open(input_file, encoding="utf-8") as f:
        text = f.read()

    sentences = _split_sentences(text)
    if not sentences:
        raise ValueError(f"{input_file} contains no sentences.")

    lengths = [len(_tokenize(s)) for s in sentences]
    lengths = [l for l in lengths if l > 0]
    if not lengths:
        raise ValueError(f"{input_file} contains no non-empty sentences.")

    mean = sum(lengths) / len(lengths)
    median = sorted(lengths)[len(lengths) // 2]

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(lengths, bins=bins, color="#1f77b4",
            edgecolor="black", linewidth=0.5)
    ax.axvline(mean, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"mean = {mean:.1f}")
    ax.axvline(median, color="#2ca02c", linestyle="--", linewidth=1.5,
               label=f"median = {median}")

    ax.set_xlabel("Sentence length (words)")
    ax.set_ylabel("Number of sentences")
    if title is None:
        title = (f"Sentence length distribution \u2014 "
                 f"{os.path.basename(input_file)} "
                 f"({len(lengths):,} sentences)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public API: sentence_length_over_text
# ---------------------------------------------------------------------------

def sentence_length_over_text(input_file, window=20, figsize=(11, 5),
                               title=None):
    """Rolling mean of sentence length across the text.

    For each position in the text we compute the mean length of the
    surrounding window of sentences. Peaks indicate descriptive
    passages; troughs indicate dialogue or rapid action.

    Parameters
    ----------
    input_file : str
    window : int
        Number of consecutive sentences to average. Larger = smoother.
    figsize : (w, h)
    title : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Not a file: {input_file}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    with open(input_file, encoding="utf-8") as f:
        text = f.read()

    sentences = _split_sentences(text)
    lengths = [len(_tokenize(s)) for s in sentences if _tokenize(s)]
    if len(lengths) < window:
        raise ValueError(
            f"{input_file} has only {len(lengths)} sentences, "
            f"need at least {window} for the chosen window."
        )

    # Rolling mean -- one value per sentence, except the first window-1
    # have no full window behind them; we centre instead.
    half = window // 2
    rolling = []
    for i in range(len(lengths)):
        lo = max(0, i - half)
        hi = min(len(lengths), i + half + 1)
        chunk = lengths[lo:hi]
        rolling.append(sum(chunk) / len(chunk))

    overall_mean = sum(lengths) / len(lengths)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(len(rolling)), rolling, color="#1f77b4", linewidth=1.5)
    ax.axhline(overall_mean, color="#d62728", linestyle="--",
               linewidth=1.2, label=f"overall mean = {overall_mean:.1f}")

    ax.set_xlabel(f"Sentence index (rolling window = {window} sentences)")
    ax.set_ylabel("Sentence length (words)")
    if title is None:
        title = (f"Sentence length trajectory \u2014 "
                 f"{os.path.basename(input_file)}")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_xlim(0, len(rolling))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public API: word_length_histogram
# ---------------------------------------------------------------------------

def word_length_histogram(input_file, max_length=20, figsize=(10, 5),
                           title=None):
    """Histogram of word lengths (in characters) for a single file.

    English words peak around 3-4 characters (because of the high
    frequency of short function words like "the", "of", "a"). The
    long tail tells you about technical or formal vocabulary.

    Parameters
    ----------
    input_file : str
    max_length : int
        Words longer than this are still counted but binned into the
        rightmost bar to keep the plot readable.
    figsize : (w, h)
    title : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Not a file: {input_file}")

    with open(input_file, encoding="utf-8") as f:
        tokens = _tokenize(f.read())
    if not tokens:
        raise ValueError(f"{input_file} contains no tokens.")

    counts = Counter()
    for w in tokens:
        L = len(w)
        if L > max_length:
            L = max_length
        counts[L] += 1

    lengths = sorted(counts.keys())
    freqs = [counts[L] for L in lengths]
    total = sum(freqs)
    percents = [f / total * 100.0 for f in freqs]

    mean = sum(len(w) for w in tokens) / len(tokens)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(lengths, percents, color="#1f77b4",
           edgecolor="black", linewidth=0.5)
    ax.axvline(mean, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"mean = {mean:.2f}")

    ax.set_xlabel("Word length (characters)")
    ax.set_ylabel("Share of tokens (%)")
    if title is None:
        title = (f"Word length distribution \u2014 "
                 f"{os.path.basename(input_file)} ({total:,} tokens)")
    ax.set_title(title)

    xt_labels = [str(L) for L in lengths]
    if lengths and lengths[-1] == max_length:
        xt_labels[-1] = f"{max_length}+"
    ax.set_xticks(lengths)
    ax.set_xticklabels(xt_labels)

    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public API: word_length_over_text
# ---------------------------------------------------------------------------

def word_length_over_text(input_file, chunk_size=1000, figsize=(11, 5),
                           title=None):
    """Rolling mean of word length across the text.

    Each chunk of `chunk_size` tokens gets one point: the mean word
    length in that chunk. Register shifts (technical, formal,
    colloquial) show as visible jumps.

    Parameters
    ----------
    input_file : str
    chunk_size : int
    figsize : (w, h)
    title : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Not a file: {input_file}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    with open(input_file, encoding="utf-8") as f:
        tokens = _tokenize(f.read())
    if not tokens:
        raise ValueError(f"{input_file} contains no tokens.")

    means = []
    positions = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]
        if not chunk:
            continue
        means.append(sum(len(w) for w in chunk) / len(chunk))
        positions.append(i + len(chunk) / 2.0)

    overall_mean = sum(len(w) for w in tokens) / len(tokens)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(positions, means, color="#1f77b4", linewidth=1.5, marker="o",
            markersize=4)
    ax.axhline(overall_mean, color="#d62728", linestyle="--",
               linewidth=1.2, label=f"overall mean = {overall_mean:.2f}")

    ax.set_xlabel(f"Token position (chunk = {chunk_size} tokens)")
    ax.set_ylabel("Mean word length (characters)")
    if title is None:
        title = (f"Word length trajectory \u2014 "
                 f"{os.path.basename(input_file)}")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_xlim(0, len(tokens))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public API: zipf_plot
# ---------------------------------------------------------------------------

def zipf_plot(input_file, top_n=None, figsize=(9, 6), title=None):
    """Log-log plot of word rank vs frequency (Zipf's law).

    Plots every distinct word in the file as a point: its rank (1st,
    2nd, ... most frequent) on the x-axis, its frequency on the y-axis,
    both on log scales. A power-law distribution like Zipf's gives an
    approximately straight line.

    Parameters
    ----------
    input_file : str
    top_n : int or None
        Plot only the top-n ranks. None means all of them.
    figsize : (w, h)
    title : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Not a file: {input_file}")

    with open(input_file, encoding="utf-8") as f:
        tokens = _tokenize(f.read())
    if not tokens:
        raise ValueError(f"{input_file} contains no tokens.")

    counts = Counter(tokens)
    sorted_freqs = sorted(counts.values(), reverse=True)
    if top_n:
        sorted_freqs = sorted_freqs[:top_n]
    ranks = list(range(1, len(sorted_freqs) + 1))

    fig, ax = plt.subplots(figsize=figsize)
    ax.loglog(ranks, sorted_freqs, "o", markersize=3, color="#1f77b4",
              alpha=0.6)

    # Reference Zipf line: f(r) = f(1) / r
    if sorted_freqs:
        ref = [sorted_freqs[0] / r for r in ranks]
        ax.loglog(ranks, ref, "--", color="#d62728", linewidth=1.5,
                  label="ideal Zipf (slope = -1)")

    ax.set_xlabel("Word rank (log)")
    ax.set_ylabel("Frequency (log)")
    if title is None:
        title = (f"Zipf's law in {os.path.basename(input_file)} "
                 f"({len(tokens):,} tokens, {len(counts):,} types)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public API: ttr_curve
# ---------------------------------------------------------------------------

def ttr_curve(input_file, step=500, figsize=(10, 5), title=None):
    """TTR computed on progressively larger samples of the same text.

    TTR is type-token ratio: types / tokens. Computed on the first
    `step` tokens, then 2*step, 3*step, ..., up to all tokens. Plot is
    monotonically decreasing -- the longer the sample, the more
    repetition you encounter, and the lower the ratio. This is exactly
    why TTR is a bad metric to compare texts of different lengths.

    Parameters
    ----------
    input_file : str
    step : int
        Sampling stride. Smaller step = more points on the curve.
    figsize : (w, h)
    title : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Not a file: {input_file}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    with open(input_file, encoding="utf-8") as f:
        tokens = _tokenize(f.read())
    if not tokens:
        raise ValueError(f"{input_file} contains no tokens.")

    xs = list(range(step, len(tokens) + 1, step))
    if xs[-1] != len(tokens):
        xs.append(len(tokens))
    ttrs = []
    for n in xs:
        sample = tokens[:n]
        ttrs.append(len(set(sample)) / len(sample))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs, ttrs, color="#1f77b4", linewidth=2, marker="o",
            markersize=4)
    ax.set_xlabel("Sample size (tokens, first N words of the text)")
    ax.set_ylabel("TTR (types / tokens)")
    if title is None:
        title = (f"TTR vs sample size \u2014 "
                 f"{os.path.basename(input_file)} ({len(tokens):,} tokens)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, len(tokens))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public API: punctuation_density
# ---------------------------------------------------------------------------

# Categories of punctuation we report on. The character is what we
# look for in the raw text; the label is what we print.
_PUNCTUATION_GROUPS = [
    (".",   "."),
    (",",   ","),
    (";",   ";"),
    (":",   ":"),
    ("?",   "?"),
    ("!",   "!"),
    ('"',   '"'),
    ("'",   "'"),
    ("-",   "-"),
    ("(",   "("),
]


def punctuation_density(input_folder):
    """Print a per-file table of punctuation usage.

    For every .txt file in `input_folder` and for each punctuation
    type in `_PUNCTUATION_GROUPS`, prints the count and the share as
    a percentage of all non-whitespace characters. The TOTAL row
    aggregates the entire folder.
    """
    if not os.path.isdir(input_folder):
        raise NotADirectoryError(f"Not a directory: {input_folder}")

    paths = _list_txt_files(input_folder)
    if not paths:
        raise FileNotFoundError(f"No .txt files in {input_folder}")

    print(f"Punctuation density in {len(paths)} files from {input_folder}")
    print()

    rows = []
    total_chars = 0
    total_punct = Counter()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        n_chars = sum(1 for c in text if not c.isspace())
        punct_counts = {label: text.count(ch)
                        for ch, label in _PUNCTUATION_GROUPS}
        rows.append({
            "file": os.path.basename(path),
            "chars": n_chars,
            "punct": punct_counts,
        })
        total_chars += n_chars
        for label, c in punct_counts.items():
            total_punct[label] += c

    # Build the table.
    table = PrettyTable()
    field_names = [_bold_cyan("File"), _bold_cyan("Chars")]
    for _, label in _PUNCTUATION_GROUPS:
        field_names.append(_bold_cyan(f"{label} %"))
    table.field_names = field_names
    table.align[field_names[0]] = "l"
    for fn in field_names[1:]:
        table.align[fn] = "r"

    for r in rows:
        cells = [_palette_colour(r["file"], 0), f"{r['chars']:,}"]
        for i, (_, label) in enumerate(_PUNCTUATION_GROUPS):
            count = r["punct"][label]
            pct = (count / r["chars"] * 100.0) if r["chars"] else 0.0
            cells.append(_palette_colour(f"{pct:.2f}", i + 1))
        table.add_row(cells)

    table.add_row(["─" * 12] + ["─" * 8] * (len(field_names) - 1))
    total_cells = [
        f"{_ANSI_BOLD}\033[36mTOTAL{_ANSI_RESET}",
        f"{total_chars:,}",
    ]
    for i, (_, label) in enumerate(_PUNCTUATION_GROUPS):
        count = total_punct[label]
        pct = (count / total_chars * 100.0) if total_chars else 0.0
        total_cells.append(_palette_colour(f"{pct:.2f}", i + 1))
    table.add_row(total_cells)

    print(table)
