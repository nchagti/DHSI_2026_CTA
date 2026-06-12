"""
Co-occurrence toolbox.

Tools for studying which words tend to appear together in a text corpus.

The input is a **plain text** file. The toolbox
splits text into sentences (default: at `.`, `!`, `?`; extra terminators
can be passed in) and tokenises by whitespace. A co-occurrence window of
`window_size` tokens never crosses a sentence boundary — if the target
word is near the end of its sentence, the window is truncated.

Public functions
----------------
    collocations_table(corpus, words, ...)  -> PrettyTable(s)
    collocations_graph(corpus, words, ...)  -> matplotlib Figure
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from prettytable import PrettyTable


# ---------------------------------------------------------------------------
# English stop-word list
# ---------------------------------------------------------------------------
# Kept inline so the module is self-contained (no nltk download).
# Standard "small" English list — overlap of nltk / sklearn / spaCy.

ENGLISH_STOPWORDS = frozenset("""
a about above after again against all am an and any are aren as at be
because been before being below between both but by can cannot could
couldn did didn do does doesn doing don down during each few for from
further had hadn has hasn have haven having he her here hers herself
him himself his how i if in into is isn it its itself just let ll me
mightn more most mustn my myself needn no nor not now of off on once
only or other ought our ours ourselves out over own re same shan she
should shouldn so some such than that the their theirs them themselves
then there these they this those through to too under until up ve very
was wasn we were weren what when where which while who whom why will
with won would wouldn you your yours yourself yourselves
""".split())


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SENTENCE_ENDS = (".", "!", "?")


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _build_sentence_split_re(extra_terminators=()):
    """
    Build a sentence-splitting regex that fires after `.`, `!`, `?`
    (and any extras passed in), when followed by whitespace.

    Edge cases like "Mr.", "Dr.", "U.S.A." are a known limitation
    of any pure-regex splitter; for pedagogy on novels and articles
    it's good enough.
    """
    terms = list(DEFAULT_SENTENCE_ENDS) + list(extra_terminators)
    # Deduplicate while preserving order.
    seen = set()
    terms = [t for t in terms if not (t in seen or seen.add(t))]
    char_class = "".join(re.escape(t) for t in terms)
    return re.compile(rf"(?<=[{char_class}])\s+")


# Token = one or more letters (Latin + extended Latin), optionally with
# digits or hyphens inside (covers "state-of-the-art", "covid-19").
# Pure digit runs are also kept.
_TOKEN_RE = re.compile(r"[A-Za-z\u00C0-\u024F][A-Za-z\u00C0-\u024F0-9\-]*"
                       r"|[0-9]+")


def _tokenize_sentence(sentence: str, lowercase: bool = True) -> list[str]:
    """Tokenise one sentence into a flat list of tokens."""
    toks = _TOKEN_RE.findall(sentence)
    if lowercase:
        toks = [t.lower() for t in toks]
    return toks


def _split_sentences(text: str, sentence_split_re) -> list[str]:
    """Split a text blob into sentence strings."""
    # Collapse all whitespace to single spaces — paragraph breaks
    # become sentence-internal spaces, real boundaries still fire on
    # the terminator characters.
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return sentence_split_re.split(text)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def _looks_like_path(s: str) -> bool:
    """
    Heuristic: a string is treated as a filesystem path if it's short
    enough to be a path and doesn't contain newlines or runs of more
    than one whitespace (which suggest prose).

    `collocations_table(text)` should work whether `text` is a path or
    a raw text blob — this function decides which one.
    """
    if len(s) > 4000:
        return False
    if "\n" in s:
        return False
    # Multiple words separated by spaces strongly suggests prose.
    # A path can have at most one space (e.g. "my folder/file.txt"),
    # but the body of a sentence has many.
    if s.strip().count(" ") > 3:
        return False
    return True


def _resolve_path(s):
    """
    Expand `~` and environment variables in a path string, then return
    a `pathlib.Path`. Works cross-platform: `~` expands to the user
    home on Linux/macOS and `%USERPROFILE%` on Windows; environment
    variables like `$HOME` (Unix) or `%FOO%` (Windows) are also expanded.
    """
    import os
    return Path(os.path.expandvars(os.path.expanduser(s)))


def _load_corpus_sentences(corpus,
                           lowercase: bool = True,
                           extra_sentence_terminators=()
                           ) -> list[list[str]]:
    """
    Resolve `corpus` to a list of tokenised sentences.

    `corpus` may be:
      - a path to a .txt file (with `~` or env vars; expanded)
      - a path to a directory (we read *.txt non-recursively)
      - a string containing the text itself
    """
    sent_split_re = _build_sentence_split_re(extra_sentence_terminators)

    # If `corpus` is a plain string, decide whether it looks like a path
    # or like raw text. Long strings or strings with newlines are text.
    if isinstance(corpus, str):
        if _looks_like_path(corpus):
            p = _resolve_path(corpus)
            if p.exists():
                path = p
            else:
                raise FileNotFoundError(f"Corpus path not found: {corpus!r} "
                                        f"(expanded to {p})")
        else:
            # Treat as raw text.
            sentences = _split_sentences(corpus, sent_split_re)
            return [_tokenize_sentence(s, lowercase) for s in sentences]
    else:
        # pathlib.Path or similar — also expand if it's still a string-like.
        path = _resolve_path(str(corpus))
        if not path.exists():
            raise FileNotFoundError(f"Corpus path not found: {path}")

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt files in directory: {path}")
    else:
        raise FileNotFoundError(f"Corpus path not found: {path}")

    all_sents = []
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        for sent in _split_sentences(text, sent_split_re):
            toks = _tokenize_sentence(sent, lowercase)
            if toks:
                all_sents.append(toks)
    return all_sents


# ---------------------------------------------------------------------------
# Counting: unigram counts + windowed co-occurrence counts
# ---------------------------------------------------------------------------

def _count(sentences: list[list[str]],
           targets: set[str],
           window_size: int,
           stop_words: frozenset,
           filter_stop_words: bool):
    """
    Walk the corpus once and accumulate:
      - N: total token count
      - unigram[w]: frequency of every word
      - target_count[t]: how many times each target appeared
      - cooc[t][w]: how many times w appeared in t's window

    Window is sentence-bounded: `_count` iterates one sentence at a
    time, so the window is naturally truncated near sentence ends.
    """
    N = 0
    unigram = Counter()
    target_count = Counter()
    cooc = defaultdict(Counter)

    for sent in sentences:
        # Unigram counts and N are taken from the ORIGINAL token
        # sequence, regardless of stop-word filtering. The marginals
        # in the association measures must reflect the actual corpus.
        N += len(sent)
        unigram.update(sent)

        # Scan for targets and their windows.
        for i, tok in enumerate(sent):
            if tok not in targets:
                continue
            target_count[tok] += 1
            lo = max(0, i - window_size)
            hi = min(len(sent), i + window_size + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                neighbour = sent[j]
                if filter_stop_words and neighbour in stop_words:
                    continue
                cooc[tok][neighbour] += 1

    return N, unigram, target_count, cooc


# ---------------------------------------------------------------------------
# Association measures
# ---------------------------------------------------------------------------
#
# Conventions:
#   c_t = total target occurrences        (target_count[t])
#   c_w = total neighbour-word occurrences (unigram[w])
#   k   = how often w appeared in t's window (cooc[t][w])
#   N   = total tokens in corpus
#
# Expected count under independence:  E = c_t * c_w / N
#
# PMI       = log2(k / E)
# T-score   = (k - E) / sqrt(k)
# LLR / G²  = 2 * Σ Oij * ln(Oij / Eij)    [Dunning 1993, 2x2 table]

def _pmi(k, c_t, c_w, N):
    if k == 0 or c_t == 0 or c_w == 0 or N == 0:
        return float("-inf")
    expected = c_t * c_w / N
    return math.log2(k / expected) if expected > 0 else float("-inf")


def _tscore(k, c_t, c_w, N):
    if k == 0 or N == 0:
        return float("-inf")
    expected = c_t * c_w / N
    return (k - expected) / math.sqrt(k)

def _zscore (k, c_t, c_w, N):
    expected = c_t * c_w / N
    return (k - expected) / math.sqrt (expected)

def _dice (k, c_t, c_w, N):
    return 2 * k / (c_t + c_w)


def _llr_2x2(k, c_t, c_w, N):
    """
    Dunning's log-likelihood ratio for a 2x2 contingency table.

    Counts:
       O11 = k                 (target & neighbour together)
       O12 = c_t - k           (target without this neighbour)
       O21 = c_w - k           (neighbour without target)
       O22 = N - c_t - c_w + k (neither)

    G² = 2 * Σ Oij * ln(Oij / Eij), with Eij from row/col marginals.
    Convention: 0 * ln(0) = 0.
    """
    if k == 0:
        return 0.0
    O11 = k
    O12 = c_t - k
    O21 = c_w - k
    O22 = N - c_t - c_w + k

    if O12 < 0 or O21 < 0 or O22 < 0:
        return 0.0

    row1 = O11 + O12
    row2 = O21 + O22
    col1 = O11 + O21
    col2 = O12 + O22
    total = row1 + row2
    if total <= 0:
        return 0.0

    E11 = row1 * col1 / total
    E12 = row1 * col2 / total
    E21 = row2 * col1 / total
    E22 = row2 * col2 / total

    def _term(O, E):
        if O <= 0 or E <= 0:
            return 0.0
        return O * math.log(O / E)

    return 2.0 * (_term(O11, E11) + _term(O12, E12)
                   + _term(O21, E21) + _term(O22, E22))


_MEASURE_FNS = {
    "pmi":             ("PMI",     _pmi),
    "t":               ("T-score", _tscore),
    "t-score":         ("T-score", _tscore),
    "tscore":          ("T-score", _tscore),
    "llr":             ("LLR",     _llr_2x2),
    "loglikelihood":   ("LLR",     _llr_2x2),
    "log-likelihood":  ("LLR",     _llr_2x2),
    "dice":            ("Dice",    _dice),
    "z":               ("Z-score", _zscore),
    "z-score":          ("Z-score", _zscore),
    "zscore":          ("Z-score", _zscore)
}


def _normalize_measures(measures):
    """Accept a string or list, return list of (display_name, fn)."""
    if isinstance(measures, str):
        measures = [measures]
    out = []
    seen = set()
    for m in measures:
        key = m.lower()
        if key not in _MEASURE_FNS:
            print (_MEASURE_FNS)
            raise ValueError(
                f"Unknown collocation measure {m!r}. "
                f"Known: pmi, t-score, z-score, dice, llr.")
        display, fn = _MEASURE_FNS[key]
        if display in seen:
            continue
        seen.add(display)
        out.append((display, fn))
    return out


def _resolve_colors(words, colors):
    """
    Build a list of colours, one per word in `words`. `colors` can be:
      - None: use the tab10 palette
      - list/tuple of same length as `words`: pos i ↔ word i
      - dict {word: colour}: missing keys fall back to tab10

    Returns a list of matplotlib-friendly RGBA tuples.
    """
    from matplotlib.colors import to_rgba

    cmap = plt.get_cmap("tab10")
    if colors is None:
        return [cmap(i % 10) for i in range(len(words))]
    if isinstance(colors, dict):
        return [to_rgba(colors[w]) if w in colors else cmap(i % 10)
                for i, w in enumerate(words)]
    if isinstance(colors, (list, tuple)):
        if len(colors) != len(words):
            raise ValueError(
                f"colors length ({len(colors)}) "
                f"must equal words length ({len(words)})")
        return [to_rgba(c) for c in colors]
    raise TypeError(f"Unsupported colors type: {type(colors).__name__}")

def load_corpus (path, lowercase = True, extra_sentence_terminators = ()):
    sentences = _load_corpus_sentences(
        path,
        lowercase=lowercase,
        extra_sentence_terminators=extra_sentence_terminators,
    )
    if not sentences:
        print("Corpus is empty.")
        return {}
    return sentences


# ---------------------------------------------------------------------------
# 1. Co-occurrence table
# ---------------------------------------------------------------------------

def collocations_table(corpus,
                       words: Sequence[str],
                       window_size: int = 5,
                       top_n: int = 10,
                       filter_stop_words: bool = True,
                       collocation_measure="llr",
                       stopwords: Iterable[str] | None = None,
                       lowercase: bool = True,
                       extra_sentence_terminators: Iterable[str] = (),
                       sort_by: str | None = None,
                       min_count: int = 3):
    """
    Find collocates for each word in `words` and print a PrettyTable
    per target.

    Parameters
    ----------
    corpus : str | Path
        Path to a .txt file, path to a folder of .txt files, or a plain
        text string.
    words : list of target words.
    window_size : tokens on each side counted as "in context"
        (5 is typical; 2-3 for syntactic; 10+ for topical).
    top_n : how many collocates to show per target.
    filter_stop_words : drop stop words from the *collocate* side
        (targets are kept regardless).
    collocation_measure : "pmi" | "t-score" | "llr", or a list of these.
        First listed measure is used to rank the table; the others are
        shown as extra columns. Pass a list e.g. ["llr", "pmi", "t-score"]
        to get all three.
    stopwords : custom iterable; if None and filter_stop_words=True,
        uses the built-in English list.
    lowercase : lowercase all tokens before counting (recommended for
        English; off for case-sensitive work).
    extra_sentence_terminators : extra characters that also end sentences,
        e.g. (";", "—"). Defaults to just (".", "!", "?").
    min_count : drop collocates with co-occurrence count below this.
        Stabilises PMI in particular (which loves rare events).

    Returns
    -------
    dict {target_word: PrettyTable}
    """
    measures = _normalize_measures(collocation_measure)
    stopwords = set(list(ENGLISH_STOPWORDS) + stopwords) if stopwords else ENGLISH_STOPWORDS
    sw = (frozenset(s.lower() for s in stopwords))

    sentences = corpus

    targets_lower = {(w.lower() if lowercase else w) for w in words}
    N, unigram, target_count, cooc = _count(
        sentences, targets_lower, window_size, sw, filter_stop_words)

    # Header — what was loaded.
    print(f"Corpus:   {len(sentences):,} sentences, {N:,} tokens, "
          f"{len(unigram):,} types")
    print(f"Window:   ±{window_size} tokens, sentence-bounded")
    print(f"Stops:    {'filtered (' + str(len(sw)) + ' words)' if filter_stop_words else 'kept'}")
    print(f"Measures: {', '.join(m for m, _ in measures)} "
          f"(ranked by {measures[0][0]})")
    print(f"Min cnt:  {min_count}")
    print()

    primary_name, primary_fn = measures[0]
    result = {}

    for target in words:
        tk = target.lower() if lowercase else target
        c_t = target_count.get(tk, 0)
        if c_t == 0:
            print(f"  ! {target!r} not found in corpus, skipping.")
            print()
            continue

        # Score every neighbour by the primary measure, keep top_n.
        scored = []
        for w, k in cooc[tk].items():
            if k < min_count:
                continue
            c_w = unigram[w]
            score = primary_fn(k, c_t, c_w, N)
            scored.append((w, k, score))
        scored.sort(key=lambda r: r[2], reverse=True)
        scored = scored[:top_n]

        # Build the table.
        tbl = PrettyTable()
        cols = ["rank", "collocate", "count"] + [m for m, _ in measures]
        tbl.field_names = cols
        for c in cols:
            tbl.align[c] = "l"

        for rank, (w, k, _) in enumerate(scored, 1):
            c_w = unigram[w]
            row = [rank, w, k]
            for m_name, m_fn in measures:
                v = m_fn(k, c_t, c_w, N)
                if v == float("-inf"):
                    row.append("—")
                else:
                    row.append(f"{v:.2f}")
            tbl.add_row(row)
        print(f"=== {target!r}  (occurs {c_t:,}× in corpus) ===")
        print(tbl if not sort_by else tbl.get_string (sortby = sort_by, reversesort = True))
        print()
        result[target] = tbl

    return result


# ---------------------------------------------------------------------------
# 2. Co-occurrence graph
# ---------------------------------------------------------------------------

def collocations_graph(corpus,
                       words: Sequence[str],
                       window_size: int = 5,
                       top_n: int = 8,
                       filter_stop_words: bool = True,
                       collocation_measure: str = "llr",
                       stopwords: Iterable[str] | None = None,
                       lowercase: bool = True,
                       extra_sentence_terminators: Iterable[str] = (),
                       min_count: int = 3,
                       colors=None,
                       figsize=(11, 8),
                       seed: int = 42,
                       ax=None):
    """
    Plot the top collocates of each target word as a network graph.

    - Node SIZE is proportional to the word's corpus frequency.
    - Node COLOUR distinguishes target words from collocates: each
      target gets its own colour; collocates are grey. A collocate
      shared by multiple targets is shown once.
    - Edge THICKNESS is proportional to the chosen association score.
    - Edge COLOUR matches the target it belongs to.

    Reading the plot:
      - thick edges → strong association
      - hub targets with many edges share a topic
      - collocates with edges from multiple targets bridge concepts

    Parameters
    ----------
    Most are identical to `collocations_table`. Additional:

    colors : per-target colours. None | list | dict {word: colour}.
        - None → use the tab10 palette in order.
        - dict → {"king": "blue", "queen": "red"}; missing words fall
          back to tab10.
        - list → same length as `words`, positional.
    figsize : matplotlib figure size.
    seed : layout reproducibility.
    ax : paint onto existing axes if provided.

    Returns the matplotlib Figure.
    """
    # Use exactly one measure for the graph (no good way to render
    # three thicknesses on one edge).
    measures = _normalize_measures(collocation_measure)
    if len(measures) != 1:
        if len(measures) > 1:
            print(f"  ! graph uses one measure; using {measures[0][0]}")
    measure_name, measure_fn = measures[0]

    sw = (frozenset(s.lower() for s in stopwords)
          if stopwords is not None else ENGLISH_STOPWORDS)

    sentences = _load_corpus_sentences(
        corpus,
        lowercase=lowercase,
        extra_sentence_terminators=extra_sentence_terminators,
    )
    if not sentences:
        print("Corpus is empty.")
        return None

    targets_lower = [w.lower() if lowercase else w for w in words]
    target_set = set(targets_lower)

    N, unigram, target_count, cooc = _count(
        sentences, target_set, window_size, sw, filter_stop_words)

    # Resolve colours. We resolve against the LOWERCASED target list
    # because that's what we use as dict keys internally. The user
    # may have passed colours keyed on either original or lowercase
    # form; try both.
    user_colors = colors
    if isinstance(colors, dict):
        # Build a lowercase-keyed copy so lookup works even if user
        # passed colors keyed on the original case.
        norm_colors = {}
        for k, v in colors.items():
            nk = k.lower() if lowercase else k
            norm_colors[nk] = v
        user_colors = norm_colors
    resolved = _resolve_colors(targets_lower, user_colors)
    target_color = dict(zip(targets_lower, resolved))

    # Build the graph: targets + their top_n collocates.
    G = nx.Graph()
    edge_target = {}        # (u, v) → target this edge belongs to

    for tw in targets_lower:
        c_t = target_count.get(tw, 0)
        if c_t == 0:
            print(f"  ! {tw!r} not in corpus, skipping.")
            continue

        # Score collocates
        scored = []
        for w, k in cooc[tw].items():
            if k < min_count:
                continue
            c_w = unigram[w]
            score = measure_fn(k, c_t, c_w, N)
            if score == float("-inf"):
                continue
            scored.append((w, score))
        scored.sort(key=lambda r: r[1], reverse=True)
        scored = scored[:top_n]

        # Add target node.
        if tw not in G:
            G.add_node(tw, freq=unigram.get(tw, 0), is_target=True)

        for w, score in scored:
            if w not in G:
                G.add_node(w, freq=unigram[w],
                           is_target=(w in target_set))
            G.add_edge(tw, w, weight=score)
            edge_target[(tw, w)] = tw
            edge_target[(w, tw)] = tw

    if G.number_of_nodes() == 0:
        print("No collocates above min_count threshold — nothing to draw.")
        return None

    # Node sizes: scale by frequency.
    freqs = np.array([G.nodes[n]["freq"] for n in G.nodes])
    if freqs.max() > 0:
        node_sizes = 200 + 1500 * np.sqrt(freqs / freqs.max())
    else:
        node_sizes = np.full(len(freqs), 300)

    # Node colours: target colour for targets, grey for collocates.
    node_colors = []
    for n in G.nodes:
        if n in target_color:
            node_colors.append(target_color[n])
        else:
            node_colors.append("lightgrey")

    # Edge widths: scale by score.
    edge_data = list(G.edges(data=True))
    weights = np.array([d["weight"] for *_, d in edge_data])
    if weights.max() > 0:
        edge_widths = 0.5 + 4.0 * np.sqrt(
            np.clip(weights, 0, None) / weights.max())
    else:
        edge_widths = np.full(len(weights), 1.0)

    edge_colors = []
    for u, v, _ in edge_data:
        tw = edge_target.get((u, v)) or edge_target.get((v, u))
        if tw is not None and tw in target_color:
            edge_colors.append(target_color[tw])
        else:
            edge_colors.append("grey")

    pos = nx.spring_layout(G, seed=seed, k=1.5 / math.sqrt(len(G.nodes)))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    nx.draw_networkx_edges(G, pos,
                           edgelist=[(u, v) for u, v, _ in edge_data],
                           width=edge_widths,
                           edge_color=edge_colors,
                           alpha=0.6, ax=ax)
    nx.draw_networkx_nodes(G, pos,
                           node_size=node_sizes,
                           node_color=node_colors,
                           edgecolors="black",
                           linewidths=0.8,
                           ax=ax)

    target_labels = {n: n for n in G.nodes if n in target_color}
    other_labels = {n: n for n in G.nodes if n not in target_color}
    nx.draw_networkx_labels(G, pos, labels=other_labels,
                            font_size=9, font_color="black", ax=ax)
    nx.draw_networkx_labels(G, pos, labels=target_labels,
                            font_size=11, font_weight="bold",
                            font_color="black", ax=ax)

    ax.set_title(
        f"Collocation graph (window=±{window_size}, "
        f"measure={measure_name}, top {top_n} per target)"
    )
    ax.set_axis_off()

    # Legend.
    handles = [plt.Line2D([], [], marker="o", color="w",
                          markerfacecolor=target_color[w],
                          markeredgecolor="black",
                          markersize=10, label=w)
               for w in targets_lower if w in target_color]
    if handles:
        ax.legend(handles=handles, title="Target word", loc="best")

    fig.tight_layout()
    return fig
