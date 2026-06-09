import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
from collections import defaultdict, Counter
import tqdm
import os
import random

def _tokenize(text):
    
    _TOKEN_KEEP = (
        r"a-zA-Z0-9"
        # r"'\u2019"
    )
    _NON_TOKEN_CHAR = re.compile(rf"[^{_TOKEN_KEEP}]")
    words = []
    for raw in text.lower().split():
        cleaned = _NON_TOKEN_CHAR.sub("", raw)
        if cleaned:
            words.append(cleaned)
    return words

def relative_frequency (frequency, words_total, absolute = False):
    thresholds = [(0, 'HUNDRED', 100), (5000, 'THOUSAND', 1000), (5000000, 'MILLION', 10000000), (5e+09, 'BILLION', 1e+9)]
    for t in thresholds:
        if words_total >= t[0]:
            scale, factor = t[1:]
    rfreq = frequency / words_total
    if absolute:
        precision = 1
        x = rfreq
        while x < 1:
            x *= 10
            precision += 1
        rfreq = round (rfreq, precision)
    else:
        rfreq *= factor
        # rfreq = f'{rfreq:.2f} per {scale}'
    # return f'Relative frequency: {rfreq}'
    return rfreq

def get_corpus_frequency_list (input, wordlist = False):
    freq = defaultdict (int)
    if wordlist:
        with open (input, encoding = 'utf8') as fin:
            fin.readline ()
            for line in fin:
                wordform, f = line.strip ('\n').split ('\t')
                freq[wordform] = int (f)
    else:
        p = Path (input)
        if p.is_dir ():
            files = list (p.glob ('*.txt'))
            iterable = tqdm.tqdm (files)
        else:
            files = [p]
            iterable = files
        for path in iterable:
            with open (path, encoding = 'utf8') as fin:
                words = _tokenize (fin.read ())
                for word in words:
                    freq[word] += 1
    return freq

def get_total_counts (dct):
    return sum (dct.values ())

def plot_word_comparison (target_data, reference_data = None):
    """
    Plots a diverging lollipop chart comparing target frequencies to a reference.
    target_data: dict {word: frequency}
    reference_data: dict {word: frequency} (optional)
    """
    words = list (target_data.keys ())
    target_vals = np.array (list (target_data.values ()))
    
    # Setup figure
    fig, ax = plt.subplots (figsize = (10, 6), dpi = 100)
    
    # Normalize/Prepare data
    y_pos = np.arange (len (words))
    
    if reference_data:
        ref_vals = np.array ([reference_data.get (w, 0) for w in words])
        diff = target_vals - ref_vals
        colors = ['#e74c3c' if d > 0 else '#3498db' for d in diff]
    else:
        ref_vals = np.zeros (len (words))
        diff = target_vals
        colors = ['#2ecc71'] * len (words)

    # Draw vertical lines (stems)
    ax.hlines (y = y_pos, xmin = np.minimum (target_vals, ref_vals), 
               xmax = np.maximum (target_vals, ref_vals), color = 'grey', alpha = 0.3)
    
    # Draw lollipops
    ax.scatter (ref_vals, y_pos, color = 'grey', alpha = 0.6, label = 'Reference', zorder = 2)
    ax.scatter (target_vals, y_pos, color = colors, s = 100, label = 'Target', zorder = 3)
    
    # Styling
    ax.set_yticks (y_pos)
    ax.set_yticklabels (words, fontweight = 'bold')
    ax.set_title ('Word Frequency Comparison: Target vs Reference', pad = 20, fontsize = 14)
    ax.set_xlabel ('Relative Frequency')
    ax.spines [['top', 'right']].set_visible (False)
    ax.grid (axis = 'x', linestyle = '--', alpha = 0.5)
    
    plt.tight_layout ()
    plt.show ()

def plot_word_frequency (target_data, reference_data = None):
    """
    Plots either a simple bar chart (for target only) or 
    a diverging lollipop chart (for target vs reference).
    """
    # Sort data by frequency for readability
    sorted_items = sorted (target_data.items (), key = lambda x: x[1])
    words = [x[0] for x in sorted_items]
    target_vals = np.array ([x[1] for x in sorted_items])
    
    fig, ax = plt.subplots (figsize = (10, 6), dpi = 100)
    
    if reference_data is None:
        # Standard sorted bar chart for target only
        ax.barh (words, target_vals, color = '#3498db', alpha = 0.8, height = .5)
        ax.set_title ('Word Relative Frequency', fontsize = 14, pad = 15)
    else:
        # Diverging lollipop for comparison
        ref_vals = np.array ([reference_data.get (w, 0) for w in words])
        y_pos = np.arange (len (words))
        
        ax.hlines (y = y_pos, xmin = np.minimum (target_vals, ref_vals), 
                   xmax = np.maximum (target_vals, ref_vals), color = 'grey', alpha = 0.3)
        ax.scatter (ref_vals, y_pos, color = 'grey', alpha = 0.6, label = 'Reference', zorder = 2)
        ax.scatter (target_vals, y_pos, color = '#e74c3c', s = 100, label = 'Target', zorder = 3)
        ax.set_yticks (y_pos)
        ax.set_yticklabels (words, fontweight = 'bold')
        ax.legend ()
        ax.set_title ('Comparison: Target vs Reference', fontsize = 14, pad = 15)

    ax.set_xlabel ('Relative Frequency')
    ax.spines [['top', 'right']].set_visible (False)
    ax.grid (axis = 'x', linestyle = '--', alpha = 0.5)
    
    plt.tight_layout ()
    plt.show ()

def zipf_plot(input, top_n=None, figsize=(9, 6), title=None, loglog = True):
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
    if type (input) == str:
        input_file = input
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"Not a file: {input_file}")

        with open(input_file, encoding="utf-8") as f:
            tokens = _tokenize(f.read())
        if not tokens:
            raise ValueError(f"{input_file} contains no tokens.")

        counts = Counter(tokens)
        token_count = len (tokens)
    else:
        counts = input
        token_count = sum (counts.values ())
    sorted_freqs = sorted(counts.values(), reverse=True)
    if top_n:
        sorted_freqs = sorted_freqs[:top_n]
    ranks = list(range(1, len(sorted_freqs) + 1))

    fig, ax = plt.subplots(figsize=figsize)
    if loglog:
        ax.loglog(ranks, sorted_freqs, "o", markersize=3, color="#1f77b4",
              alpha=0.6)
    else:
        ax.plot (ranks, sorted_freqs, "o", markersize = 3, color = "#1f77b4")

    # Reference Zipf line: f(r) = f(1) / r
    if sorted_freqs:
        ref = [sorted_freqs[0] / r for r in ranks]
        if loglog:
            ax.loglog(ranks, ref, "--", color="#d62728", linewidth=1.5,
                    label="ideal Zipf (slope = -1)")

    ax.set_xlabel(f"Word rank{' (log)' if loglog else ''}")
    ax.set_ylabel(f"Frequency{' (log)' if loglog else ''}")
    if title is None:
        if type (input) == str:
            target = os.path.basename(input_file)
        else:
            target = 'dataset'
        title = (f"Zipf's law in {target} "
                 f"({token_count:,} tokens, {len(counts):,} types)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    return fig

def tokenize_all (input_path, randomize = False):
    path = Path (input_path)
    if path.is_file ():
        files = [path]
    elif path.is_dir ():
        files = list (path.glob ('*.txt'))
    tokens = []
    if len (files) > 1:
        iterable = tqdm.tqdm (files)
    else:
        iterable = files
    for input_file in iterable:
        with open(input_file, encoding="utf-8") as f:
            tokens += _tokenize(f.read())
    if randomize:
        random.shuffle (tokens)
    return tokens

def heaps_plot (tokens, start_pct = 1):

    tokens_count = []
    vocabulary_size = []
    seen_lemmas = set ()

    for i, lemma in enumerate (tokens, 1):
        seen_lemmas.add (lemma)
        tokens_count.append (i)
        vocabulary_size.append (len (seen_lemmas))

    M_data = np.array (tokens_count)
    V_data = np.array (vocabulary_size)

    end_idx = int (len (M_data) * start_pct)
    M_sliced = M_data[:end_idx]
    V_sliced = V_data[:end_idx]

    plt.figure (figsize = (10, 6))
    plt.plot (M_sliced, V_sliced, 
              color = 'red', linewidth = 1)
    plt.ticklabel_format (style = 'plain', axis = 'both')
    plt.title ('Heaps\'s Law', fontsize = 14)
    plt.xlabel ('Number of tokens', fontsize = 12)
    plt.ylabel ('Number of types', fontsize = 12)
    plt.grid (True, linestyle = ':', alpha = 0.6)
    plt.legend (fontsize = 11)
    plt.show ()


