import pandas as pd
import numpy as np
import tqdm
import re
from pathlib import Path
from collections import defaultdict

def tfidf (documents: list[list[str]]) -> pd.DataFrame:
    """
    Compute TF-IDF scores for a list of tokenized documents.
 
    Args:
        documents: List of documents, where each document is a list of words.
                   Example: [["cat", "sat", "mat"], ["cat", "hat"]]
 
    Returns:
        A DataFrame with documents as rows and unique words as columns,
        where each cell contains the TF-IDF score.
    """
    # --- Term Frequency (TF) ---
    # TF(t, d) = count of term t in document d / total terms in document d
    tf_data = []
    doc_names = []
    for name, doc in documents:
        total_terms = len (doc)
        term_counts = pd.Series (doc).value_counts ()
        tf_data.append (term_counts / total_terms)
        doc_names.append (name)
 
    tf = pd.DataFrame (tf_data).fillna (0)
 
    # --- Inverse Document Frequency (IDF) ---
    # IDF(t) = log((1 + N) / (1 + df(t))) + 1  (sklearn-style smoothing)
    # where N = number of documents, df(t) = number of documents containing term t
    N = len (documents)
    df = (tf > 0).sum (axis = 0)  # document frequency per term
    idf = np.log((1 + N) / (1 + df))
 
    # --- TF-IDF ---
    tfidf = tf * idf
 
    # Rename index to reflect document numbers
    tfidf.index = [doc_names[i] for i in range(N)]
 
    return tfidf

def _tokenize(text, drop_names = False):
    
    _TOKEN_KEEP = (
        r"a-zA-Z0-9"
        # r"'\u2019"
    )
    _NON_TOKEN_CHAR = re.compile(rf"[^{_TOKEN_KEEP}]")
    words = []
    for raw in text.split():
        if drop_names and raw[0].isupper ():
            continue
        if "'" in raw:
            raw = raw.split ("'")[0]
        raw = raw.lower ()
        cleaned = _NON_TOKEN_CHAR.sub("", raw)
        if cleaned:
            words.append(cleaned)
    return words

def dir2docs (path, drop_names = False):
    documents = []
    path = Path (path)
    if not path.is_dir ():
        raise ValueError (f'Not a directory: {path.absolute ()}')
    files = list (path.glob ('*.txt'))
    for fpath in tqdm.tqdm (files):
        with open (fpath, encoding = 'utf8') as fin:
            text = fin.read ()
            documents.append ((fpath.name, _tokenize (text, drop_names = drop_names)))
    
    return documents

def top_terms_matrix (tf_idf_matrix, n: int = 20, show_numbers = False):
    """
    Return a matrix of the top-n most relevant terms per document,
    formatted as "word (score)" strings.
 
    Args:
        documents: List of documents, where each document is a list of words.
        n:         Number of top terms to return per document (default 100).
 
    Returns:
        A DataFrame with shape (num_docs, n).
        Rows are documents, columns are ranked positions (rank_1 … rank_n).
        Each cell contains "word (score)" or "" if the document has fewer
        than n unique terms.
    """
 
    rows = []
    for doc_id, scores in tf_idf_matrix.iterrows ():
        # Keep only terms that actually appear in this document, sort descending
        top = (
            scores[scores > 0]
            .sort_values (ascending = False)
            .head (n)
        )
        # Format each entry as "word (score)"
        if show_numbers:
            formatted = [f"{word} ({score:.4f})" for word, score in top.items ()]
        else:
            formatted = [f"{word}" for word, score in top.items ()]
        # Pad with empty strings if the doc has fewer than n unique terms
        formatted += [""] * (n - len (formatted))
        rows.append (formatted)
 
    columns = [f"rank_{i+1}" for i in range (n)]
    return pd.DataFrame (rows, index = tf_idf_matrix.index, columns = columns)

def get_corpus_frequency_list (input, wordlist = False, drop_names = False):
    freq = defaultdict (int)
    if wordlist:
        with open (input, encoding = 'utf8') as fin:
            fin.readline ()
            for line in fin:
                wordform, f = line.strip ('\n').split ('\t')
                freq[wordform] = int (f)
    else:
        try:
            p = Path (input)
            if p.is_dir ():
                files = list (p.glob ('*.txt'))
                iterable = tqdm.tqdm (files)
            else:
                files = [p]
                iterable = files
        except TypeError:
            if type (input) == list:
                iterable = input
            else:
                raise ValueError ('Wrong input')
        for path in iterable:
            with open (path, encoding = 'utf8') as fin:
                words = _tokenize (fin.read (), drop_names = drop_names)
                for word in words:
                    freq[word] += 1
    return freq

def get_divided_corpora (input, drop_names = False, list_a = None, list_b = None):
    p = Path (input)
    if not p.is_dir ():
        raise ValueError (f'Not a directory: {p.absolute ()}')
    files = list (p.glob ('*.txt'))
    files_a = []
    files_b = []
    if list_a:
        files_a = [f for f in files if f in list_a or f.name in list_a]
    if list_b:
        files_b = [f for f in files if f in list_b or f.name in list_b]
    freq_a = get_corpus_frequency_list (files_a, drop_names = drop_names)
    freq_b = get_corpus_frequency_list (files_b, drop_names = drop_names)

    return freq_a, freq_b
    

def calculate_keyness(
    target: dict[str, int],
    reference: dict[str, int],
    top_n: int | None = None,
    min_freq: int = 1,
) -> list[tuple[str, float]]:
    """
    Calculate keyness for words in *target* against *reference* using
    Log-Likelihood (G²).
 
    Parameters
    ----------
    target    : word → frequency for the corpus under study
    reference : word → frequency for the baseline corpus
    top_n     : return only the N highest-LL items (None = all)
    min_freq  : skip words with target frequency below this threshold
 
    Returns
    -------
    List of (word, log_likelihood) tuples, sorted by LL descending.
    Positive LL means over-represented in target; the raw score is always ≥ 0.
    """
    target_total = sum(target.values())
    ref_total = sum(reference.values())
 
    if target_total == 0:
        raise ValueError("Target corpus is empty.")
    if ref_total == 0:
        raise ValueError("Reference corpus is empty.")
 
    n = target_total + ref_total
    results = []
 
    for word, a in target.items():
        if a < min_freq:
            continue
 
        b = reference.get(word, 0)
 
        # Expected frequencies from marginal totals
        e1 = target_total * (a + b) / n
        e2 = ref_total    * (a + b) / n
 
        ll = 2.0 * (
            (a * np.log(a / e1) if a else 0.0) +
            (b * np.log(b / e2) if b else 0.0)
        )
 
        results.append((word, round(ll, 4)))
 
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n] if top_n is not None else results