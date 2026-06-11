"""
toolbox_tm.py
=============

Topic modelling toolbox for plain-text folders.

This module wraps gensim's LDA so you can go from a folder of files
to a working topic model in 4 lines.

Public API
----------
load_corpus(input_folder, exclude_stop_words=True, min_word_len=3)
    Read every .txt file in a folder, tokenise, optionally drop stop
    words, return {filename: [tokens]}.

build_dictionary(processed, min_df=2, max_df=0.8)
    Build a gensim Dictionary filtered to keep only words that appear
    in at least min_df documents and at most max_df fraction of them.

fit_lda(processed, dictionary, num_topics=10, passes=10,
        random_state=42)
    Train an LDA model. Returns a gensim LdaModel.

top_words_per_topic(lda_model, n=10)
    Print a table of the top-n words for every topic.

top_documents_per_topic(lda_model, processed, dictionary, n=3)
    Print a table of the n documents most strongly associated with
    each topic.

document_topic_heatmap(lda_model, processed, dictionary, **kwargs)
    Heatmap of how strongly each document loads on each topic.
    Returns a matplotlib Figure.

Assumed input
-------------
Plain-text files, one document per file, already lemmatised. Files
are read as UTF-8. Tokenisation is identical to toolbox_gs.py
(lowercase, strip punctuation, keep Devanagari + a few Indic scripts).
Filenames are used as document IDs throughout.
"""

import os
import re
from collections import Counter

import matplotlib.pyplot as plt
from prettytable import PrettyTable

import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

from wordcloud import WordCloud


# ---------------------------------------------------------------------------
# Tokenisation (same rules as toolbox_gs.py)
# ---------------------------------------------------------------------------

_TOKEN_KEEP = (
    r"\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF"
    r"\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF"
    r"\u0D00-\u0D7F\u0600-\u06FF\u0750-\u077F\u200C\u200D"
    r"a-zA-Z0-9'\u2019"
)
_NON_TOKEN_CHAR = re.compile(rf"[^{_TOKEN_KEEP}]")


def _tokenize(text, drop_ners = False):
    words = []
    for raw in re.split(r"[ \t\r\n']", text):
        cleaned = _NON_TOKEN_CHAR.sub("", raw)
        if cleaned:
            if not drop_ners or cleaned[0].islower():
                words.append(cleaned.lower ())
    return words


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
    "itbe", "wouldnot", "hath", "benot", "havenot", "couldnot", "youhave", "whatbe", "shouldnot"
})


# ---------------------------------------------------------------------------
# Helpers (ANSI colour, file listing)
# ---------------------------------------------------------------------------

_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"

_PALETTE = [
    "\033[33m", "\033[32m", "\033[34m", "\033[35m", "\033[36m",
    "\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m",
]


def _palette_colour(value, idx):
    return f"{_PALETTE[idx % len(_PALETTE)]}{value}{_ANSI_RESET}"


def _bold_cyan(s):
    return f"{_ANSI_BOLD}\033[36m{s}{_ANSI_RESET}"


def _list_txt_files(folder):
    paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".txt") and os.path.isfile(os.path.join(folder, f))
    ]
    paths.sort()
    return paths


# ---------------------------------------------------------------------------
# Public API: load_corpus
# ---------------------------------------------------------------------------

def load_corpus(input_folder, exclude_stop_words=True, min_word_len=3, drop_ners = True, custom_stopwords = [], chunk_size = None):
    """Read and tokenise every .txt file in a folder.

    Parameters
    ----------
    input_folder : str
        Folder of .txt files. Filenames are used as document IDs.
    exclude_stop_words : bool
        If True, drop the standard English stop word list (~180 words)
        from every document. Default True -- you almost never want
        stop words in topic models.
    min_word_len : int
        Drop tokens shorter than this. Default 3, which removes
        residual one- and two-letter tokens that often slip through
        lemmatisation (e.g. "i", "m", "ll", remnants of "I'm",
        "I'll"). Set to 1 to keep everything.

    Returns
    -------
    dict
        {filename: [tokens]} in alphabetical filename order.
    """
    if not os.path.isdir(input_folder):
        raise NotADirectoryError(f"Not a directory: {input_folder}")

    paths = _list_txt_files(input_folder)
    if not paths:
        raise FileNotFoundError(f"No .txt files in {input_folder}")

    processed = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            tokens = _tokenize(f.read(), drop_ners = drop_ners)
        if exclude_stop_words:
            stopwords = list (_ENGLISH_STOP_WORDS) + custom_stopwords
            tokens = [t for t in tokens if t not in stopwords]
        if min_word_len > 1:
            tokens = [t for t in tokens if len(t) >= min_word_len]
        if chunk_size:
            for i in range (len (tokens) // chunk_size):
                chunk = tokens[i * chunk_size : (i + 1) * chunk_size]
                processed[f'{os.path.basename (p)}_{i + 1}'] = chunk
        else:
            processed[os.path.basename(p)] = tokens

    total = sum(len(t) for t in processed.values())
    print(f"Loaded {len(processed)} documents, {total:,} tokens total.")
    return processed


# ---------------------------------------------------------------------------
# Public API: build_dictionary
# ---------------------------------------------------------------------------

def build_dictionary(processed, min_df=2, max_df=0.8):
    """Build a gensim Dictionary from a preprocessed corpus.

    Words too rare or too frequent across documents are dropped --
    they don't carry topic information.

    Parameters
    ----------
    processed : dict
        Output of load_corpus.
    min_df : int
        Drop words appearing in fewer than this many documents.
        Default 2 -- a word that appears in only one document can't
        connect topics across the corpus.
    max_df : float (0 to 1)
        Drop words appearing in more than this fraction of documents.
        Default 0.8 -- words appearing nearly everywhere don't
        distinguish topics.

    Returns
    -------
    gensim.corpora.Dictionary
    """
    if not processed:
        raise ValueError("processed is empty -- did load_corpus run?")

    documents = list(processed.values())
    n_docs = len(documents)

    dictionary = corpora.Dictionary(documents)
    size_before = len(dictionary)

    dictionary.filter_extremes(no_below=min_df, no_above=max_df,
                               keep_n=None)
    size_after = len(dictionary)

    print(f"Vocabulary: {size_before:,} unique words -> "
          f"{size_after:,} after filtering")
    print(f"  Kept words appearing in at least {min_df} document(s) "
          f"and at most {max_df:.0%} of {n_docs} documents.")
    return dictionary


# ---------------------------------------------------------------------------
# Public API: fit_lda
# ---------------------------------------------------------------------------

def fit_lda(processed, dictionary, num_topics=10, passes=10,
            random_state=None, alpha = 'auto'):
    """Train an LDA topic model.

    Parameters
    ----------
    processed : dict
        Output of load_corpus.
    dictionary : gensim.corpora.Dictionary
        Output of build_dictionary.
    num_topics : int
        How many topics to discover. There's no "correct" value --
        more topics = finer-grained but harder to interpret. Typical
        range for a literary corpus is 5 to 30.
    passes : int
        Number of passes over the corpus during training. More =
        better fit but slower. 10 is a reasonable default.
    random_state : int
        For reproducibility. LDA is stochastic, so without a fixed
        seed you get different topics each run.

    Returns
    -------
    gensim.models.LdaModel
    """
    if not processed:
        raise ValueError("processed is empty.")
    if num_topics < 2:
        raise ValueError(f"num_topics must be >= 2, got {num_topics}")

    # Convert each document to its bag-of-words vector.
    documents = list(processed.values())
    bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
    alpha_info = f'alpha: {alpha}' if alpha != 'auto' else ''
    print(f"Training LDA: num_topics={num_topics}, passes={passes} {alpha_info}...")
    lda = LdaModel(
        corpus=bow_corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=random_state,
        alpha=alpha,
        per_word_topics=False,
    )
    print(f"Done. Model has {num_topics} topics over a vocabulary of "
          f"{len(dictionary):,} words.")

    # Attach the bow corpus and filename order to the model so the
    # other helpers can find them.
    lda._processed_filenames = list(processed.keys())
    lda._bow_corpus = bow_corpus
    return lda


# ---------------------------------------------------------------------------
# Public API: top_words_per_topic
# ---------------------------------------------------------------------------

def top_words_per_topic(lda_model, n=10):
    """Print the top-n words of every topic, with their weights.

    Each topic is a probability distribution over the vocabulary;
    higher weight = the word is more characteristic of that topic.
    """
    num_topics = lda_model.num_topics

    table = PrettyTable()
    table.field_names = [_bold_cyan("Topic")] + [
        _bold_cyan(f"#{i + 1}") for i in range(n)
    ]
    table.align[_bold_cyan("Topic")] = "l"
    for i in range(n):
        table.align[_bold_cyan(f"#{i + 1}")] = "l"

    for topic_id in range(num_topics):
        terms = lda_model.show_topic(topic_id, topn=n)
        cells = [_palette_colour(f"Topic {topic_id}", topic_id)]
        for word, weight in terms:
            cells.append(f"{word} ({weight:.3f})")
        # Pad if fewer than n terms came back (very small vocab).
        while len(cells) < n + 1:
            cells.append("")
        table.add_row(cells)

    print(table)


# ---------------------------------------------------------------------------
# Public API: top_documents_per_topic
# ---------------------------------------------------------------------------

def top_documents_per_topic(lda_model, processed, dictionary, n=3):
    """Print the documents most strongly associated with each topic.

    For every topic, we score each document by how much of it loads
    on that topic (LDA gives a per-document distribution over topics)
    and report the n highest-scoring documents, with their topic
    share as a percentage.
    """
    filenames = list(processed.keys())
    bow_corpus = [dictionary.doc2bow(processed[fn]) for fn in filenames]
    num_topics = lda_model.num_topics

    # doc_topics[i] = list of (topic_id, weight) for the i-th doc.
    doc_topics = [
        lda_model.get_document_topics(bow, minimum_probability=0.0)
        for bow in bow_corpus
    ]

    # Build a topic_id -> [(doc_index, weight), ...] mapping.
    by_topic = {t: [] for t in range(num_topics)}
    for doc_idx, topics in enumerate(doc_topics):
        for topic_id, weight in topics:
            by_topic[topic_id].append((doc_idx, weight))

    table = PrettyTable()
    table.field_names = [_bold_cyan("Topic")] + [
        _bold_cyan(f"Doc {i + 1}") for i in range(n)
    ]
    table.align[_bold_cyan("Topic")] = "l"
    for i in range(n):
        table.align[_bold_cyan(f"Doc {i + 1}")] = "l"

    for topic_id in range(num_topics):
        ranked = sorted(by_topic[topic_id], key=lambda x: -x[1])[:n]
        cells = [_palette_colour(f"Topic {topic_id}", topic_id)]
        for doc_idx, weight in ranked:
            label = f"{filenames[doc_idx]} ({weight * 100:.1f}%)"
            cells.append(label)
        while len(cells) < n + 1:
            cells.append("")
        table.add_row(cells)

    print(table)


# ---------------------------------------------------------------------------
# Public API: document_topic_heatmap
# ---------------------------------------------------------------------------

def document_topic_heatmap(
    lda_model,
    processed,
    dictionary,
    cmap="YlOrRd",
    figsize=None,
    annotate=True,
    title=None,
):
    """Heatmap of how strongly each document loads on each topic.

    Rows are documents (in filename order), columns are topics. Each
    cell shows the topic's share of that document as a percentage.

    Parameters
    ----------
    lda_model : gensim LdaModel
    processed : dict
    dictionary : gensim Dictionary
    cmap : str
        Matplotlib colormap. Default "YlOrRd".
    figsize : (w, h) or None
        Auto-scales by default.
    annotate : bool
        If True, write percentages inside cells.
    title : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    filenames = list(processed.keys())
    n_docs = len(filenames)
    num_topics = lda_model.num_topics

    matrix = []
    for fn in filenames:
        bow = dictionary.doc2bow(processed[fn])
        topics = dict(lda_model.get_document_topics(
            bow, minimum_probability=0.0))
        row = [topics.get(t, 0.0) * 100.0 for t in range(num_topics)]
        matrix.append(row)

    if figsize is None:
        figsize = (max(8, num_topics * 0.7),
                   max(4, n_docs * 0.4))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)

    ax.set_xticks(range(num_topics))
    ax.set_xticklabels([f"T{i}" for i in range(num_topics)],
                       fontsize=10)
    ax.set_yticks(range(n_docs))
    ax.set_yticklabels(filenames, fontsize=9)
    ax.set_xlabel("Topic")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Share of document (%)", fontsize=9)

    if annotate:
        max_val = max((max(row) for row in matrix), default=0)
        threshold = max_val * 0.55 if max_val > 0 else 0
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                colour = "white" if val > threshold else "black"
                if val >= 1:
                    label = f"{val:.0f}"
                elif val > 0:
                    label = "·"
                else:
                    label = ""
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=8, color=colour)

    if title is None:
        title = (f"Document \u00d7 topic distribution "
                 f"({n_docs} docs \u00d7 {num_topics} topics)")
    ax.set_title(title, fontsize=12, pad=12)

    fig.tight_layout()
    return fig


def tm_wordcloud (topic_frequencies, tnumber, colormap = 'plasma', max_words = 50, pad = 1, margin = 10):
    """Generates and displays a word cloud from a dictionary of word frequencies."""
    # Initialize the WordCloud object with custom styling
    wordcloud = WordCloud (
        width = 800,
        height = 400,
        background_color = 'white',
        max_words = max_words,
        colormap = colormap,
        margin = margin
    )

    # Generate the cloud using the word weights from the topic model
    wordcloud.generate_from_frequencies (frequencies = topic_frequencies)

    # Display the generated image using matplotlib
    plt.figure (figsize = (10, 5))
    plt.imshow (wordcloud, interpolation = 'bilinear')
    plt.axis ('off')  # Hide the axes for a cleaner look
    plt.title (f'Topic {tnumber}', fontsize = 25, pad = 20, fontweight = 'bold')
    plt.tight_layout (pad = pad)
    plt.show ()

def topic_wordclouds (model, max_words = 50, include_topics = [], colormap = 'plasma', margin = 10):
    num_topics = range (model.num_topics)
    to_show = [t for t in num_topics if t in include_topics] if include_topics else num_topics
    for topic_number in to_show:
        frequencies = dict (model.show_topic (topic_number, topn = max_words))
        tm_wordcloud (frequencies, topic_number, max_words = max_words, colormap = colormap, margin = margin)

def check_coherence (model, processed, dictionary):
    coherence_model_lda = CoherenceModel(model = model, texts = list (processed.values ()), dictionary = dictionary, coherence='c_v')
    return coherence_model_lda.get_coherence()

def find_optimal_coherence (processed, dictionary, target_range, passes = 15):
    model_list = []
    coherence_values = []
    for num_topics in target_range:
        model = fit_lda (processed, dictionary, num_topics = num_topics, passes = passes)
        model_list.append (model)
        
        # Calculate Coherence
        coherencemodel = CoherenceModel (model = model, texts = list (processed.values ()), dictionary = dictionary, coherence = 'c_v')
        coherence_values.append (coherencemodel.get_coherence ())
    
    # Plot the results to find the "elbow" or highest point
    topic_range = target_range
    plt.plot (topic_range, coherence_values, marker = 'o')
    plt.xlabel ('Number of Topics (K)')
    plt.ylabel ('Coherence Score (C_v)')
    plt.title ('Choosing Optimal Topic Count')
    plt.show ()

def generate_html (model, processed, dictionary, filename = 'lda_visualization.html'):
    documents = list(processed.values())
    bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
    vis_data = gensimvis.prepare (
        topic_model = model,      # Your trained Gensim LdaModel or LdaMulticore
        corpus = bow_corpus,              # The Bag-of-Words corpus used for training
        dictionary = dictionary,      # The Gensim Dictionary object
        sort_topics = True            # Sorts topics by total document frequency
    )

    # 2. Display the interactive chart directly inside a Jupyter Notebook
    pyLDAvis.enable_notebook ()
    pyLDAvis.display (vis_data)

    # 3. Alternative: Save the visualization as a standalone HTML file
    # This is perfect for sharing results with colleagues or embedding in a research folder
    pyLDAvis.save_html (vis_data, filename)

def get_dominant_topics (text_tokens, lda_model, dictionary, top_n = 3):
    '''
    Identifies the top N topics that have the highest cumulative probability across the entire text.
    '''
    bow = dictionary.doc2bow (text_tokens)
    topic_distribution = lda_model.get_document_topics (bow, minimum_probability = 0.0)
    
    # Sort topics by probability in descending order and grab the top N
    sorted_topics = sorted (topic_distribution, key = lambda x: x[1], reverse = True)
    top_topics_ids = [topic[0] for topic in sorted_topics[:top_n]]
    
    return top_topics_ids


def rolling_topic_distribution (text_tokens, lda_model, dictionary, target_topics, no_steps = 10, overlap_ratio = 0.5):
    '''
    Slices a single tokenized text into overlapping windows and tracks the trajectory of target topics.
    '''
    total_tokens = len (text_tokens)
    
    # Determine window size based on steps and desired overlap
    # To cover the text in no_steps with a given overlap:
    # total_tokens = window_size + (no_steps - 1) * step_size
    # where step_size = window_size * (1 - overlap_ratio)
    step_size = int (total_tokens / (no_steps - (no_steps - 1) * overlap_ratio))
    window_size = int (step_size / (1 - overlap_ratio)) if overlap_ratio < 1 else step_size

    # Fallback safety for shorter texts or edge cases
    if window_size >= total_tokens or window_size == 0:
        window_size = total_tokens // 2
        step_size = window_size // 2

    # Initialize a data structure to store trajectories: {topic_id: [prob_step1, prob_step2, ...]}
    trajectories = {topic_id: [] for topic_id in target_topics}
    x_labels = []

    for step_idx in range (no_steps):
        start_idx = step_idx * step_size
        end_idx = start_idx + window_size
        
        # Ensure we don't overshoot the text length drastically
        chunk = text_tokens[start_idx:end_idx]
        if not chunk:
            break
            
        chunk_bow = dictionary.doc2bow (chunk)
        chunk_topics = dict (lda_model.get_document_topics (chunk_bow, minimum_probability = 0.0))
        
        # Record the probability for each of our primary topics
        for topic_id in target_topics:
            trajectories[topic_id].append (chunk_topics.get (topic_id, 0.0))
            
        # Create a label representing the narrative progress percentage
        progress_pct = int ((start_idx + (len (chunk) / 2)) / total_tokens * 180)  # midpoint representation
        x_labels.append (f'{progress_pct}%')

    return trajectories, x_labels


def plot_topic_trajectory (trajectories, x_labels, lda_model, name, num_words = 3):
    '''
    Visualizes the sequential changes of the selected topics across the text timeline.
    '''
    plt.figure (figsize = (11, 5))
    
    for topic_id, probabilities in trajectories.items ():
        # Fetch top words for the legend label to make it human-readable
        topic_terms = lda_model.show_topic (topic_id, topn = num_words)
        topic_label = f'Topic {topic_id}: ' + ', '.join ([term[0] for term in topic_terms])
        
        # Plot the trajectory line
        plt.plot (probabilities, marker = 'o', linewidth = 2, label = topic_label)

    plt.title (f'Thematic Shift [{name}]', fontsize = 13, pad = 15)
    plt.xlabel ('Narrative Timeline (Text Progress)', fontsize = 11)
    plt.ylabel ('Topic Probability (Density)', fontsize = 11)
    plt.xticks (range (len (x_labels)), x_labels)
    plt.grid (True, linestyle = '--', alpha = 0.5)
    plt.legend (loc = 'upper left', bbox_to_anchor = (1.02, 1), borderaxespad = 0.0)
    plt.tight_layout ()
    plt.show ()