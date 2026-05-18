import collections
from typing import List, Literal, Tuple, Union

import gensim.corpora as corpora
import nltk
import pandas as pd
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from sentence_transformers import SentenceTransformer, models


# =============================================================================
# Lazy nltk setup
# =============================================================================

def _ensure_nltk():
    for resource, path in (('stopwords', 'corpora/stopwords'),
                           ('punkt_tab', 'tokenizers/punkt_tab')):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


# =============================================================================
# Model initialisation
# =============================================================================

def initialize_model(embedding_model: Union[SentenceTransformer, str] = 'specter',
                     nr_topics: int = None,
                     n_gram_range: Union[Literal['unigram', 'bigram'], Tuple[int, int]] = 'unigram',
                     verbose: bool = False):
    """Validate inputs and resolve embedding model / n_gram_range shorthand."""
    if isinstance(n_gram_range, str):
        if n_gram_range == 'unigram':
            n_gram_range = (1, 1)
        elif n_gram_range == 'bigram':
            n_gram_range = (2, 2)
        else:
            raise ValueError("n_gram_range must be 'unigram', 'bigram', or a (int, int) tuple")
    elif not isinstance(n_gram_range, tuple):
        raise TypeError("n_gram_range must be a string or tuple")

    if isinstance(embedding_model, str):
        key = embedding_model.lower()
        if key == 'specter':
            embedding_model = SentenceTransformer('allenai-specter')
        elif key == 'specter2':
            embedding_model = SentenceTransformer('allenai/specter2_base')
        elif key == 'aspire':
            word_embedding_model = models.Transformer('allenai/aspire-sentence-embedder',
                                                      max_seq_length=512)
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
            embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        elif key == 'scibert':
            word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased',
                                                      max_seq_length=512)
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
            embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        else:
            raise ValueError(f"Unknown embedding model shorthand: {embedding_model!r}")

    return embedding_model, n_gram_range


# =============================================================================
# Fit
# =============================================================================

def fit(data: List[str],
        embeddings=None,
        embedding_model: Union[SentenceTransformer, str] = 'specter',
        nr_topics: int = None,
        n_gram_range: Union[Literal['unigram', 'bigram'], Tuple[int, int]] = 'unigram',
        coherence_scorer: str = "c_v",
        verbose: bool = False,
        **bertopic_kwargs):
    """Fit a BERTopic model and return topics, probabilities, sizes, the model,
    topic words, and coherence/diversity metrics."""
    opts = {}

    if not isinstance(embedding_model, SentenceTransformer) and not isinstance(n_gram_range, tuple):
        embedding_model, opts['n_gram_range'] = initialize_model(
            embedding_model, nr_topics, n_gram_range, verbose,
        )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        nr_topics=nr_topics,
        verbose=verbose,
        **opts,
        **bertopic_kwargs,
    )
    if embeddings is None:
        topics, probabilities = topic_model.fit_transform(data)
    else:
        topics, probabilities = topic_model.fit_transform(data, embeddings)

    metrics     = _calculate_metrics(data, topic_model, topics, coherence_scorer)
    topic_sizes = _calculate_topic_sizes(topics)
    topic_words = topic_model.topic_representations_
    return topics, probabilities, topic_sizes, topic_model, topic_words, metrics


# =============================================================================
# Metrics
# =============================================================================

def _calculate_topic_sizes(topics: List[int]) -> dict:
    return dict(collections.Counter(topics))


def _calculate_metrics(texts: List[str], topic_model: BERTopic,
                       topics: List[int], coherence_scorer: str) -> dict:
    """Compute Topic Coherence and Topic Diversity."""
    topic_dict  = topic_model.topic_representations_
    topic_words = {k: [x[0] for x in topic_dict[k]] for k in topic_dict}
    word_list   = list(topic_words.values())
    word_list.pop(0)  # remove the outlier topic (-1)

    coherence_score = _calculate_coherence(topic_model, texts, topics, coherence_scorer)
    diversity_score = TopicDiversity(topk=10).score({'topics': word_list})
    return {'Coherence': coherence_score, 'Diversity': diversity_score}


def _calculate_coherence(topic_model: BERTopic, docs: List[str],
                         topics: List[int], coherence_scorer: str) -> float:
    """Compute coherence via gensim CoherenceModel."""
    _ensure_nltk()

    documents = pd.DataFrame({'Document': docs, 'ID': range(len(docs)), 'Topic': topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

    vectorizer = topic_model.vectorizer_model
    analyzer   = vectorizer.build_analyzer()
    tokens     = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus     = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [
        [w for w, _ in topic_model.get_topic(t)]
        for t in range(len(set(topics)) - 1)
    ]

    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokens,
        corpus=corpus,
        dictionary=dictionary,
        processes=-1,
        coherence=coherence_scorer,
    )
    return coherence_model.get_coherence()
