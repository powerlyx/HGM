"""Evaluation metrics for memory QA predictions."""

import os
import time
from typing import List, Dict, Union
import statistics
from collections import defaultdict
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics.pairwise import cosine_similarity
import logging
from openai import OpenAI

try:
    from openai import APITimeoutError, APIConnectionError, InternalServerError, RateLimitError

    RETRYABLE_OPENAI_ERRORS = (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
        TimeoutError,
    )
except Exception:
    RETRYABLE_OPENAI_ERRORS = (TimeoutError,)

logger = logging.getLogger("amem_utils")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120"))
OPENAI_MAX_RETRIES = max(0, int(os.getenv("OPENAI_MAX_RETRIES", "4")))
OPENAI_RETRY_BASE_DELAY = max(0.1, float(os.getenv("OPENAI_RETRY_BASE_DELAY", "1.0")))
ENABLE_BERT_SCORE = os.getenv("AMEM_ENABLE_BERT_SCORE", "0") == "1"
ENABLE_SBERT_SIMILARITY = os.getenv("AMEM_ENABLE_SBERT_SIMILARITY", "0") == "1"
DOWNLOAD_NLTK_DATA = os.getenv("AMEM_DOWNLOAD_NLTK_DATA", "0") == "1"

_embedding_client: Union[OpenAI, None] = None
_embedding_client_error: Union[Exception, None] = None
_nltk_data_checked = False


def _ensure_nltk_data(download: bool = False) -> None:
    """Check NLTK resources without causing network I/O during imports."""
    global _nltk_data_checked
    if _nltk_data_checked:
        return

    missing = []
    for resource in ("tokenizers/punkt", "corpora/wordnet"):
        try:
            nltk.data.find(resource)
        except LookupError:
            missing.append(resource.split("/")[-1])

    if missing and download:
        for package_name in missing:
            try:
                nltk.download(package_name, quiet=True)
            except Exception as exc:
                logger.warning("Unable to download NLTK data %s: %s", package_name, exc)
    elif missing:
        logger.debug("NLTK data missing; falling back where possible: %s", ", ".join(missing))

    _nltk_data_checked = True


def _get_embedding_client() -> Union[OpenAI, None]:
    global _embedding_client, _embedding_client_error
    if _embedding_client is not None:
        return _embedding_client
    if _embedding_client_error is not None:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        _embedding_client_error = ValueError("OPENAI_API_KEY is missing")
        logger.warning("OpenAI embeddings disabled for metrics: OPENAI_API_KEY is missing")
        return None

    base_url = os.getenv("OPENAI_BASE_URL")
    try:
        _embedding_client = OpenAI(api_key=api_key, base_url=base_url)
        return _embedding_client
    except Exception as exc:
        _embedding_client_error = exc
        logger.warning("Failed to initialize OpenAI embeddings client for metrics: %s", exc)
        return None


def _embed_texts_with_retry(texts: List[str]) -> Union[List[List[float]], None]:
    client = _get_embedding_client()
    if client is None:
        return None

    last_exc = None
    for attempt in range(OPENAI_MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[str(text or "") for text in texts],
                timeout=OPENAI_TIMEOUT_SECONDS,
            )
            return [item.embedding for item in response.data]
        except RETRYABLE_OPENAI_ERRORS as exc:
            last_exc = exc
            if attempt < OPENAI_MAX_RETRIES:
                delay = OPENAI_RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Metric embedding call failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt + 1, OPENAI_MAX_RETRIES + 1, exc, delay,
                )
                time.sleep(delay)
                continue
            logger.error("Metric embedding call failed after %d attempts: %s", OPENAI_MAX_RETRIES + 1, exc)
            return None
        except Exception as exc:
            logger.error("Metric embedding call failed with non-retryable error: %s", exc)
            return None

    logger.error("Metric embedding call ended without a response: %s", last_exc)
    return None


def simple_tokenize(text):
    text = str(text)
    return text.lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split()


def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rougeL_f': scores['rougeL'].fmeasure
    }


def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    _ensure_nltk_data(download=DOWNLOAD_NLTK_DATA)
    try:
        pred_tokens = nltk.word_tokenize(prediction.lower())
        ref_tokens = [nltk.word_tokenize(reference.lower())]
    except LookupError:
        pred_tokens = simple_tokenize(prediction.lower())
        ref_tokens = [simple_tokenize(reference.lower())]

    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth = SmoothingFunction().method1

    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        except Exception:
            score = 0.0
        scores[f'bleu{n}'] = score

    return scores


def calculate_bert_scores(prediction: str, reference: str) -> Dict[str, float]:
    try:
        from bert_score import score as bert_score

        P, R, F1 = bert_score([prediction], [reference], lang='en', verbose=False)
        return {
            'bert_precision': P.item(),
            'bert_recall': R.item(),
            'bert_f1': F1.item()
        }
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return {'bert_precision': 0.0, 'bert_recall': 0.0, 'bert_f1': 0.0}


def calculate_meteor_score(prediction: str, reference: str) -> float:
    try:
        _ensure_nltk_data(download=DOWNLOAD_NLTK_DATA)
        return meteor_score([reference.split()], prediction.split())
    except Exception as e:
        print(f"Error calculating METEOR score: {e}")
        return 0.0


def calculate_sentence_similarity(prediction: str, reference: str) -> float:
    try:
        vectors = _embed_texts_with_retry([prediction, reference])
        if not vectors or len(vectors) < 2:
            return 0.0
        similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Error calculating sentence similarity: {e}")
        return 0.0


def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    if not prediction or not reference:
        return {
            "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,
            "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,
            "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
        }

    prediction = str(prediction).strip()
    reference = str(reference).strip()

    exact_match = int(prediction.lower() == reference.lower())

    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens

    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    rouge_scores = calculate_rouge_scores(prediction, reference)
    bleu_scores = calculate_bleu_scores(prediction, reference)
    if ENABLE_BERT_SCORE:
        bert_scores = calculate_bert_scores(prediction, reference)
    else:
        bert_scores = {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}

    meteor = calculate_meteor_score(prediction, reference)
    if ENABLE_SBERT_SIMILARITY:
        sbert_similarity = calculate_sentence_similarity(prediction, reference)
    else:
        sbert_similarity = 0.0

    metrics = {
        "exact_match": exact_match,
        "f1": f1,
        **rouge_scores,
        **bleu_scores,
        **bert_scores,
        "meteor": meteor,
        "sbert_similarity": sbert_similarity
    }

    return metrics


def aggregate_metrics(all_metrics: List[Dict[str, float]], all_categories: List[int]) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
    if not all_metrics:
        return {}

    aggregates = defaultdict(list)
    category_aggregates = defaultdict(lambda: defaultdict(list))

    for metrics, category in zip(all_metrics, all_categories):
        for metric_name, value in metrics.items():
            aggregates[metric_name].append(value)
            category_aggregates[category][metric_name].append(value)

    results = {"overall": {}}

    for metric_name, values in aggregates.items():
        results["overall"][metric_name] = {
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }

    for category in sorted(category_aggregates.keys()):
        results[f"category_{category}"] = {}
        for metric_name, values in category_aggregates[category].items():
            if values:
                results[f"category_{category}"][metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }

    return results
