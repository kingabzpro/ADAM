from math import isclose, isfinite


def load_classifier(model: str = "ProsusAI/finbert", *, device: str = "cpu", local_files_only: bool = False):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=local_files_only)
    weights = AutoModelForSequenceClassification.from_pretrained(model, local_files_only=local_files_only)
    return pipeline("text-classification", model=weights, tokenizer=tokenizer, device=device)


def sentiment_evidence(texts: list[str], classifier) -> dict:
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("At least one nonempty article is required")
    outputs = classifier(texts, top_k=None, truncation=True)
    if len(outputs) != len(texts):
        raise ValueError("FinBERT returned the wrong number of results")
    labels = {"positive", "negative", "neutral"}
    articles = []
    for output in outputs:
        try:
            scores = {item["label"].lower(): float(item["score"]) for item in output}
            if (len(output) != 3 or set(scores) != labels
                    or any(not isfinite(score) or not 0 <= score <= 1 for score in scores.values())
                    or not isclose(sum(scores.values()), 1.0, abs_tol=1e-5)):
                raise ValueError("Invalid FinBERT scores")
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("Invalid FinBERT response") from error
        articles.append({"label": max(scores, key=scores.get), "scores": scores})
    return {"articles": articles,
            "mean_scores": {label: sum(item["scores"][label] for item in articles) / len(articles)
                            for label in sorted(labels)}}


async def assess_news(ticker, evidence, model):
    return await model.assess(ticker, "news", evidence)
