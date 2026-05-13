from __future__ import annotations

def build_sentiment_prompt(review_text: str, aspect: str, evidence: str) -> tuple[str, str]:
    """Build a prompt for aspect-conditioned sentiment classification."""
    system = (
        "You are an expert in Aspect-Based Sentiment Analysis (ABSA).\n"
        "Classify the sentiment of the specific aspect provided below, based strictly on the review text and the supporting evidence.\n\n"
        "Instructions:\n"
        "1. Focus only on the sentiment expressed towards the specific aspect.\n"
        "2. If the sentiment is clearly favorable, return 'positive'.\n"
        "3. If the sentiment is clearly unfavorable, return 'negative'.\n"
        "4. If the sentiment is balanced, mixed, or missing, return 'neutral'.\n"
        "5. Return ONLY the sentiment word (positive/negative/neutral) in lowercase."
    )
    user = (
        f"Review: {review_text}\n"
        f"Aspect: {aspect}\n"
        f"Evidence: {evidence}"
    )
    return system, user

def build_batch_sentiment_prompt(review_text: str, aspects: list[str]) -> tuple[str, str]:
    """Build a prompt for classifying multiple aspects in one go."""
    aspects_str = "\n".join([f"- {a}" for a in aspects])
    system = (
        "You are an expert in Aspect-Based Sentiment Analysis (ABSA).\n"
        "Classify the sentiment for each of the aspects listed below based on the review text.\n\n"
        "Instructions:\n"
        "1. Return a JSON list of objects, one for each aspect.\n"
        "2. Each object must have fields: 'aspect' (string) and 'sentiment' (one of: positive, negative, neutral).\n"
        "3. Return ONLY the raw JSON list, no preamble or extra text.\n"
        "Example: [{\"aspect\": \"food\", \"sentiment\": \"positive\"}]"
    )
    user = (
        f"Review: {review_text}\n\n"
        "Aspects to classify:\n"
        f"{aspects_str}"
    )
    return system, user
