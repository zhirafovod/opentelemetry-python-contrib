"""
DeepEval Metric-Specific Labels
Shows what labels each metric actually returns
"""

def get_metric_specific_labels():
    """
    DeepEval metrics return specific labels, not just Pass/Fail:
    
    BiasMetric:
        - "Not Biased" (score < threshold)
        - "Biased" (score >= threshold)
    
    ToxicityMetric:
        - "Not Toxic" (score < threshold) 
        - "Toxic" (score >= threshold)
    
    HallucinationMetric:
        - "Not Hallucinated" (score < threshold)
        - "Hallucinated" (score >= threshold)
    
    AnswerRelevancyMetric:
        - "Relevant" (score >= threshold)
        - "Not Relevant" (score < threshold)
    
    SummarizationMetric:
        - "Good Summary" (score >= threshold)
        - "Poor Summary" (score < threshold)
    
    FaithfulnessMetric:
        - "Faithful" (score >= threshold)
        - "Not Faithful" (score < threshold)
    """
    return {
        "bias": {
            "success_label": "Not Biased",
            "failure_label": "Biased"
        },
        "toxicity": {
            "success_label": "Not Toxic", 
            "failure_label": "Toxic"
        },
        "hallucination": {
            "success_label": "Not Hallucinated",
            "failure_label": "Hallucinated"
        },
        "answerrelevancy": {
            "success_label": "Relevant",
            "failure_label": "Not Relevant"
        },
        "summarization": {
            "success_label": "Good Summary",
            "failure_label": "Poor Summary"
        },
        "faithfulness": {
            "success_label": "Faithful",
            "failure_label": "Not Faithful"
        },
        "sentiment": {
            "success_label": "Positive",
            "failure_label": "Negative",
            "neutral_label": "Neutral"
        }
    }

def get_metric_label(metric_name: str, score: float, threshold: float = 0.5) -> str:
    """Get the appropriate label for a metric based on its score"""
    labels = get_metric_specific_labels()
    
    if metric_name in labels:
        # Special handling for sentiment analysis (3-way classification)
        if metric_name == "sentiment":
            if score >= 0.525:  # Positive threshold (matches VADER's 0.05 mapped to normalized scale)
                return labels[metric_name]["success_label"]  # "Positive"
            elif score <= 0.475:  # Negative threshold (matches VADER's -0.05 mapped to normalized scale)
                return labels[metric_name]["failure_label"]  # "Negative"
            else:
                return labels[metric_name]["neutral_label"]  # "Neutral"
        
        # Standard binary classification for other metrics
        # For bias, toxicity, and hallucination: lower scores are better (success)
        if metric_name in ["bias", "toxicity", "hallucination"]:
            if score < threshold:
                return labels[metric_name]["success_label"]
            else:
                return labels[metric_name]["failure_label"]
        else:
            # For relevancy, summarization, faithfulness: higher scores are better
            if score >= threshold:
                return labels[metric_name]["success_label"]
            else:
                return labels[metric_name]["failure_label"]
    
    # Fallback to generic Pass/Fail
    return "Pass" if score >= threshold else "Fail"
