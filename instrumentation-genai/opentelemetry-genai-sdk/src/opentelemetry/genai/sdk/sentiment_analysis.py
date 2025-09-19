"""
NLTK Sentiment Analysis for GenAI SDK
Demonstrates NLTK sentiment analysis integration with DeepEval-compatible format
"""

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, Any

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class NLTKSentimentAnalyzer:
    """NLTK-based sentiment analysis compatible with DeepEval result format"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        scores = self.analyzer.polarity_scores(text)
        
        # NLTK VADER returns:
        # {
        #     'neg': 0.0,      # Negative sentiment (0.0 to 1.0)
        #     'neu': 0.5,      # Neutral sentiment (0.0 to 1.0)  
        #     'pos': 0.5,      # Positive sentiment (0.0 to 1.0)
        #     'compound': 0.0  # Overall sentiment (-1.0 to 1.0)
        # }
        
        # Convert compound score from [-1, 1] to [0, 1] scale
        compound_score = scores['compound']
        normalized_score = (compound_score + 1) / 2
    
        # Use VADER's original thresholds mapped to normalized scale
        # Original VADER: positive >= 0.05, negative <= -0.05
        # Normalized: positive >= 0.525, negative <= 0.475
        if normalized_score >= 0.525:
            sentiment = "Positive"
        elif normalized_score <= 0.475:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        # Return in DeepEval-compatible format (without hardcoded label)
        return {
            "score": normalized_score,
            "reason": f"Sentiment analysis: {sentiment} (compound: {scores['compound']:.3f})",
            "range": "[0,1]",
            "judge_model": "NLTK VADER",
            "raw_scores": scores,  # Include original NLTK scores for reference
            "breakdown": {
                "positive": scores['pos'],
                "negative": scores['neg'],
                "neutral": scores['neu']
            }
        }

def demo_nltk_sentiment():
    """Demo function showing NLTK sentiment analysis scores"""
    
    analyzer = NLTKSentimentAnalyzer()
    
    test_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is terrible. I hate it and it doesn't work at all.",
        "The weather is okay today. Nothing special.",
        "The AI assistant provided helpful and accurate information.",
        "The response was confusing and contained several errors."
    ]
    
    print("NLTK Sentiment Analysis Demo")
    print("=" * 50)
    
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"\nText: {text}")
        print(f"Score: {result['score']:.3f}")
        print(f"Label: {result['label']}")
        print(f"Raw NLTK scores: {result['raw_scores']}")
        print("-" * 30)

if __name__ == "__main__":
    demo_nltk_sentiment()
