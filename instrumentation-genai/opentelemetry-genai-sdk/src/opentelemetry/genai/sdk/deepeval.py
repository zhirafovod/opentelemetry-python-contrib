from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import HallucinationMetric
from deepeval.metrics import ToxicityMetric
from deepeval.metrics import BiasMetric
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
import openai
import os
import json
from datetime import datetime, timedelta
import base64
import requests

class TokenManager:
    def __init__(self, client_id, client_secret, app_key, cache_file=".token.json"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.app_key = app_key
        self.cache_file = cache_file
        self.token_url = "https://id.cisco.com/oauth2/default/v1/token"
        
    def _get_cached_token(self):
        if not os.path.exists(self.cache_file):
            return None
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
                
            expires_at = datetime.fromisoformat(cache_data['expires_at'])
            if datetime.now() < expires_at - timedelta(minutes=5):
                return cache_data['access_token']
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return None
    
    def _fetch_new_token(self):
        payload = "grant_type=client_credentials"
        value = base64.b64encode(f'{self.client_id}:{self.client_secret}'.encode('utf-8')).decode('utf-8')
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {value}"
        }
        
        response = requests.post(self.token_url, headers=headers, data=payload)
        response.raise_for_status()
        
        token_data = response.json()
        expires_in = token_data.get('expires_in', 3600)
        expires_at = datetime.now() + timedelta(seconds=expires_in)
        
        cache_data = {
            'access_token': token_data['access_token'],
            'expires_at': expires_at.isoformat(),
        }
        
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        os.chmod(self.cache_file, 0o600)
        return token_data['access_token']
    
    def get_token(self):
        token = self._get_cached_token()
        if token:
            return token
        return self._fetch_new_token()
    
    def cleanup_token_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r+b') as f:
                length = f.seek(0, 2)
                f.seek(0)
                f.write(b'\0' * length)
            os.remove(self.cache_file)

class CiscoChatOpenAILLM(DeepEvalBaseLLM):
    """Custom DeepEval LLM wrapper using OpenAI client for Cisco-hosted Azure OpenAI"""
    
    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        
        cisco_client_id = os.getenv("CISCO_CLIENT_ID")
        cisco_client_secret = os.getenv("CISCO_CLIENT_SECRET")
        self.cisco_app_key = os.getenv("CISCO_APP_KEY")
        
        self.token_manager = TokenManager(cisco_client_id, cisco_client_secret, self.cisco_app_key, "/tmp/.token.json")
        api_key = self.token_manager.get_token()
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url='https://chat-ai.cisco.com/openai/deployments/gpt-4.1',
            default_headers={"api-key": api_key},
            timeout=30.0,
            max_retries=2
        )
    
    def load_model(self):
        return self
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
                extra_body={"user": '{"appkey": "' + self.cisco_app_key + '"}'}
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Error in generate: {e}")
            return ""
    
    async def a_generate(self, prompt: str) -> str:
        try:
            # Get fresh token for async operations
            api_key = self.token_manager.get_token()
            async_client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url='https://chat-ai.cisco.com/openai/deployments/gpt-4.1',
                default_headers={"api-key": api_key},
                timeout=30.0,
                max_retries=2
            )
            response = await async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
                extra_body={"user": '{"appkey": "' + self.cisco_app_key + '"}'}
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Error in a_generate: {e}")
            return ""
    
    def get_model_name(self) -> str:
        return self.model

def evaluate_all_metrics(prompt:str, output:str, retrieval_context:list) -> dict:
    ground_truth_db = {
        "what is the capital of france?": {
            "context": ["France is a country in Europe. Paris is the capital city of France."],
            "retrieval_context": ["Paris is the capital and most populous city of France."]
        },
        "what is the capital of germany?": {
            "context": ["Germany is a country in Europe. Berlin is the capital city of Germany."],
            "retrieval_context": ["Berlin is the capital and largest city of Germany."]
        },
        "what is the capital of italy?": {
            "context": ["Italy is a country in Europe. Rome is the capital city of Italy."],
            "retrieval_context": ["Rome is the capital and largest city of Italy."]
        },
        "what is the capital of spain?": {
            "context": ["Spain is a country in Europe. Madrid is the capital city of Spain."],
            "retrieval_context": ["Madrid is the capital and largest city of Spain."]
        },
        "what is the capital of united kingdom?": {
            "context": ["United Kingdom is a country in Europe. London is the capital city of United Kingdom."],
            "retrieval_context": ["London is the capital and largest city of the United Kingdom."]
        },
        "what is the capital of japan?": {
            "context": ["Japan is a country in Asia. Tokyo is the capital city of Japan."],
            "retrieval_context": ["Tokyo is the capital and most populous city of Japan."]
        },
        "what is the capital of canada?": {
            "context": ["Canada is a country in North America. Ottawa is the capital city of Canada."],
            "retrieval_context": ["Ottawa is the capital city of Canada."]
        },
        "what is the capital of australia?": {
            "context": ["Australia is a country and continent. Canberra is the capital city of Australia."],
            "retrieval_context": ["Canberra is the capital city of Australia."]
        },
        "what is the capital of brazil?": {
            "context": ["Brazil is a country in South America. Brasília is the capital city of Brazil."],
            "retrieval_context": ["Brasília is the capital city of Brazil."]
        },
        "what is the capital of india?": {
            "context": ["India is a country in Asia. New Delhi is the capital city of India."],
            "retrieval_context": ["New Delhi is the capital city of India."]
        },
        "what is the capital of united states?": {
            "context": ["United States is a country in North America. Washington D.C. is the capital city of United States."],
            "retrieval_context": ["Washington D.C. is the capital city of the United States."]
        }
    }
    
    normalized_prompt = prompt.lower().strip().rstrip('?')
    
    if normalized_prompt in ground_truth_db:
        context = ground_truth_db[normalized_prompt]["context"]
        if not retrieval_context:  # Use lookup if not provided
            retrieval_context = ground_truth_db[normalized_prompt]["retrieval_context"]
    else:
        context = ["No specific ground truth available for this question."]
        if not retrieval_context:
            retrieval_context = ["No specific retrieval context available."]
    
    test_case = LLMTestCase(
        input=prompt,
        actual_output=output,
        retrieval_context=retrieval_context,
        context=context 
    )

    cisco_llm = CiscoChatOpenAILLM()
    
    relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=cisco_llm, async_mode=False)
    summarization_metric = SummarizationMetric(model=cisco_llm, async_mode=False)
    toxicity_metric = ToxicityMetric(model=cisco_llm, async_mode=False)
    bias_metric = BiasMetric(model=cisco_llm, async_mode=False)
    hallucination_metric = HallucinationMetric(model=cisco_llm, async_mode=False)
    
    # Add NLTK sentiment analysis
    from .sentiment_analysis import NLTKSentimentAnalyzer
    sentiment_analyzer = NLTKSentimentAnalyzer()
    
    metrics = [relevancy_metric, summarization_metric, toxicity_metric, bias_metric, hallucination_metric]
    
    results = {}
    
    for metric in metrics:
        try:
            metric.measure(test_case)
            
            if hasattr(metric, 'score') and metric.score is not None:
                metric_name = type(metric).__name__.lower().replace('metric', '')
                
                # Use metric-specific labels instead of generic Pass/Fail
                from .deepeval_labels import get_metric_label
                threshold = getattr(metric, 'threshold', 0.5)
                metric_label = get_metric_label(metric_name, metric.score, threshold)
                
                results[metric_name] = {
                    "score": metric.score,
                    "reason": getattr(metric, 'reason', ''),
                    "label": metric_label,
                    "range": "[0,1]",
                    "judge_model": getattr(metric, 'evaluation_model', 'unknown'),
                    "threshold": threshold
                }
                
        except Exception as e:
            metric_name = type(metric).__name__.lower().replace('metric', '')
            results[metric_name] = {"error": str(e)}
            print(f"Metric: {metric_name} - ERROR: {str(e)}")
            print("-" * 80)
    
    # Add NLTK sentiment analysis
    try:
        sentiment_result = sentiment_analyzer.analyze_sentiment(output)
        
        # Apply consistent labeling using the same system as other metrics
        from .deepeval_labels import get_metric_label
        sentiment_score = sentiment_result["score"]
        sentiment_label = get_metric_label("sentiment", sentiment_score)
        
        # Update the result with the consistent label
        sentiment_result["label"] = sentiment_label
        sentiment_result["threshold"] = "0.525-0.475 (neutral zone)"
        
        results["sentiment"] = sentiment_result
    except Exception as e:
        results["sentiment"] = {"error": str(e)}
        print(f"Metric: sentiment - ERROR: {str(e)}")
        print("-" * 80)
    print(results)
    return results