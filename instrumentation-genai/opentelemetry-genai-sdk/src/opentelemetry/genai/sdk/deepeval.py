from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics import SummarizationMetric
from deepeval.metrics import BiasMetric
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
import openai
import os

class CiscoChatOpenAILLM(DeepEvalBaseLLM):
    """Custom DeepEval LLM wrapper using OpenAI client for Cisco-hosted Azure OpenAI"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url='https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini',
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
                extra_body={"user": '{"appkey": "<key>"}'}
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Error in generate: {e}")
            return ""
    
    async def a_generate(self, prompt: str) -> str:
        try:
            async_client = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url='https://chat-ai.cisco.com/openai/deployments/gpt-4o-mini',
                default_headers={"api-key": os.getenv("OPENAI_API_KEY")},
                timeout=30.0,
                max_retries=2
            )
            response = await async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
                extra_body={"user": '{"appkey": "<key>"}'}
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
    faithfulness_metric = FaithfulnessMetric(model=cisco_llm, async_mode=False)
    summarization_metric = SummarizationMetric(model=cisco_llm, async_mode=False)
    bias_metric = BiasMetric(model=cisco_llm, async_mode=False)
    hallucination_metric = HallucinationMetric(model=cisco_llm, async_mode=False)
    
    metrics = [relevancy_metric, faithfulness_metric, summarization_metric, bias_metric, hallucination_metric]
    
    results = {}
    
    for metric in metrics:
        try:
            metric.measure(test_case)
            
            if hasattr(metric, 'score') and metric.score is not None:
                metric_name = type(metric).__name__.lower().replace('metric', '')
                results[metric_name] = {
                    "score": metric.score,
                    "reason": getattr(metric, 'reason', ''),
                    "label": "Pass" if (metric.score or 0) > 0.5 else "Fail",
                    "range": "[0,1]",
                    "judge_model": getattr(metric, 'evaluation_model', 'unknown')
                }
                
        except Exception as e:
            metric_name = type(metric).__name__.lower().replace('metric', '')
            results[metric_name] = {"error": str(e)}
    
    return results