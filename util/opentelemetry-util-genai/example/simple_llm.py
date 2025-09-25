"""
A simple mock LLM class for demonstration purposes.
"""

import random
import time


class SimpleLLM:
    def __init__(self, name="mock-llm"):
        self.name = name

    def generate(self, prompt):
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.3))
        if "fail" in prompt:
            raise RuntimeError("Simulated LLM failure!")
        return f"Echo: {prompt[::-1]}"
