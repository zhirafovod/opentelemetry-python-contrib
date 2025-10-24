#!/usr/bin/env python3
"""Example demonstrating automatic Traceloop span translation.

The TraceloopSpanProcessor is automatically enabled when you install the package!
A .pth file ensures the translator is loaded on Python startup, and a deferred
registration hook attaches it when the TracerProvider is set up.

No import needed - just install the package and it works!

You can still customize the transformation rules by setting environment variables
or by manually calling enable_traceloop_translator() with custom parameters.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

# Load .env first
load_dotenv()

try:
    from traceloop.sdk import Traceloop
    from traceloop.sdk.decorators import task, workflow
    from openai import OpenAI
    Traceloop.init(disable_batch=True, api_endpoint="http://localhost:4318")
except ImportError:
    raise RuntimeError("Install traceloop-sdk: pip install traceloop-sdk")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

@task(name="joke_creation")
def create_joke():
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    return completion.choices[0].message.content

@task(name="signature_generation")
def generate_signature(joke: str):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Also tell me a joke about yourself!"}],
    )

    return completion.choices[0].message.content

@workflow(name="pirate_joke_generator")
def joke_workflow():
    eng_joke = create_joke()
    #pirate_joke = translate_joke_to_pirate(eng_joke)
    signature = generate_signature(eng_joke)
    print(eng_joke + "\n\n" + signature)

if __name__ == "__main__":
    #run_example()
    joke_workflow()
