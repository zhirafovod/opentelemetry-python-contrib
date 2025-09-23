# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import uuid

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.data import Error
from opentelemetry.instrumentation.langchain.config import Config
from opentelemetry.instrumentation.langchain.utils import dont_throw
from opentelemetry.util.genai.types import EmbeddingInvocation
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes

def embeddings_wrapper(telemetry_handler: TelemetryHandler):
    #TODO update comment
    """Wrap the embed_query method of Embeddings classes to trace it."""
    
    @dont_throw
    def traced_method(wrapped, instance, args, kwargs):
        if Config.is_instrumentation_suppressed():
            return wrapped(*args, **kwargs)
        
        # Get model information from the instance
        model_name = getattr(instance, 'model', None) or getattr(instance, 'model_name', None) or instance.__class__.__name__

        input_value = args[0]
        # If input_value is a single string, put it in a list
        if isinstance(input_value, str):
            input_value = [input_value]

        embedding_kwargs = {
            "input": input_value,
        }

        invocation = EmbeddingInvocation(
            operation_name=gen_ai_attributes.GenAiOperationNameValues.EMBEDDINGS.value,
            request_model=model_name,
            attributes=embedding_kwargs
        )

        start_invocation = telemetry_handler.start_embedding(invocation)

        try:
            # Call the original method - wrapped is already bound to the instance
            result = wrapped(*args, **kwargs)
            
            # Determine the dimension count based on result structure
            if isinstance(result[0], list):
                dimension_count = len(result[0])
            else:
                dimension_count = len(result)

            start_invocation.dimension_count = dimension_count
            
            telemetry_handler.stop_embedding(start_invocation)
            return result

        except Exception as ex:
            embedding_error = Error(message=str(ex), type=type(ex))
            telemetry_handler.fail_embedding(start_invocation, embedding_error)
            raise

    return traced_method