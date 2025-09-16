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

import inspect


def _is_async_method(fn):
    # check if co-routine function or async generator( example : using async & yield)
    if inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn):
        return True

    # Check if this is a wrapped function that might hide the original async nature
    # Look for common wrapper attributes that might contain the original function
    for attr_name in ["__wrapped__", "func", "_func", "function"]:
        if hasattr(fn, attr_name):
            wrapped_fn = getattr(fn, attr_name)
            if wrapped_fn and callable(wrapped_fn):
                if inspect.iscoroutinefunction(
                    wrapped_fn
                ) or inspect.isasyncgenfunction(wrapped_fn):
                    return True
                # Recursively check in case of multiple levels of wrapping
                if _is_async_method(wrapped_fn):
                    return True

    return False


def _is_async_generator(fn):
    """Check if function is an async generator, looking through wrapped functions"""
    if inspect.isasyncgenfunction(fn):
        return True

    # Check if this is a wrapped function that might hide the original async generator nature
    for attr_name in ["__wrapped__", "func", "_func", "function"]:
        if hasattr(fn, attr_name):
            wrapped_fn = getattr(fn, attr_name)
            if wrapped_fn and callable(wrapped_fn):
                if inspect.isasyncgenfunction(wrapped_fn):
                    return True
                # Recursively check in case of multiple levels of wrapping
                if _is_async_generator(wrapped_fn):
                    return True

    return False


def _get_original_function_name(fn):
    """Extract the original function name from potentially wrapped functions"""
    if hasattr(fn, "__qualname__") and fn.__qualname__:
        return fn.__qualname__

    # Look for the original function in common wrapper attributes
    for attr_name in ["__wrapped__", "func", "_func", "function"]:
        if hasattr(fn, attr_name):
            wrapped_fn = getattr(fn, attr_name)
            if wrapped_fn and callable(wrapped_fn):
                if (
                    hasattr(wrapped_fn, "__qualname__")
                    and wrapped_fn.__qualname__
                ):
                    return wrapped_fn.__qualname__
                # Recursively check in case of multiple levels of wrapping
                result = _get_original_function_name(wrapped_fn)
                if result:
                    return result

    # Fallback to function name if qualname is not available
    return getattr(fn, "__name__", "unknown_function")


def _extract_tool_functions_from_args_kwargs(args, kwargs):
    """Extract tool functions from function arguments"""
    from opentelemetry.genai.sdk.data import ToolFunction

    tool_functions = []

    # Try to find tools in various places
    tools = None

    # Check kwargs for tools
    if kwargs.get("tools"):
        tools = kwargs["tools"]
    elif kwargs.get("functions"):
        tools = kwargs["functions"]

    # Check args for objects that might have tools
    if not tools and len(args) > 0:
        for arg in args:
            if hasattr(arg, "tools"):
                tools = getattr(arg, "tools", [])
                break
            elif hasattr(arg, "functions"):
                tools = getattr(arg, "functions", [])
                break

    # Convert tools to ToolFunction objects
    if tools:
        for tool in tools:
            try:
                # Handle different tool formats
                if hasattr(tool, "name"):
                    # LangChain-style tool
                    tool_name = tool.name
                    tool_description = getattr(tool, "description", "")
                elif isinstance(tool, dict) and "name" in tool:
                    # Dict-style tool
                    tool_name = tool["name"]
                    tool_description = tool.get("description", "")
                elif hasattr(tool, "__name__"):
                    # Function-style tool
                    tool_name = tool.__name__
                    tool_description = getattr(tool, "__doc__", "") or ""
                else:
                    continue

                tool_functions.append(
                    ToolFunction(
                        name=tool_name,
                        description=tool_description,
                        parameters={},  # Add parameter extraction if needed
                    )
                )
            except Exception:
                # Skip tools that can't be processed
                continue

    return tool_functions


def _find_llm_instance(args, kwargs):
    """Find LLM instance using multiple approaches"""
    llm_instance = None

    try:
        import sys

        frame = sys._getframe(2)  # Get the decorated function's frame
        func = frame.f_code

        # Try to get the function object from the frame
        if hasattr(frame, "f_globals"):
            for name, obj in frame.f_globals.items():
                if (
                    hasattr(obj, "__code__")
                    and obj.__code__ == func
                    and hasattr(obj, "llm")
                ):
                    llm_instance = obj.llm
                    break
    except Exception:
        pass

    # Check kwargs for LLM instance
    if not llm_instance:
        for key, value in kwargs.items():
            if key.lower() in ["llm", "model", "client"] and _is_llm_instance(
                value
            ):
                llm_instance = value
                break

    # Check args for LLM instance
    if not llm_instance:
        for arg in args:
            if _is_llm_instance(arg):
                llm_instance = arg
                break
            # Check for bound tools that contain an LLM
            elif hasattr(arg, "llm") and _is_llm_instance(arg.llm):
                llm_instance = arg.llm
                break

    #  Frame inspection to look in local variables
    if not llm_instance:
        try:
            import sys

            frame = sys._getframe(
                2
            )  # Go up 2 frames to get to the decorated function
            local_vars = frame.f_locals

            # Look for ChatOpenAI or similar instances in local variables
            for var_name, var_value in local_vars.items():
                if _is_llm_instance(var_value):
                    llm_instance = var_value
                    break
                elif hasattr(var_value, "llm") and _is_llm_instance(
                    var_value.llm
                ):
                    # Handle bound tools case
                    llm_instance = var_value.llm
                    break
        except Exception:
            pass

    return llm_instance


def _is_llm_instance(obj):
    """Check if an object is an LLM instance"""
    if not hasattr(obj, "__class__"):
        return False

    class_name = obj.__class__.__name__
    module_name = (
        obj.__class__.__module__
        if hasattr(obj.__class__, "__module__")
        else ""
    )

    # Check for common LLM class patterns
    llm_patterns = [
        "ChatOpenAI",
        "OpenAI",
        "AzureOpenAI",
        "AzureChatOpenAI",
        "ChatAnthropic",
        "Anthropic",
        "ChatGoogleGenerativeAI",
        "GoogleGenerativeAI",
        "ChatVertexAI",
        "VertexAI",
        "ChatOllama",
        "Ollama",
        "ChatHuggingFace",
        "HuggingFace",
        "ChatCohere",
        "Cohere",
    ]

    return (
        any(pattern in class_name for pattern in llm_patterns)
        or "langchain" in module_name.lower()
    )


def _extract_llm_config_attributes(llm_instance, attributes):
    """Extract configuration attributes from LLM instance"""
    try:
        # Extract model
        if hasattr(llm_instance, "model_name") and llm_instance.model_name:
            attributes["request_model"] = str(llm_instance.model_name)
        elif hasattr(llm_instance, "model") and llm_instance.model:
            attributes["request_model"] = str(llm_instance.model)

        # Extract temperature
        if (
            hasattr(llm_instance, "temperature")
            and llm_instance.temperature is not None
        ):
            attributes["request_temperature"] = float(llm_instance.temperature)

        # Extract max_tokens
        if (
            hasattr(llm_instance, "max_tokens")
            and llm_instance.max_tokens is not None
        ):
            attributes["request_max_tokens"] = int(llm_instance.max_tokens)

        # Extract top_p
        if hasattr(llm_instance, "top_p") and llm_instance.top_p is not None:
            attributes["request_top_p"] = float(llm_instance.top_p)

        # Extract top_k
        if hasattr(llm_instance, "top_k") and llm_instance.top_k is not None:
            attributes["request_top_k"] = int(llm_instance.top_k)

        # Extract frequency_penalty
        if (
            hasattr(llm_instance, "frequency_penalty")
            and llm_instance.frequency_penalty is not None
        ):
            attributes["request_frequency_penalty"] = float(
                llm_instance.frequency_penalty
            )

        # Extract presence_penalty
        if (
            hasattr(llm_instance, "presence_penalty")
            and llm_instance.presence_penalty is not None
        ):
            attributes["request_presence_penalty"] = float(
                llm_instance.presence_penalty
            )

        # Extract seed
        if hasattr(llm_instance, "seed") and llm_instance.seed is not None:
            attributes["request_seed"] = int(llm_instance.seed)

        # Extract stop sequences
        if hasattr(llm_instance, "stop") and llm_instance.stop is not None:
            stop = llm_instance.stop
            if isinstance(stop, (list, tuple)):
                attributes["request_stop_sequences"] = list(stop)
            else:
                attributes["request_stop_sequences"] = [str(stop)]
        elif (
            hasattr(llm_instance, "stop_sequences")
            and llm_instance.stop_sequences is not None
        ):
            stop = llm_instance.stop_sequences
            if isinstance(stop, (list, tuple)):
                attributes["request_stop_sequences"] = list(stop)
            else:
                attributes["request_stop_sequences"] = [str(stop)]

    except Exception as e:
        print(f"Error extracting LLM config attributes: {e}")


def _extract_direct_parameters(args, kwargs, attributes):
    """Fallback method to extract parameters directly from args/kwargs"""
    # Temperature
    print("args:", args)
    print("kwargs:", kwargs)
    temperature = kwargs.get("temperature")
    if temperature is not None:
        attributes["request_temperature"] = float(temperature)
    elif hasattr(args[0] if args else None, "temperature"):
        temperature = getattr(args[0], "temperature", None)
        if temperature is not None:
            attributes["request_temperature"] = float(temperature)

    # Max tokens
    max_tokens = kwargs.get("max_tokens") or kwargs.get(
        "max_completion_tokens"
    )
    if max_tokens is not None:
        attributes["request_max_tokens"] = int(max_tokens)
    elif hasattr(args[0] if args else None, "max_tokens"):
        max_tokens = getattr(args[0], "max_tokens", None)
        if max_tokens is not None:
            attributes["request_max_tokens"] = int(max_tokens)

    # Top P
    top_p = kwargs.get("top_p")
    if top_p is not None:
        attributes["request_top_p"] = float(top_p)
    elif hasattr(args[0] if args else None, "top_p"):
        top_p = getattr(args[0], "top_p", None)
        if top_p is not None:
            attributes["request_top_p"] = float(top_p)

    # Top K
    top_k = kwargs.get("top_k")
    if top_k is not None:
        attributes["request_top_k"] = int(top_k)
    elif hasattr(args[0] if args else None, "top_k"):
        top_k = getattr(args[0], "top_k", None)
        if top_k is not None:
            attributes["request_top_k"] = int(top_k)

    # Frequency penalty
    frequency_penalty = kwargs.get("frequency_penalty")
    if frequency_penalty is not None:
        attributes["request_frequency_penalty"] = float(frequency_penalty)
    elif hasattr(args[0] if args else None, "frequency_penalty"):
        frequency_penalty = getattr(args[0], "frequency_penalty", None)
        if frequency_penalty is not None:
            attributes["request_frequency_penalty"] = float(frequency_penalty)

    # Presence penalty
    presence_penalty = kwargs.get("presence_penalty")
    if presence_penalty is not None:
        attributes["request_presence_penalty"] = float(presence_penalty)
    elif hasattr(args[0] if args else None, "presence_penalty"):
        presence_penalty = getattr(args[0], "presence_penalty", None)
        if presence_penalty is not None:
            attributes["request_presence_penalty"] = float(presence_penalty)

    # Stop sequences
    stop_sequences = kwargs.get("stop_sequences") or kwargs.get("stop")
    if stop_sequences is not None:
        if isinstance(stop_sequences, (list, tuple)):
            attributes["request_stop_sequences"] = list(stop_sequences)
        else:
            attributes["request_stop_sequences"] = [str(stop_sequences)]
    elif hasattr(args[0] if args else None, "stop_sequences"):
        stop_sequences = getattr(args[0], "stop_sequences", None)
        if stop_sequences is not None:
            if isinstance(stop_sequences, (list, tuple)):
                attributes["request_stop_sequences"] = list(stop_sequences)
            else:
                attributes["request_stop_sequences"] = [str(stop_sequences)]

    # Seed
    seed = kwargs.get("seed")
    if seed is not None:
        attributes["request_seed"] = int(seed)
    elif hasattr(args[0] if args else None, "seed"):
        seed = getattr(args[0], "seed", None)
        if seed is not None:
            attributes["request_seed"] = int(seed)
