def _serialize_object(obj, max_depth=3, current_depth=0):
    """
    Intelligently serialize an object to a more meaningful representation
    """
    if current_depth > max_depth:
        return f"<{type(obj).__name__}:max_depth_reached>"

    # Handle basic JSON-serializable types
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        try:
            return [
                _serialize_object(item, max_depth, current_depth + 1)
                for item in obj[:10]
            ]  # Limit to first 10 items
        except Exception:
            return f"<{type(obj).__name__}:length={len(obj)}>"

    # Handle dictionaries
    if isinstance(obj, dict):
        try:
            serialized = {}
            for key, value in list(obj.items())[:10]:  # Limit to first 10 items
                serialized[str(key)] = _serialize_object(
                    value, max_depth, current_depth + 1
                )
            return serialized
        except Exception:
            return f"<dict:keys={len(obj)}>"

    # Handle common object types with meaningful attributes
    try:
        # Check class attributes first
        class_attrs = {}
        for attr_name in dir(type(obj)):
            if (
                not attr_name.startswith("_")
                and not callable(getattr(type(obj), attr_name, None))
                and hasattr(obj, attr_name)
            ):
                try:
                    attr_value = getattr(obj, attr_name)
                    if not callable(attr_value):
                        class_attrs[attr_name] = _serialize_object(
                            attr_value, max_depth, current_depth + 1
                        )
                        if len(class_attrs) >= 5:  # Limit attributes
                            break
                except Exception:
                    continue

        # Check if object has a __dict__ with interesting attributes
        instance_attrs = {}
        if hasattr(obj, "__dict__"):
            obj_dict = obj.__dict__
            if obj_dict:
                # Extract meaningful attributes (skip private ones and callables)
                for key, value in obj_dict.items():
                    if not key.startswith("_") and not callable(value):
                        try:
                            instance_attrs[key] = _serialize_object(
                                value, max_depth, current_depth + 1
                            )
                            if len(instance_attrs) >= 5:  # Limit attributes
                                break
                        except Exception:
                            continue

        # Combine class and instance attributes
        all_attrs = {**class_attrs, **instance_attrs}

        if all_attrs:
            return {
                "__class__": type(obj).__name__,
                "__module__": getattr(type(obj), "__module__", "unknown"),
                "attributes": all_attrs,
            }

        # Special handling for specific types
        if hasattr(obj, "message") and hasattr(obj.message, "parts"):
            # Handle RequestContext-like objects
            try:
                parts_content = []
                for part in obj.message.parts:
                    if hasattr(part, "root") and hasattr(part.root, "text"):
                        parts_content.append(part.root.text)
                return {
                    "__class__": type(obj).__name__,
                    "message_content": parts_content,
                }
            except Exception:
                pass

        # Check for common readable attributes
        for attr in ["name", "id", "type", "value", "content", "text", "data"]:
            if hasattr(obj, attr):
                try:
                    attr_value = getattr(obj, attr)
                    if not callable(attr_value):
                        return {
                            "__class__": type(obj).__name__,
                            attr: _serialize_object(
                                attr_value, max_depth, current_depth + 1
                            ),
                        }
                except Exception:
                    continue

        # Fallback to class information
        return {
            "__class__": type(obj).__name__,
            "__module__": getattr(type(obj), "__module__", "unknown"),
            "__repr__": str(obj)[:100] + ("..." if len(str(obj)) > 100 else ""),
        }

    except Exception:
        # Final fallback
        return f"<{type(obj).__name__}:serialization_failed>"


def cameltosnake(camel_string: str) -> str:
    if not camel_string:
        return ""
    elif camel_string[0].isupper():
        return f"_{camel_string[0].lower()}{cameltosnake(camel_string[1:])}"
    else:
        return f"{camel_string[0]}{cameltosnake(camel_string[1:])}"


def camel_to_snake(s):
    if len(s) <= 1:
        return s.lower()

    return cameltosnake(s[0].lower() + s[1:])

