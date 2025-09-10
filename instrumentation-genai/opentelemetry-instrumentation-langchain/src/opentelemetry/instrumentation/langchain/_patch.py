def patch_leaf_subclasses(base_class, method_name, wrapper):
    """
    Patches a method on leaf subclasses of a base class using wrapt's function wrapper pattern.
    
    Args:
        base_class: The base class whose leaf subclasses will be patched
        method_name: Name of the method to patch
        wrapper: A function that will be called with (wrapped, instance, args, kwargs)
                and should return the result of the wrapped function
    """
    all_subclasses = _get_all_subclasses(base_class)
    leaf_subclasses = _get_leaf_subclasses(all_subclasses)
    
    import wrapt
    import functools
    
    for subclass in leaf_subclasses:
        # Skip if the class doesn't have the method or it's not callable
        if not hasattr(subclass, method_name) or not callable(getattr(subclass, method_name)):
            continue
            
        # Get the original method
        original_method = getattr(subclass, method_name)
        
        # Skip if already patched (check for our specific attribute)
        if hasattr(original_method, '_opentelemetry_wrapped'):
            continue
            
        # Create a wrapper function that preserves the original method's signature
        @functools.wraps(original_method)
        def wrapped_method(*args, **kwargs):
            # The first argument is 'self' for instance methods
            instance = args[0] if args else None
            method_args = args[1:] if args else ()
            
            # Call the original method through the wrapper
            return wrapper(original_method, instance, method_args, kwargs)
            
        # Mark as wrapped to prevent double-wrapping
        wrapped_method._opentelemetry_wrapped = True
        
        # Store the original method in case we need it later
        wrapped_method._original_method = original_method
        
        # Replace the method on the class
        setattr(subclass, method_name, wrapped_method)


def _get_leaf_subclasses(all_subclasses):
    """
    Returns only the leaf classes (classes with no subclasses) from a set of classes.
    Args:
        all_subclasses: Set of classes to filter
    Returns:
        set: Classes that have no subclasses within the provided set
    """
    leaf_classes = set()
    for cls in all_subclasses:
        # A class is a leaf if no other class in the set is its subclass
        is_leaf = True
        for other_cls in all_subclasses:
            if other_cls != cls and issubclass(other_cls, cls):
                is_leaf = False
                break
        if is_leaf:
            leaf_classes.add(cls)
    return leaf_classes


def _get_all_subclasses(cls):
    """
    Gets all subclasses of a given class.
    Args:
        cls: The base class to find subclasses for
    Returns:
        set: All subclasses (direct and indirect) of the given class
    """
    subclasses = set()
    for subclass in cls.__subclasses__():
        subclasses.add(subclass)
        subclasses.update(_get_all_subclasses(subclass))
    return subclasses