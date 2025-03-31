class Config:
    """
    Shared static config for LangChain OTel instrumentation.
    """

    exception_logger = None
    # to globally suppress instrumentation
    _suppress_instrumentation = False

    @classmethod
    def suppress_instrumentation(cls, suppress: bool = True):
        cls._suppress_instrumentation = suppress

    @classmethod
    def is_instrumentation_suppressed(cls) -> bool:
        return cls._suppress_instrumentation
