class LLMError(RuntimeError):
    pass


class LLMConfigurationError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMProviderError(LLMError):
    pass


class LLMCapabilityError(LLMError):
    pass
