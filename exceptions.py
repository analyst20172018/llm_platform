"""Public exceptions for request configuration errors."""


class RequestValidationError(ValueError):
    """Invalid request settings, distinct from provider and transport failures.

    ``code`` is machine-readable; ``parameters`` names the conflicting settings.
    Only a safe message and parameter names belong here, never request contents.
    """

    def __init__(self, message: str, *, code: str, parameters: tuple[str, ...]):
        super().__init__(message)
        self.code = code
        self.parameters = parameters
