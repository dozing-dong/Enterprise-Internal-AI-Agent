"""Custom exceptions for the API layer.

Defined in a dedicated module to avoid circular imports between app.py and the route modules.
"""


class RagException(Exception):
    """Expected business exception raised inside the RAG pipeline.

    status_code maps to an HTTP status code so callers can distinguish
    between error categories:
    - 400: invalid request parameters (client problem)
    - 503: downstream service (Bedrock / pgvector) unavailable
    - 500: unexpected internal error
    """

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
