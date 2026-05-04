"""FastAPI service entry point."""

import uvicorn


def start_api() -> None:
    """Start the FastAPI service."""
    # host=127.0.0.1 binds to localhost only, which is safer and simpler for development.
    # port=8000 is the common FastAPI default and easy to remember.
    # reload=True auto-restarts on code changes, ideal for local development.
    uvicorn.run("backend.api.app:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    start_api()
