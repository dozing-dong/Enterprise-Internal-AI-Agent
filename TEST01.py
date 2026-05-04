"""Legacy entry point name kept for compatibility; forwards to FastAPI startup."""

from run_api import start_api


if __name__ == "__main__":
    start_api()
