"""兼容旧入口名，默认转发到 FastAPI 启动。"""

from run_api import start_api


if __name__ == "__main__":
    start_api()
