"""FastAPI 服务启动入口。"""

import uvicorn


def start_api() -> None:
    """启动 FastAPI 服务。"""
    # host=127.0.0.1 表示只在本机开放，开发阶段更安全也更简单。
    # port=8000 是 FastAPI 常见默认端口，方便记忆。
    # reload=True 表示修改代码后自动重启，适合本地开发。
    uvicorn.run("backend.api.app:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    start_api()
