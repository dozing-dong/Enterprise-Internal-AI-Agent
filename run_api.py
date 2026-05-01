import uvicorn


if __name__ == "__main__":
    # 这里直接启动 FastAPI 服务。
    # host=127.0.0.1 表示只在本机开放，学习阶段更安全也更简单。
    # port=8000 是 FastAPI 常见默认端口，方便记忆。
    # reload=True 表示你修改代码后，服务会自动重启，适合开发学习。
    uvicorn.run("backend.api.app:app", host="127.0.0.1", port=8000, reload=True)
