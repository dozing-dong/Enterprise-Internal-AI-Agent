"""API 层自定义异常。

集中定义在独立模块中，避免 app.py 与路由模块之间的循环导入。
"""


class RagException(Exception):
    """RAG 流程中可预期的业务异常。

    status_code 用于映射 HTTP 状态码，让调用方能区分不同的错误类型：
    - 400: 请求参数有误（客户端问题）
    - 503: 下游服务（Bedrock / pgvector）不可用
    - 500: 未预期的内部错误
    """

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
