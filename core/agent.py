"""Agent 基类"""
from abc import ABC, abstractmethod
from typing import Any, Optional
from .message import Message
from .llm import MyLLM
from .config import Config

class Agent(ABC):
    """Agent 基类"""

    def __init__(
            self, 
            name: str,
            llm: MyLLM, 
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
        ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs: Any) -> str:
        """运行 Agent"""
        pass

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)

    def get_history(self) -> list[Message]:
        """获取消息历史记录"""
        return self._history.copy()

    def clear_history(self):
        """清空消息历史记录"""
        self._history.clear()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"