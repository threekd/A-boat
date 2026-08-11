"""配置管理"""
import os
from typing import Any, Dict, Optional
from pydantic import BaseModel

class Config(BaseModel):
    """HelloAgents 配置类"""

    # LLM configuration
    default_model: str = "deepseek-v4-flash"
    default_provider: str = "deepseek"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # System configuration
    debug: bool = False
    log_level: str = "INFO"

    # Other configurations
    max_history_length: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置"""
        max_tokens_str = os.getenv("MAX_TOKENS")
        return cls(
            debug=os.getenv("DEBUG", "False").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
            max_tokens=int(max_tokens_str) if max_tokens_str else None,
        )
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return self.model_dump()
    