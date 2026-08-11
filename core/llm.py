import os
from openai import OpenAI
from .HelloAgentsLLM import HelloAgentsLLM

class MyLLM(HelloAgentsLLM):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        **kwargs
    ):
        if provider is None:
            provider = self._auto_detect_provider(api_key, base_url)
        if provider == "modelscope":
            print("正在使用自定义的ModelScope Provider")
            self.provider = "modelscope"

            self.api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
            self.base_url = base_url or os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1/")
            if not self.api_key:
                raise ValueError("ModelScope API key not found. Please set MODELSCOPE_API_KEY environment variable.")
            self.model = model or os.getenv("LLM_MODEL_ID") or "Qwen/Qwen2.5-VL-72B-Instruct"
            self.temperature = kwargs.get("temperature", 0.7)
            self.max_tokens = kwargs.get("max_tokens")
            self.timeout = kwargs.get("timeout", 60)

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        elif provider == "deepseek":
            print("正在使用DeepSeek Provider")
            self.provider = "deepseek"

            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            if not self.api_key:
                raise ValueError("DeepSeek API key not found. Please set DEEPSEEK_API_KEY environment variable.")
            self.model = model  or "deepseek-v4-flash"
            self.temperature = kwargs.get("temperature", 0.7)
            self.max_tokens = kwargs.get("max_tokens")
            self.timeout = kwargs.get("timeout", 60)

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        else:
            # 其他 provider：透传给父类（注意父类参数名是 apiKey/baseUrl/timeout）
            init_kwargs = {}
            if model:
                init_kwargs["model"] = model
            if api_key:
                init_kwargs["apiKey"] = api_key
            if base_url:
                init_kwargs["baseUrl"] = base_url
            super().__init__(**init_kwargs, timeout=kwargs.get("timeout", 60))
    def _auto_detect_provider(self, api_key: str | None = None, base_url: str | None = None) -> str | None:
        # 自动检测 provider

        # 1. 检查特定提供商的环境变量 (最高优先级)
        if os.getenv("DEEPSEEK_API_KEY"): return "deepseek"
        if os.getenv("OPENAI_API_KEY"): return "openai"
        if os.getenv("MODELSCOPE_API_KEY"): return "modelscope"
        # 获取通用的环境变量
        actual_api_key = api_key or os.getenv("LLM_API_KEY")
        actual_base_url = base_url or os.getenv("LLM_BASE_URL")

        # 2. 根据 base_url 判断
        if actual_base_url:
            base_url_lower = actual_base_url.lower()
            if "api-inference.modelscope.cn" in base_url_lower: return "modelscope"
            if "api.deepseek.com" in base_url_lower: return "deepseek"
            if "open.bigmodel.cn" in base_url_lower: return "zhipu"
            if "localhost" in base_url_lower or "127.0.0.1" in base_url_lower:
                if ":11434" in base_url_lower: return "ollama"
                if ":8000" in base_url_lower: return "vllm"
                return "local" # 其他本地端口
            
        # 3. 根据 API 密钥格式辅助判断
        if actual_api_key:
            if actual_api_key.startswith("ms-"): return "modelscope"
            # ... 其他密钥格式判断

        # 4. 默认返回 'auto'，使用通用配置
        return "auto"
    def _resolve_credentials(self, api_key: str | None = None, base_url: str | None = None) -> tuple[str, str] | None:
        """根据provider解析API密钥和base_url"""
        if self.provider == "deepseek":
            resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            resolved_base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
            if not resolved_api_key:
                raise ValueError("DeepSeek API key not found. Please set DEEPSEEK_API_KEY environment variable.")
            return resolved_api_key, resolved_base_url
        if self.provider == "openai":
            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
            if not resolved_api_key:
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY or LLM_API_KEY environment variable.")
            return resolved_api_key, resolved_base_url

        elif self.provider == "modelscope":
            resolved_api_key = api_key or os.getenv("MODELSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api-inference.modelscope.cn/v1/"
            if not resolved_api_key:
                raise ValueError("ModelScope API key not found. Please set MODELSCOPE_API_KEY or LLM_API_KEY environment variable.")
            return resolved_api_key, resolved_base_url
        
