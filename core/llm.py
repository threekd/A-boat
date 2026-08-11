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
        if provider == "modelscope":
            print("正在使用自定义的ModelScope Provider")
            self.provider = "modelscope"

            self.apikey = api_key or os.getenv("MODELSCOPE_API_KEY")
            self.base_url = base_url or os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1/")
            if not self.apikey:
                raise ValueError("ModelScope API key not found. Please set MODELSCOPE_API_KEY environment variable.")
            self.model = model or os.getenv("LLM_MODEL_ID") or "Qwen/Qwen2.5-VL-72B-Instruct"
            self.temperature = kwargs.get("temperature", 0.7)
            self.max_tokens = kwargs.get("max_tokens")
            self.timeout = kwargs.get("timeout", 60)

            self._client = OpenAI(api_key=self.apikey, base_url=self.base_url, timeout=self.timeout)
        elif provider == "deepseek":
            print("正在使用DeepSeek Provider")
            self.provider = "deepseek"

            self.apikey = api_key or os.getenv("DEEPSEEK_API_KEY")
            self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            if not self.apikey:
                raise ValueError("DeepSeek API key not found. Please set DEEPSEEK_API_KEY environment variable.")
            self.model = model  or "deepseek-v4-flash"
            self.temperature = kwargs.get("temperature", 0.7)
            self.max_tokens = kwargs.get("max_tokens")
            self.timeout = kwargs.get("timeout", 60)

            self._client = OpenAI(api_key=self.apikey, base_url=self.base_url, timeout=self.timeout)
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