# import os
# from pydantic import BaseModel
# from deepeval.models import DeepEvalBaseLLM
# from langchain.chat_models import init_chat_model


# class GroqModel(DeepEvalBaseLLM):
#     def __init__(self, name="qwen/qwen3.8-27b", provider="groq"):
#         self.name = name
#         self.chat = init_chat_model(
#             name,
#             model_provider=provider,
#             api_key=os.environ["GROQ_API_KEY"],
#         )

#     def load_model(self):
#         return self.chat

#     def generate(self, prompt: str, schema: BaseModel = None):
#         chat = self.load_model()
#         if schema:
#             # DeepEval metrics often need structured (JSON) output
#             return chat.with_structured_output(schema).invoke(prompt)
#         return chat.invoke(prompt).content

#     async def a_generate(self, prompt: str, schema: BaseModel = None):
#         chat = self.load_model()
#         if schema:
#             return await chat.with_structured_output(schema).ainvoke(prompt)
#         return (await chat.ainvoke(prompt)).content

#     def get_model_name(self):
#         return f"{self.name} (groq)"

# eval_model = GroqModel()






# import os
# from pydantic import BaseModel
# from deepeval.models import DeepEvalBaseLLM
# from langchain.chat_models import init_chat_model

# class JudgeModel(DeepEvalBaseLLM):
#     def __init__(self, name, provider):
#         self.name = name
#         self.chat = init_chat_model(name, model_provider=provider)

#     def load_model(self):
#         return self.chat

#     def generate(self, prompt: str, schema: BaseModel = None):
#         chat = self.load_model()
#         if schema:
#             return chat.with_structured_output(schema).invoke(prompt)
#         return chat.invoke(prompt).content

#     async def a_generate(self, prompt: str, schema: BaseModel = None):
#         chat = self.load_model()
#         if schema:
#             return await chat.with_structured_output(schema).ainvoke(prompt)
#         return (await chat.ainvoke(prompt)).content

#     def get_model_name(self):
#         return self.name

# eval_model = JudgeModel(name="gemini-3.5-flash-lite", provider="google_genai")
 
#name="gemini-3.5-flash-lite", provider="google_genai"








import os
from pydantic import BaseModel
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception
from deepeval.models import DeepEvalBaseLLM
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter


def is_rate_limit(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


retry_on_limit = retry(
    retry=retry_if_exception(is_rate_limit),
    wait=wait_fixed(40),
    stop=stop_after_attempt(8),
    reraise=True,
)

# 0.2 requests/second = about 12 per minute, under the 15 limit
limiter = InMemoryRateLimiter(requests_per_second=0.2)


class JudgeModel(DeepEvalBaseLLM):
    def __init__(self, name, provider):
        self.name = name
        self.chat = init_chat_model(
            name,
            model_provider=provider,
            rate_limiter=limiter,
            max_retries=0,   # tenacity handles retries instead
        )

    def load_model(self):
        return self.chat

    @retry_on_limit
    def generate(self, prompt: str, schema: BaseModel = None):
        chat = self.load_model()
        if schema:
            return chat.with_structured_output(schema).invoke(prompt)
        return chat.invoke(prompt).content

    @retry_on_limit
    async def a_generate(self, prompt: str, schema: BaseModel = None):
        chat = self.load_model()
        if schema:
            return await chat.with_structured_output(schema).ainvoke(prompt)
        return (await chat.ainvoke(prompt)).content

    def get_model_name(self):
        return self.name

eval_model = JudgeModel(name="gemini-3.5-flash-lite", provider="google_genai")