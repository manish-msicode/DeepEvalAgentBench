from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    model="gemini-3.5-flash-lite", 
    model_provider="google_genai", 
    temperature=0,
    # reasoning_effort="low"
)

