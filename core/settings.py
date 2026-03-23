from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    DEFAULT_LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0
    
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    GROQ_TEMPERATURE: float
    
    GEMINI_API_KEY: str
    DATABASE_URL: str
    READ_DATABASE_URL: str
    
    E2B_API_KEY: str
    
    LANGSMITH_TRACING: str
    LANGSMITH_ENDPOINT: str
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str
    
    

    class Config:
        env_file = ".env"


settings = Settings()