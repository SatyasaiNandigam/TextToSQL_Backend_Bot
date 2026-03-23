import yaml
from langchain_core.prompts import ChatPromptTemplate


def load_prompt(path: str):

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    messages = [
        (msg["role"], msg["content"])
        for msg in config["messages"]
    ]

    return ChatPromptTemplate.from_messages(messages)