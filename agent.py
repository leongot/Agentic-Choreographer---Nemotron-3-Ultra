"""
agent.py — agente ReAct usato in modalità supervisionata con OpenRouter/Nemotron.

"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from prompts import UNIVERSAL_PROMPT
from tools import all_tools

MODEL_NAME = "nvidia/nemotron-3-ultra-550b-a55b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY non trovata. Impostala con: "
        'setx OPENROUTER_API_KEY "la_tua_api_key"'
    )

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    max_tokens=1024,
    api_key=api_key,
    max_retries=0,
    base_url=OPENROUTER_BASE_URL,
    timeout=120,
    default_headers={
        "HTTP-Referer": "http://127.0.0.1:5000",
        "X-Title": "Agentic Choreography Thesis Project",
    },
)

trimmer = trim_messages(
    max_tokens=6000,
    token_counter=count_tokens_approximately,
    strategy="last",
    include_system=True,
    allow_partial=False,
    start_on="human",
)


def pre_model_hook(state):
    trimmed_messages = trimmer.invoke(state["messages"])
    return {"messages": trimmed_messages}


memory = MemorySaver()

PROMPT_FINALE_AGENTE = UNIVERSAL_PROMPT if isinstance(UNIVERSAL_PROMPT, str) else str(UNIVERSAL_PROMPT)

agent = create_react_agent(
    llm,
    tools=all_tools,
    prompt=PROMPT_FINALE_AGENTE,
    checkpointer=memory,
    pre_model_hook=pre_model_hook,
)
