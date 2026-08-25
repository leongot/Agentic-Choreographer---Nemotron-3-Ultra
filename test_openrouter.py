from __future__ import annotations

import os
from openai import OpenAI

MODEL_NAME = "nvidia/nemotron-3-ultra-550b-a55b:free"

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY non trovata. Impostala con: "
        'setx OPENROUTER_API_KEY "la_tua_api_key"'
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers={
        "HTTP-Referer": "http://127.0.0.1:5000",
        "X-Title": "Agentic Choreography Thesis Project",
    },
)

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "Rispondi sempre in italiano."},
        {"role": "user", "content": "Dimmi solo: OpenRouter con Nemotron funziona."},
    ],
    temperature=0.2,
    max_tokens=100,
)

print(response.choices[0].message.content)
