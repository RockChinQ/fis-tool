import os
from typing import Optional

import openai
import inquirer


chat_model: Optional[openai.OpenAI] = None


def init_model():
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        api_key = inquirer.text(
            message="请输入 OpenAI API Key",
            validate=lambda _, x: len(x) > 0,
        )

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")

    return openai.OpenAI(api_key=api_key, base_url=base_url)


def ask_question(
    question
):
    global chat_model

    if not chat_model:
        chat_model = init_model()

    response = chat_model.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        stream=True
    )

    for chunk in response:
        yield chunk.choices[0].delta.content
