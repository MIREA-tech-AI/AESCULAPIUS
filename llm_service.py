import openai

from settings_loader import AppSettings


def make_client(settings: AppSettings) -> openai.OpenAI:
    """Клиент OpenAI-совместимого API."""
    return openai.OpenAI(
        base_url=settings.llm.base_url,
        api_key=settings.llm.api_key,
    )


def summarize_medical_text(
    client: openai.OpenAI,
    settings: AppSettings,
    file_content: str,
) -> str:
    """Запрос суммаризации по настройкам из конфига и промптам."""
    gen = settings.generation
    user_text = f"{settings.user_intro}\n\n{file_content}"
    messages = [
        {"role": "system", "content": settings.system_prompt},
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ]
    extra_body: dict = {
        "top_k": gen.top_k,
        "chat_template_kwargs": {"enable_thinking": gen.enable_thinking},
    }
    response = client.chat.completions.create(
        model=settings.llm.model,
        messages=messages,
        temperature=gen.temperature,
        max_tokens=gen.max_tokens,
        extra_body=extra_body,
    )
    return response.choices[0].message.content or ""
