from openai import OpenAI


class OpenAIClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def get_chat_completion(self, model, messages):
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=model,
        )
        return chat_completion
