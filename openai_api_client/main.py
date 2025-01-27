import os
from .client import OpenAIClient


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    model = "gpt-4o-mini"
    user_input = input("Enter your message: ")
    messages = [
        {
            "role": "user",
            "content": user_input,
        }
    ]
    openai_client = OpenAIClient(api_key)
    chat_completion = openai_client.get_chat_completion(model, messages)
    print(chat_completion.choices[0].message.content)
    print(chat_completion.choices)


if __name__ == "__main__":
    main()
