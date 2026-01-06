import litellm
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

import logging

logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class Agent:
    def __init__(self):
        self.messages = [{
            "role": "system",
            "content": """You are a helpful AI assistant that can complete tasks using available tools via function calling.

When tools are available, use them to retrieve information before answering.
Provide clear, accurate answers based on the data you retrieve.
If you cannot find information, state this clearly rather than guessing.
Do not repeat the same action multiple times.
Follow any instructions as to the formatting of the final answer."""
        }]

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Process message and respond using LLM."""
        user_input = get_message_text(message)

        print(f"Purple agent received:\n{user_input}\n")

        self.messages.append({
            "role": "user",
            "content": user_input,
        })

        try:
            model_response = litellm.completion(
                messages=self.messages,
                model="openai/gpt-4o",
                temperature=0.0,
            )

            if not model_response.choices or not model_response.choices[0].message:
                raise ValueError("LLM returned empty response")

            content = model_response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned None content")

            self.messages.append({
                "role": "assistant",
                "content": content,
            })

            print(f"Purple agent responding:\n{content}\n")

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=content))],
                name="Response",
            )

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"Purple agent error: {error_msg}")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=error_msg))],
                name="Error",
            )