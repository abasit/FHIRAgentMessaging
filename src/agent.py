import logging

import litellm
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text

logger = logging.getLogger("fhir_purple_agent")


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

        logger.info(f"Received message:\n{user_input}")

        self.messages.append({
            "role": "user",
            "content": user_input,
        })

        try:
            logger.debug("Calling LLM...")
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

            logger.info(f"Responding:\n{content}")

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=content))],
                name="Response",
            )

        except Exception as e:
            logger.error(f"Error during LLM call: {e}", exc_info=True)
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Error: {str(e)}"))],
                name="Error",
            )