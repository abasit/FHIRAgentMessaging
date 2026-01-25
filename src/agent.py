"""
FHIR Purple Agent - LLM conversation handler.

Processes messages from the green agent and responds using an LLM.
Maintains conversation history for multi-turn interactions.
"""

import logging

import litellm
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text

logger = logging.getLogger("fhir_purple_agent")

SYSTEM_PROMPT = """You are a helpful AI assistant that can complete tasks using available tools.

When tools are available, use them to retrieve information before answering.
Provide clear, accurate answers based on the data you retrieve.
If you cannot find information, state this clearly rather than guessing.
Do not repeat the same action multiple times.
Follow any instructions for formatting the final answer."""

DEFAULT_MODEL = "openai/gpt-4o-mini"


class Agent:
    """Handles conversation with LLM, maintaining message history."""

    def __init__(self, context_id: str):
        self.context_id = context_id
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Process incoming message and respond using LLM."""
        user_input = get_message_text(message)
        logger.info(f"[Task {self.context_id}] Received message")
        logger.debug(f"[Task {self.context_id}] Input:\n{user_input}")

        self.messages.append({"role": "user", "content": user_input})

        try:
            response = litellm.completion(
                messages=self.messages,
                model=DEFAULT_MODEL,
                temperature=0.0,
                top_p=0,
                seed=0,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty response")

            self.messages.append({"role": "assistant", "content": content})
            logger.info(f"[Task {self.context_id}] Responded")
            logger.debug(f"[Task {self.context_id}] Output:\n{content}")

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=content))],
                name="Response",
            )

        except Exception as e:
            logger.error(f"[Task {self.context_id}] Error: {e}", exc_info=True)
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Failed to complete task: Internal error."))],
                name="Error",
            )