from __future__ import annotations

from capsule_brain.llm.gateway import LLMGateway
from capsule_brain.llm.models import LLMRequest


class StructuredGoalDecomposer:
    def __init__(self, gateway: LLMGateway, model: str | None = None) -> None:
        self.gateway = gateway
        self.model = model

    async def __call__(self, goal_text: str) -> list[str]:
        data, _ = await self.gateway.generate_json(
            LLMRequest(
                model=self.model,
                response_format="json",
                temperature=0.2,
                system=(
                    "You are a goal decomposition engine. Return only a JSON object "
                    'with one field, "tasks", containing an ordered array of concise, '
                    "independently actionable task strings."
                ),
                prompt=f"Decompose this goal into executable tasks:\n{goal_text}",
            )
        )
        tasks = data.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("Goal decomposition response is missing tasks[]")
        cleaned = [str(task).strip() for task in tasks if str(task).strip()]
        if not cleaned:
            raise ValueError("Goal decomposition produced no tasks")
        return cleaned
