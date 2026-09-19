from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.learner import LearnerState
from app.schemas.memory_context import RelevantMemoryContext


class NormalizedLearningContext(BaseModel):
    """The normalized context payload passed to AI/ML-2 for pedagogy and generation."""

    context_id: str = "ctx_001"
    topic: str
    objective: str
    learner_state: LearnerState
    curriculum_references: List[str] = Field(
        default_factory=list,
        description="Grounded excerpts from syllabus or approved textbooks",
    )
    time_budget_minutes: Optional[int] = 15
    memory_context: Optional[RelevantMemoryContext] = Field(
        default=None,
        description="Optional scoped memory intelligence from AI/ML-2 Memory Engine",
    )
