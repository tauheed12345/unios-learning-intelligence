from enum import Enum
from pydantic import BaseModel, Field


class TeachingStrategy(str, Enum):
    FOUNDATIONAL = "foundational"
    REINFORCEMENT = "reinforcement"
    ADVANCEMENT = "advancement"
    REMEDIATION = "remediation"


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class PresentationMode(str, Enum):
    TRADITIONAL = "traditional"
    VISUAL = "visual"
    STORY = "story"
    SIMULATION = "simulation"
    ANIMATION = "animation"
    INTERACTIVE = "interactive"


class PedagogyDecision(BaseModel):
    """The pedagogy strategy chosen for the learner and topic."""

    strategy: TeachingStrategy
    difficulty: DifficultyLevel
    presentation_mode: PresentationMode
    explanation_depth: str = Field(
        description="Depth of explanation: high-level, step-by-step, or deep-dive"
    )
    rationale: str = Field(
        description="Pedagogical reasoning based on learner mastery and stage"
    )
