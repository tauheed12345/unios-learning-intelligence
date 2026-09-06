from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class LearnerStage(str, Enum):
    BACHELOR = "bachelor"
    MASTER = "master"
    GRADUATE = "graduate"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LearnerState(BaseModel):
    """Represents the current knowledge, skill, and behavioral state of the learner."""

    learner_id: str = "learner_001"
    stage: LearnerStage = LearnerStage.BACHELOR
    academic_program: str = "Computer Science"
    current_semester: Optional[int] = None
    concept_mastery: Dict[str, float] = Field(
        default_factory=dict,
        description="Concept name mapped to mastery score between 0.0 and 1.0",
    )
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    weak_topics: List[str] = Field(default_factory=list)
    strong_topics: List[str] = Field(default_factory=list)
    career_goal: Optional[str] = None
    learning_preference: Optional[str] = "visual"
