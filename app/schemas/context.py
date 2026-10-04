from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.learner import LearnerState
from app.schemas.memory_context import RelevantMemoryContext


class NormalizedLearningContext(BaseModel):
    """The normalized context payload passed to AI/ML-2 for pedagogy and generation.
    Supports both legacy context_id and canonical cross-service request_id.
    """

    context_id: str = "ctx_001"
    request_id: Optional[str] = Field(
        default=None,
        description="Canonical cross-service tracing ID (synced with context_id)",
    )
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
    academic_domain: Optional[str] = Field(
        default=None,
        description="Canonical academic domain (e.g. engineering, commerce, humanities, science, management, law)",
    )
    academic_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detailed academic context dictionary from Backend",
    )
    content_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Approved syllabus and reference material from Backend/retrieval",
    )
    performance: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Recent learner performance telemetry from Backend",
    )
    mastery: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Authoritative topic mastery records from Backend",
    )
    goals: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Active academic and career goals",
    )
    request_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Interpreted request context from AI/ML-1",
    )

    def model_post_init(self, __context):
        # Sync request_id and context_id
        if self.request_id is None and self.context_id:
            self.request_id = self.context_id
        elif self.context_id == "ctx_001" and self.request_id:
            self.context_id = self.request_id

        # Sync academic_domain to learner_state if specified
        if self.academic_domain and self.learner_state:
            if not self.learner_state.academic_domain:
                self.learner_state.academic_domain = self.academic_domain

        # Extract content references from content_context if curriculum_references is empty
        if self.content_context and not self.curriculum_references:
            refs = self.content_context.get("relevant_content", [])
            if isinstance(refs, list):
                extracted = []
                for item in refs:
                    if isinstance(item, str):
                        extracted.append(item)
                    elif isinstance(item, dict) and "text" in item:
                        extracted.append(item["text"])
                    elif isinstance(item, dict) and "excerpt" in item:
                        extracted.append(item["excerpt"])
                if extracted:
                    self.curriculum_references = extracted

