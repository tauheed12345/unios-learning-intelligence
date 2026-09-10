import re
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from app.schemas.learner import LearnerStage, LearnerIdentity


class SkillCategory(str, Enum):
    """Categorization of technical and domain skills."""
    PROGRAMMING = "programming"
    FRAMEWORK = "framework"
    THEORY = "theory"
    TOOLS = "tools"
    SOFT_SKILLS = "soft_skills"


class DeclaredSkill(BaseModel):
    """A skill self-declared by the learner during onboarding."""

    skill_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the skill, language, or concept (e.g. Python, Docker)",
    )
    self_rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Self-assessment score between 1 (novice) and 5 (expert)",
    )
    category: SkillCategory = Field(
        default=SkillCategory.PROGRAMMING,
        description="Functional skill taxonomy category",
    )

    @field_validator("skill_name")
    @classmethod
    def validate_skill_name(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("skill_name must be a string.")
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("skill_name cannot be empty or whitespace only.")
        if len(cleaned) < 2:
            raise ValueError("skill_name must have at least 2 characters.")
        if len(cleaned) > 100:
            raise ValueError("skill_name exceeds maximum length of 100 characters.")
        return cleaned


class LearningPreferencesInput(BaseModel):
    """Learner pedagogical and instructional preferences collected during onboarding."""

    preferred_medium: str = Field(
        default="interactive",
        description="Preferred style: visual, hands-on, theoretical, interactive, conversational",
    )
    pace: str = Field(
        default="moderate",
        description="Self-identified pace: fast-track, moderate, relaxed",
    )
    weekly_hours: int = Field(
        default=10,
        ge=1,
        le=80,
        description="Available weekly hours dedicated to learning (1 to 80)",
    )
    practical_vs_theory_ratio: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Ratio of practical project work vs theoretical conceptual study (0.0 to 1.0)",
    )

    @field_validator("preferred_medium", "pace")
    @classmethod
    def normalize_preference_strings(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Preference value must be a string.")
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("Preference value cannot be empty.")
        return cleaned


class LearnerProfileCreate(BaseModel):
    """Contract for submitting a new onboarding profile from UniOS Backend / KIE."""

    learner_id: str = Field(
        ...,
        description="Unique identifier of the learner from Backend/KIE",
    )
    stage: LearnerStage = Field(
        default=LearnerStage.BACHELOR,
        description="Current academic/career stage",
    )
    academic_program: Optional[str] = Field(
        default="Computer Science",
        description="Academic degree or department",
    )
    current_semester: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Current enrolled semester",
    )
    target_role: str = Field(
        ...,
        min_length=2,
        description="Desired target career role (e.g. Fullstack Engineer, ML Architect)",
    )
    declared_skills: List[DeclaredSkill] = Field(
        default_factory=list,
        description="Self-declared technical baseline skills",
    )
    interests: List[str] = Field(
        default_factory=list,
        description="Key domain interests and learning curiosities",
    )
    preferences: LearningPreferencesInput = Field(
        default_factory=LearningPreferencesInput,
        description="Cognitive and study preferences",
    )
    motivation_statement: Optional[str] = Field(
        default=None,
        description="Learner's primary stated motivation or career transition driver",
    )
    prior_projects_summary: Optional[str] = Field(
        default=None,
        description="Summary of past hands-on engineering or academic projects",
    )

    @field_validator("learner_id")
    @classmethod
    def validate_learner_id(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("learner_id must be a string.")
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("learner_id cannot be empty or whitespace.")
        if not re.match(r"^[a-zA-Z0-9_\-\.:@]+$", cleaned):
            raise ValueError("learner_id contains invalid characters.")
        return cleaned

    @field_validator("target_role")
    @classmethod
    def validate_target_role(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("target_role must be a string.")
        cleaned = v.strip()
        if len(cleaned) < 2:
            raise ValueError("target_role must contain at least 2 characters.")
        return cleaned


class LearnerProfileSummary(BaseModel):
    """Compact summary of a learner profile for overview, listings, and logging."""

    learner_id: str
    stage: LearnerStage
    target_role: str
    declared_skills_count: int = 0
    top_skills: List[str] = Field(default_factory=list)
    weekly_hours: int = 10
    preferred_medium: str = "interactive"


class LearnerProfileValidationResult(BaseModel):
    """Result returned by the lightweight /validate-profile preflight endpoint."""

    is_valid: bool = True
    learner_id: str
    stage: LearnerStage = LearnerStage.BACHELOR
    target_role: str
    issues: List[str] = Field(
        default_factory=list,
        description="List of detected validation problems or profile gaps",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable suggestions to improve onboarding data quality",
    )
    declared_skills_count: int = 0
    estimated_readiness_indicator: str = Field(
        default="pending_ai_analysis",
        description="Heuristic readiness indicator prior to full LLM analysis",
    )


class OnboardingInputProfile(BaseModel):
    """Normalized onboarding profile ingested from UniOS Backend / KIE for AI/ML-2 intelligence analysis."""

    learner_id: str = Field(
        ...,
        description="Unique identifier of the learner (UUID or clean handle)",
    )
    identity: Optional[LearnerIdentity] = Field(
        default=None,
        description="Optional structured identity metadata consumed from Backend",
    )
    stage: LearnerStage = Field(
        default=LearnerStage.BACHELOR,
        description="Academic development stage",
    )
    academic_program: str = Field(
        default="Computer Science",
        description="Current enrolled degree or department",
    )
    current_semester: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Current enrolled semester",
    )
    target_role: str = Field(
        ...,
        min_length=2,
        description="Target engineering or academic role",
    )
    declared_skills: List[DeclaredSkill] = Field(
        default_factory=list,
        description="Baseline skills self-assessed by learner",
    )
    interests: List[str] = Field(
        default_factory=list,
        description="Topics and domain areas of interest",
    )
    preferences: LearningPreferencesInput = Field(
        default_factory=LearningPreferencesInput,
        description="Study schedule and cognitive preferences",
    )
    motivation_statement: Optional[str] = Field(
        default=None,
        description="Learner's statement of intent or motivation",
    )
    prior_projects_summary: Optional[str] = Field(
        default=None,
        description="Past project or coursework summary",
    )

    @field_validator("learner_id")
    @classmethod
    def validate_learner_id(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("learner_id must be a string.")
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("learner_id cannot be empty or whitespace.")
        if len(cleaned) > 128:
            raise ValueError("learner_id exceeds maximum length of 128 characters.")
        if not re.match(r"^[a-zA-Z0-9_\-\.:@]+$", cleaned):
            raise ValueError("learner_id contains invalid characters.")
        return cleaned

    @field_validator("target_role")
    @classmethod
    def validate_target_role(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("target_role must be a string.")
        cleaned = v.strip()
        if len(cleaned) < 2:
            raise ValueError("target_role must contain at least 2 characters.")
        return cleaned

    @field_validator("interests", mode="before")
    @classmethod
    def sanitize_interests(cls, v: Optional[List[str]]) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return []

    def to_summary(self) -> LearnerProfileSummary:
        """Helper to create a LearnerProfileSummary from this input profile."""
        top_skills = [s.skill_name for s in self.declared_skills if s.self_rating >= 4]
        return LearnerProfileSummary(
            learner_id=self.learner_id,
            stage=self.stage,
            target_role=self.target_role,
            declared_skills_count=len(self.declared_skills),
            top_skills=top_skills,
            weekly_hours=self.preferences.weekly_hours,
            preferred_medium=self.preferences.preferred_medium,
        )
