import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from app.core.config import settings
from app.schemas import (
    PedagogyDecision,
    TeachingStrategy,
    DifficultyLevel,
    PresentationMode,
    GeneratedLesson,
    LessonBlock,
    LessonBlockType,
    LearnerStage,
    OnboardingInputProfile,
    LearnerIntelligenceReport,
    SkillAnalysis,
    KnowledgeAnalysis,
    LearningStyleProfile,
    CareerGoalProfile,
    MotivationProfile,
    ReadinessAssessment,
    ReadinessTier,
)


class LLMProviderError(Exception):
    """Application-level controlled error for LLM provider connectivity or rate-limit issues."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class LLMParseError(LLMProviderError):
    """Application-level controlled error when LLM output cannot be safely parsed or validated."""

    def __init__(self, message: str, raw_content: Optional[str] = None, status_code: int = 502):
        super().__init__(message, status_code=status_code)
        self.raw_content = raw_content


def extract_json_payload(raw_text: str) -> str:
    """Defensively extracts JSON substring from LLM response text, stripping markdown blocks or conversational text."""
    if not raw_text or not isinstance(raw_text, str):
        raise LLMParseError("LLM returned empty or non-string response.")

    text = raw_text.strip()

    # Strip markdown code blocks like ```json ... ``` or ``` ... ```
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()

    # Locate outermost JSON object braces
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace : last_brace + 1].strip()

    return text


def normalize_intelligence_dict(data: Dict[str, Any], profile: OnboardingInputProfile) -> Dict[str, Any]:
    """Defensively repairs and normalizes key names, aliases, and missing optional structures in LLM output

    WITHOUT silently fabricating critical learner ground truth.
    """
    if not isinstance(data, dict):
        raise LLMParseError("Normalized LLM output must be a dictionary object.")

    # 1. Enforce ground-truth learner metadata
    data["learner_id"] = profile.learner_id
    data["stage"] = profile.stage.value
    data["target_role"] = profile.target_role

    # 2. Normalize top-level section key aliases (camelCase / synonyms)
    key_mapping = {
        "skillAnalysis": "skill_analysis",
        "skills": "skill_analysis",
        "skill_baseline": "skill_analysis",
        "knowledgeAnalysis": "knowledge_analysis",
        "knowledge": "knowledge_analysis",
        "conceptual_analysis": "knowledge_analysis",
        "learningStyle": "learning_style",
        "learning_style_profile": "learning_style",
        "style": "learning_style",
        "careerGoals": "career_goals",
        "career_goals_profile": "career_goals",
        "career": "career_goals",
        "motivationProfile": "motivation",
        "motivation_profile": "motivation",
        "readinessAssessment": "readiness",
        "readiness_assessment": "readiness",
        "executiveSummary": "executive_summary",
        "summary": "executive_summary",
    }
    for old_k, new_k in key_mapping.items():
        if old_k in data and new_k not in data:
            data[new_k] = data.pop(old_k)

    # 3. Defensive defaults for missing sections
    if not isinstance(data.get("skill_analysis"), dict):
        data["skill_analysis"] = {}
    if not isinstance(data.get("knowledge_analysis"), dict):
        data["knowledge_analysis"] = {}
    if not isinstance(data.get("learning_style"), dict):
        data["learning_style"] = {}
    if not isinstance(data.get("career_goals"), dict):
        data["career_goals"] = {}
    if not isinstance(data.get("motivation"), dict):
        data["motivation"] = {}
    if not isinstance(data.get("readiness"), dict):
        data["readiness"] = {}

    # 4. Normalize nested aliases in skill_analysis
    sa = data["skill_analysis"]
    if "strengths" in sa and "core_strengths" not in sa:
        sa["core_strengths"] = sa.pop("strengths")
    if "gaps" in sa and "critical_skill_gaps" not in sa:
        sa["critical_skill_gaps"] = sa.pop("gaps")
    if "proficiency" in sa and "proficiency_level" not in sa:
        sa["proficiency_level"] = sa.pop("proficiency")
    if "summary" in sa and "baseline_summary" not in sa:
        sa["baseline_summary"] = sa.pop("summary")

    # 5. Normalize nested aliases in knowledge_analysis
    ka = data["knowledge_analysis"]
    if "topics" in ka and "recommended_foundation_topics" not in ka:
        ka["recommended_foundation_topics"] = ka.pop("topics")
    if "foundation_topics" in ka and "recommended_foundation_topics" not in ka:
        ka["recommended_foundation_topics"] = ka.pop("foundation_topics")

    # 6. Normalize nested aliases in career_goals
    cg = data["career_goals"]
    cg["target_role"] = profile.target_role
    if "milestones" in cg and "key_milestones" not in cg:
        cg["key_milestones"] = cg.pop("milestones")
    if "competencies" in cg and "high_priority_competencies" not in cg:
        cg["high_priority_competencies"] = cg.pop("competencies")
    if "alignment_score" in cg and "role_alignment_score" not in cg:
        cg["role_alignment_score"] = cg.pop("alignment_score")
    if "timeline" in cg and "estimated_timeline_months" not in cg:
        cg["estimated_timeline_months"] = cg.pop("timeline")

    # 7. Normalize nested aliases in motivation
    mo = data["motivation"]
    if "driver" in mo and "primary_driver" not in mo:
        mo["primary_driver"] = mo.pop("driver")
    if "hooks" in mo and "engagement_hooks" not in mo:
        mo["engagement_hooks"] = mo.pop("hooks")
    if "triggers" in mo and "potential_frustration_triggers" not in mo:
        mo["potential_frustration_triggers"] = mo.pop("triggers")
    if "advice" in mo and "resilience_advice" not in mo:
        mo["resilience_advice"] = mo.pop("advice")

    # 8. Normalize nested aliases in readiness
    rd = data["readiness"]
    if "score" in rd and "overall_readiness_score" not in rd:
        rd["overall_readiness_score"] = rd.pop("score")
    if "tier" in rd and "readiness_tier" not in rd:
        rd["readiness_tier"] = rd.pop("tier")
    if "entry_level" in rd and "recommended_entry_level" not in rd:
        rd["recommended_entry_level"] = rd.pop("entry_level")
    if "recommendations" in rd and "onboarding_recommendations" not in rd:
        rd["onboarding_recommendations"] = rd.pop("recommendations")

    # Executive summary fallback if omitted
    if not data.get("executive_summary"):
        data["executive_summary"] = (
            f"Onboarding analysis for {profile.learner_id} targeting '{profile.target_role}' "
            f"in stage {profile.stage.value}."
        )

    return data


class BaseLLMProvider(ABC):
    """Abstract base provider for generating structured pedagogical and learner intelligence."""

    @abstractmethod
    def generate_pedagogy_decision(self, prompt: str) -> PedagogyDecision:
        """Generate a structured pedagogy decision from a prompt."""
        pass

    @abstractmethod
    def generate_lesson(
        self,
        prompt: str,
        context_id: str,
        topic: str,
        pedagogy_decision: Optional[PedagogyDecision] = None,
    ) -> GeneratedLesson:
        """Generate structured lesson blocks from a prompt."""
        pass

    @abstractmethod
    def analyze_learner_onboarding(
        self, profile: OnboardingInputProfile, prompt: str
    ) -> LearnerIntelligenceReport:
        """Generate structured multi-dimensional Learner Intelligence from an onboarding profile."""
        pass


class MockLLMProvider(BaseLLMProvider):
    """Deterministic, contract-compliant provider for testing and zero-cost local development.
    Requires NO API keys and runs completely offline.
    """

    def generate_pedagogy_decision(self, prompt: str) -> PedagogyDecision:
        return PedagogyDecision(
            strategy=TeachingStrategy.FOUNDATIONAL,
            difficulty=DifficultyLevel.BEGINNER,
            presentation_mode=PresentationMode.VISUAL,
            explanation_depth="step-by-step",
            rationale="Deterministic mock: Learner requires foundational scaffolding with visual demonstrations.",
        )

    def generate_lesson(
        self,
        prompt: str,
        context_id: str,
        topic: str,
        pedagogy_decision: Optional[PedagogyDecision] = None,
    ) -> GeneratedLesson:
        if pedagogy_decision is None:
            pedagogy_decision = self.generate_pedagogy_decision(prompt)

        return GeneratedLesson(
            context_id=context_id,
            topic=topic,
            pedagogy_decision=pedagogy_decision,
            blocks=[
                LessonBlock(
                    type=LessonBlockType.OBJECTIVE,
                    title="Learning Objective",
                    content=f"Understand the fundamental mechanics and efficiency of {topic}.",
                ),
                LessonBlock(
                    type=LessonBlockType.EXPLANATION,
                    title="Core Concept",
                    content=f"This structured lesson explains {topic} using step-by-step conceptual breakdowns.",
                ),
                LessonBlock(
                    type=LessonBlockType.WORKED_EXAMPLE,
                    title="Walkthrough Example",
                    content=f"A concrete, beginner-friendly walkthrough demonstrating {topic} in practice.",
                ),
                LessonBlock(
                    type=LessonBlockType.VISUAL_SPEC,
                    title="Visual Representation Spec",
                    content="Visual layout instructions for LessonRenderer.",
                    metadata={"renderer": "diagram", "diagram_type": "flowchart"},
                ),
                LessonBlock(
                    type=LessonBlockType.PRACTICE_TASK,
                    title="Check for Understanding",
                    content=f"Apply your understanding of {topic} to solve a simple challenge.",
                    metadata={
                        "task_type": "mcq",
                        "options": ["A", "B", "C"],
                        "answer": "B",
                    },
                ),
            ],
        )

    def analyze_learner_onboarding(
        self, profile: OnboardingInputProfile, prompt: str
    ) -> LearnerIntelligenceReport:
        ratings = [s.self_rating for s in profile.declared_skills] if profile.declared_skills else [2]
        avg_rating = sum(ratings) / len(ratings)

        # Stage-calibrated profiling
        if profile.stage == LearnerStage.MASTER:
            prof_level = "advanced" if avg_rating >= 3.0 else "intermediate"
            readiness_tier = ReadinessTier.HIGH if avg_rating >= 3.5 else ReadinessTier.MODERATE
            readiness_score = min(0.95, max(0.65, round(avg_rating / 5.0 + 0.15, 2)))
            entry_level = "advanced" if avg_rating >= 3.5 else "intermediate"
            primary_driver = "academic_excellence"
            pacing = "accelerated"
        elif profile.stage == LearnerStage.GRADUATE:
            prof_level = "intermediate" if avg_rating >= 2.5 else "beginner"
            readiness_tier = ReadinessTier.MODERATE if avg_rating >= 2.5 else ReadinessTier.NEEDS_SCAFFOLDING
            readiness_score = min(0.85, max(0.45, round(avg_rating / 5.0, 2)))
            entry_level = "intermediate" if avg_rating >= 3.0 else "foundational"
            primary_driver = "career_transition"
            pacing = "standard"
        else:  # BACHELOR
            if avg_rating >= 3.8:
                prof_level = "advanced"
                readiness_tier = ReadinessTier.HIGH
                readiness_score = 0.88
                entry_level = "advanced"
            elif avg_rating >= 2.5:
                prof_level = "intermediate"
                readiness_tier = ReadinessTier.MODERATE
                readiness_score = 0.68
                entry_level = "intermediate"
            else:
                prof_level = "beginner"
                readiness_tier = ReadinessTier.NEEDS_SCAFFOLDING
                readiness_score = 0.42
                entry_level = "foundational"
            primary_driver = "project_creation"
            pacing = "scaffolded" if prof_level == "beginner" else "standard"

        top_skills = [s.skill_name for s in profile.declared_skills if s.self_rating >= 3]
        skill_gaps = [
            f"Advanced {profile.target_role} Architecture",
            "Production Testing & Quality Automation",
            "Scalable Systems Design",
        ]

        return LearnerIntelligenceReport(
            learner_id=profile.learner_id,
            stage=profile.stage,
            target_role=profile.target_role,
            executive_summary=(
                f"Learner {profile.learner_id} is in {profile.stage.value} stage pursuing '{profile.target_role}'. "
                f"Demonstrates {prof_level} baseline competency ({avg_rating:.1f}/5 avg) with {profile.preferences.preferred_medium} learning preference."
            ),
            skill_analysis=SkillAnalysis(
                baseline_summary=f"Demonstrated self-assessed average competency of {avg_rating:.1f}/5 across {len(profile.declared_skills)} declared skills.",
                proficiency_level=prof_level,
                core_strengths=top_skills or ["Foundational Problem Solving"],
                critical_skill_gaps=skill_gaps,
            ),
            knowledge_analysis=KnowledgeAnalysis(
                conceptual_depth="foundational" if prof_level == "beginner" else "applied",
                prerequisite_health="needs_remediation" if prof_level == "beginner" else "solid",
                theoretical_vs_applied_balance=(
                    "Prefers practical project-building with scaffolding"
                    if profile.preferences.practical_vs_theory_ratio >= 0.5
                    else "Prefers conceptual foundations first"
                ),
                recommended_foundation_topics=[
                    "Data Structures & Algorithmic Thinking",
                    "API Design Principles",
                    f"{profile.target_role} Core Fundamentals",
                ],
            ),
            learning_style=LearningStyleProfile(
                dominant_modality=profile.preferences.preferred_medium,
                secondary_modality="interactive",
                recommended_pacing=pacing,
                feedback_cadence="Immediate feedback after every worked practice task",
                content_format_priorities=["visual_spec", "worked_example", "practice_task"],
            ),
            career_goals=CareerGoalProfile(
                target_role=profile.target_role,
                role_alignment_score=round(min(1.0, max(0.2, avg_rating / 5.0)), 2),
                key_milestones=[
                    f"Milestone 1: Core {profile.target_role} foundations",
                    "Milestone 2: Guided end-to-end project implementation",
                    "Milestone 3: Advanced domain design and optimization",
                ],
                high_priority_competencies=[
                    "Modular System Design",
                    "Clean Code & Error Handling",
                    "Performance Profiling",
                ],
                estimated_timeline_months=max(3, int(300 / max(1, profile.preferences.weekly_hours * 4))),
            ),
            motivation=MotivationProfile(
                primary_driver=primary_driver,
                intrinsic_vs_extrinsic="balanced",
                engagement_hooks=[
                    "Milestone-based interactive code builds",
                    "Real-world scenario simulation",
                ],
                potential_frustration_triggers=[
                    "Abstract theoretical lectures without immediate coding",
                    "Steep difficulty spikes in algorithmic concepts",
                ],
                resilience_advice="Break complex problems into verifiable 15-minute micro-objectives.",
            ),
            readiness=ReadinessAssessment(
                overall_readiness_score=readiness_score,
                readiness_tier=readiness_tier,
                recommended_entry_level=entry_level,
                onboarding_recommendations=[
                    "Complete foundational prerequisite refresher",
                    "Set up first guided portfolio project milestone",
                    "Schedule regular weekly study sprints",
                ],
            ),
        )


class GroqLLMProvider(BaseLLMProvider):
    """Live LLM provider backed by Groq Cloud for fast inference.
    Only instantiated when live Groq is explicitly configured.
    """

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMProviderError("Groq API key is required to instantiate GroqLLMProvider.", status_code=500)
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key, timeout=25.0)
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Groq client: {str(e)}", status_code=500)
        self.model = model

    def _safe_chat_completion(self, messages: list, max_tokens: int, temperature: float) -> str:
        """Executes chat completion with robust error handling for rate limits, timeouts, and API errors."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as err:
            err_type = type(err).__name__
            err_msg = str(err)
            if "RateLimit" in err_type or "429" in err_msg:
                raise LLMProviderError(
                    "Groq API rate limit exceeded (HTTP 429). Please retry shortly or use mock provider.",
                    status_code=429,
                )
            if "Timeout" in err_type or "Connection" in err_type:
                raise LLMProviderError(
                    f"Groq API connection or timeout failure: {err_msg}",
                    status_code=504,
                )
            raise LLMProviderError(
                f"Groq LLM provider call failed ({err_type}): {err_msg}",
                status_code=502,
            )

    def generate_pedagogy_decision(self, prompt: str) -> PedagogyDecision:
        system_instruction = (
            "You are an expert pedagogy engine for UniOS. "
            "Analyze the learner context and topic, then output strictly valid JSON matching this schema:\n"
            "{\n"
            '  "strategy": "foundational" | "reinforcement" | "advancement" | "remediation",\n'
            '  "difficulty": "beginner" | "intermediate" | "advanced",\n'
            '  "presentation_mode": "traditional" | "visual" | "story" | "simulation" | "animation" | "interactive",\n'
            '  "explanation_depth": "high-level" | "step-by-step" | "deep-dive",\n'
            '  "rationale": "<concise explanation of pedagogical reasoning>"\n'
            "}"
        )
        raw_content = self._safe_chat_completion(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            max_tokens=350,
            temperature=settings.LLM_TEMPERATURE,
        )
        json_text = extract_json_payload(raw_content)
        try:
            data = json.loads(json_text)
            return PedagogyDecision.model_validate(data)
        except Exception as e:
            raise LLMParseError(f"Failed to parse pedagogy decision from LLM: {str(e)}", raw_content=raw_content)

    def generate_lesson(
        self,
        prompt: str,
        context_id: str,
        topic: str,
        pedagogy_decision: Optional[PedagogyDecision] = None,
    ) -> GeneratedLesson:
        if pedagogy_decision is None:
            pedagogy_decision = self.generate_pedagogy_decision(prompt)

        system_instruction = (
            "You are a structured lesson generation engine for UniOS. "
            "Generate an ordered list of 3-5 structured learning blocks.\n"
            "CRITICAL BOUNDARY: Under NO circumstances should you output raw HTML, JSX, or React UI code.\n"
            "Output strictly valid JSON matching this schema:\n"
            "{\n"
            '  "blocks": [\n'
            "    {\n"
            '      "type": "objective" | "explanation" | "worked_example" | "visual_spec" | "practice_task",\n'
            '      "title": "<block title>",\n'
            '      "content": "<structured block text content>",\n'
            '      "metadata": {}\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        raw_content = self._safe_chat_completion(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            max_tokens=850,
            temperature=settings.LLM_TEMPERATURE,
        )
        json_text = extract_json_payload(raw_content)
        try:
            data = json.loads(json_text)
            blocks = [LessonBlock.model_validate(b) for b in data.get("blocks", [])]
            return GeneratedLesson(
                context_id=context_id,
                topic=topic,
                pedagogy_decision=pedagogy_decision,
                blocks=blocks,
            )
        except Exception as e:
            raise LLMParseError(f"Failed to parse lesson blocks from LLM: {str(e)}", raw_content=raw_content)

    def analyze_learner_onboarding(
        self, profile: OnboardingInputProfile, prompt: str
    ) -> LearnerIntelligenceReport:
        system_instruction = (
            "You are the UniOS Learner Intelligence Engine (AI/ML-2). "
            "Analyze the learner onboarding profile across all 6 core dimensions: "
            "1. Skill Analysis, 2. Knowledge Analysis, 3. Learning Style, "
            "4. Career Goals, 5. Motivation, 6. Readiness. "
            "Output strictly valid JSON adhering to the LearnerIntelligenceReport schema. "
            "Target output under 850 tokens. Do NOT output HTML, JSX, or conversational text."
        )
        # Safe completion token limit around 800-900 tokens (850 target)
        raw_content = self._safe_chat_completion(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            max_tokens=850,
            temperature=settings.LLM_TEMPERATURE,
        )

        # 1. Defensively extract JSON substring
        json_text = extract_json_payload(raw_content)

        # 2. Parse JSON
        try:
            parsed_data = json.loads(json_text)
        except Exception as err:
            raise LLMParseError(
                f"LLM did not return valid JSON: {str(err)}",
                raw_content=raw_content,
            )

        # 3. Defensively repair aliases and maintain learner ground truth
        normalized_data = normalize_intelligence_dict(parsed_data, profile)

        # 4. Strict Pydantic validation
        try:
            return LearnerIntelligenceReport.model_validate(normalized_data)
        except Exception as err:
            raise LLMParseError(
                f"LearnerIntelligenceReport validation failed on repaired LLM output: {str(err)}",
                raw_content=raw_content,
            )


def get_llm_provider(force_mock: bool = False) -> BaseLLMProvider:
    """Factory function returning the active provider based on explicit configuration."""
    if force_mock:
        return MockLLMProvider()

    # Live Groq mode ONLY activates when explicitly configured
    if settings.LLM_PROVIDER.lower() == "groq" and bool(settings.GROQ_API_KEY):
        return GroqLLMProvider(
            api_key=settings.GROQ_API_KEY, model=settings.LLM_MODEL
        )
    return MockLLMProvider()
