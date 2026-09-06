import json
import time
from abc import ABC, abstractmethod
from typing import Optional
from groq import Groq
from app.core.config import settings
from app.schemas import (
    PedagogyDecision,
    TeachingStrategy,
    DifficultyLevel,
    PresentationMode,
    GeneratedLesson,
    LessonBlock,
    LessonBlockType,
)


class BaseLLMProvider(ABC):
    """Abstract base provider for generating structured pedagogical intelligence."""

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


class MockLLMProvider(BaseLLMProvider):
    """Deterministic provider for testing and zero-cost local development."""

    def generate_pedagogy_decision(self, prompt: str) -> PedagogyDecision:
        print("[LLM-Provider: MOCK] Returning deterministic pedagogy decision...")
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
        print(f"[LLM-Provider: MOCK] Returning deterministic lesson for '{topic}'...")
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


class GroqLLMProvider(BaseLLMProvider):
    """Live LLM provider backed by Groq Cloud for fast inference."""

    def __init__(self, api_key: str, model: str):
        self.client = Groq(api_key=api_key, timeout=30.0)
        self.model = model

    def generate_pedagogy_decision(self, prompt: str) -> PedagogyDecision:
        print(f"\n=======================================================")
        print(f"[GROQ LLM] CALLING MODEL: '{self.model}' (Pedagogy Decision)")
        print(f"=======================================================")
        t0 = time.time()

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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=400,
        )
        elapsed = time.time() - t0
        raw_content = response.choices[0].message.content
        data = json.loads(raw_content)
        decision = PedagogyDecision.model_validate(data)

        print(f"[GROQ LLM] Pedagogy Decision generated in {elapsed:.2f}s:")
        print(f"  -> Strategy:          {decision.strategy.value}")
        print(f"  -> Difficulty:        {decision.difficulty.value}")
        print(f"  -> Presentation Mode: {decision.presentation_mode.value}")
        print(f"  -> Explanation Depth: {decision.explanation_depth}")
        print(f"  -> Rationale:         {decision.rationale[:120]}...")
        return decision

    def generate_lesson(
        self,
        prompt: str,
        context_id: str,
        topic: str,
        pedagogy_decision: Optional[PedagogyDecision] = None,
    ) -> GeneratedLesson:
        if pedagogy_decision is None:
            pedagogy_decision = self.generate_pedagogy_decision(prompt)

        print(f"\n=======================================================")
        print(f"[GROQ LLM] CALLING MODEL: '{self.model}' (Structured Lesson Blocks)")
        print(f"=======================================================")
        t0 = time.time()

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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=900,
        )
        elapsed = time.time() - t0
        raw_content = response.choices[0].message.content
        data = json.loads(raw_content)
        blocks = [LessonBlock.model_validate(b) for b in data["blocks"]]

        print(f"[GROQ LLM] Generated {len(blocks)} Structured Blocks in {elapsed:.2f}s:")
        for idx, b in enumerate(blocks, 1):
            print(f"  {idx}. [{b.type.value.upper()}] {b.title}")
        print(f"=======================================================\n")

        return GeneratedLesson(
            context_id=context_id,
            topic=topic,
            pedagogy_decision=pedagogy_decision,
            blocks=blocks,
        )


def get_llm_provider() -> BaseLLMProvider:
    """Factory function returning the active provider based on configuration."""
    if settings.LLM_PROVIDER.lower() == "groq" and settings.GROQ_API_KEY:
        return GroqLLMProvider(
            api_key=settings.GROQ_API_KEY, model=settings.LLM_MODEL
        )
    return MockLLMProvider()
