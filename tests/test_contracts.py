import unittest
from pydantic import ValidationError
from app.schemas import (
    LearnerStage,
    ConfidenceLevel,
    LearnerState,
    NormalizedLearningContext,
    TeachingStrategy,
    LessonBlockType,
)
from app.services import (
    build_pedagogy_prompt,
    build_lesson_prompt,
    get_llm_provider,
)


class TestSprint1IntelligenceHarness(unittest.TestCase):
    """Evaluation and contract test harness for AI/ML-2 Sprint-1."""

    def setUp(self):
        self.provider = get_llm_provider()

    def test_bachelor_learner_contract(self):
        """Verify Bachelor stage normalized context and foundational prompt generation."""
        state = LearnerState(
            learner_id="bachelor_01",
            stage=LearnerStage.BACHELOR,
            confidence=ConfidenceLevel.LOW,
            weak_topics=["pointers"],
            concept_mastery={"syntax": 0.9, "pointers": 0.2},
        )
        ctx = NormalizedLearningContext(
            topic="Memory Pointers",
            objective="Understand memory addresses and dereferencing",
            learner_state=state,
        )

        prompt = build_pedagogy_prompt(ctx)
        self.assertIn("Bachelor's Degree", prompt)
        self.assertIn("Memory Pointers", prompt)

        decision = self.provider.generate_pedagogy_decision(prompt)
        self.assertIsInstance(decision.strategy, TeachingStrategy)

    def test_master_learner_contract(self):
        """Verify Master's stage normalized context and specialized prompt generation."""
        state = LearnerState(
            learner_id="master_01",
            stage=LearnerStage.MASTER,
            confidence=ConfidenceLevel.HIGH,
            concept_mastery={"distributed_systems": 0.8},
        )
        ctx = NormalizedLearningContext(
            topic="Raft Consensus",
            objective="Evaluate leader election and log replication",
            learner_state=state,
        )

        prompt = build_pedagogy_prompt(ctx)
        self.assertIn("Master's Degree", prompt)
        self.assertIn("Raft Consensus", prompt)

    def test_graduate_learner_contract(self):
        """Verify Graduate stage normalized context and career/interview prompt generation."""
        state = LearnerState(
            learner_id="grad_01",
            stage=LearnerStage.GRADUATE,
            career_goal="Backend Engineer",
            weak_topics=["system_design"],
        )
        ctx = NormalizedLearningContext(
            topic="System Design Mock",
            objective="Design rate limiter for high-traffic API",
            learner_state=state,
        )

        prompt = build_pedagogy_prompt(ctx)
        self.assertIn("Recent Graduate", prompt)
        self.assertIn("System Design Mock", prompt)

    def test_structured_lesson_generation_and_boundary_rules(self):
        """Verify that lesson blocks are structured and strictly contain NO raw UI/HTML/React code."""
        state = LearnerState(stage=LearnerStage.BACHELOR)
        ctx = NormalizedLearningContext(
            topic="Binary Search",
            objective="Master O(log n) search",
            learner_state=state,
        )

        decision = self.provider.generate_pedagogy_decision(build_pedagogy_prompt(ctx))
        lesson = self.provider.generate_lesson(
            prompt=build_lesson_prompt(ctx, decision),
            context_id=ctx.context_id,
            topic=ctx.topic,
        )

        # Check lesson structure
        self.assertGreaterEqual(len(lesson.blocks), 1)
        block_types = [b.type for b in lesson.blocks]
        self.assertIn(LessonBlockType.OBJECTIVE, block_types)
        self.assertIn(LessonBlockType.VISUAL_SPEC, block_types)

        # Non-negotiable boundary check: strictly NO raw UI code
        forbidden_tags = [
            "<div",
            "<html",
            "<script",
            "import React",
            "export default function",
        ]
        for block in lesson.blocks:
            for forbidden in forbidden_tags:
                self.assertNotIn(
                    forbidden,
                    block.content,
                    f"Boundary violation: Found raw UI code '{forbidden}' in block content!",
                )

    def test_invalid_context_rejection(self):
        """Verify that invalid payloads without required fields raise ValidationError."""
        with self.assertRaises(ValidationError):
            # Missing required fields 'topic', 'objective', 'learner_state'
            NormalizedLearningContext.model_validate({})


if __name__ == "__main__":
    unittest.main()
