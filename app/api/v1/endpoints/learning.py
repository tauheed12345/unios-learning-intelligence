from fastapi import APIRouter
from app.schemas import NormalizedLearningContext, GeneratedLesson
from app.services import (
    build_pedagogy_prompt,
    build_lesson_prompt,
    get_llm_provider,
)

router = APIRouter()


@router.post(
    "/plan-lesson",
    response_model=GeneratedLesson,
    summary="Plan and Generate Lesson (AI/ML-2)",
    description="Accepts NormalizedLearningContext, applies Pedagogy guidelines, and generates structured lesson blocks via LLM.",
)
@router.post(
    "/generate-lesson",
    response_model=GeneratedLesson,
    summary="Generate Structured Lesson (AI/ML-2 Alias)",
    description="Alias endpoint for generating a structured lesson from NormalizedLearningContext.",
)
def plan_and_generate_lesson(context: NormalizedLearningContext) -> GeneratedLesson:
    """AI/ML-2 capability interface: Takes normalized context and generates a structured lesson."""
    print(f"\n>>> [API /api/v1/learning] Received Lesson Request:")
    print(f"    - Context ID: {context.context_id}")
    print(f"    - Topic:      {context.topic}")
    print(f"    - Objective:  {context.objective}")
    print(f"    - Learner:    {context.learner_state.learner_id} (Stage: {context.learner_state.stage.value})")
    print(f"    - Confidence: {context.learner_state.confidence.value}")
    print(f"    - Weaknesses: {context.learner_state.weak_topics}")

    provider = get_llm_provider()

    # 1. Pedagogy decision
    pedagogy_prompt = build_pedagogy_prompt(context)
    pedagogy_decision = provider.generate_pedagogy_decision(pedagogy_prompt)

    # 2. Structured lesson generation
    lesson_prompt = build_lesson_prompt(context, pedagogy_decision)
    lesson = provider.generate_lesson(
        prompt=lesson_prompt,
        context_id=context.context_id,
        topic=context.topic,
        pedagogy_decision=pedagogy_decision,
    )

    print(f">>> [API /api/v1/learning] Returning {len(lesson.blocks)} blocks to caller.\n")
    return lesson
