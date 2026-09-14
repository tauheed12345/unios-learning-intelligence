from fastapi import APIRouter, HTTPException, status
from app.schemas import NormalizedLearningContext, GeneratedLesson
from app.services import (
    build_pedagogy_prompt,
    build_lesson_prompt,
    get_llm_provider,
    LLMProviderError,
    LLMParseError,
)

router = APIRouter()


@router.post(
    "/plan-lesson",
    response_model=GeneratedLesson,
    status_code=status.HTTP_200_OK,
    summary="Plan and Generate Lesson (AI/ML-2)",
    description="Accepts NormalizedLearningContext, applies Pedagogy guidelines, and generates structured lesson blocks via LLM.",
    responses={
        200: {"description": "Structured lesson successfully planned and generated."},
        422: {"description": "Validation error: Malformed learning context."},
        429: {"description": "Too Many Requests: LLM provider rate limit exceeded."},
        502: {"description": "Bad Gateway: LLM provider unavailable or unparseable output."},
    },
)
@router.post(
    "/generate-lesson",
    response_model=GeneratedLesson,
    status_code=status.HTTP_200_OK,
    summary="Generate Structured Lesson (AI/ML-2 Alias)",
    description="Alias endpoint for generating a structured lesson from NormalizedLearningContext.",
    responses={
        200: {"description": "Structured lesson successfully generated."},
        422: {"description": "Validation error: Malformed learning context."},
        429: {"description": "Too Many Requests: LLM provider rate limit exceeded."},
        502: {"description": "Bad Gateway: LLM provider unavailable or unparseable output."},
    },
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

    try:
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
    except LLMProviderError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected internal error during lesson generation: {str(err)}",
        )

    print(f">>> [API /api/v1/learning] Returning {len(lesson.blocks)} blocks to caller.\n")
    return lesson
