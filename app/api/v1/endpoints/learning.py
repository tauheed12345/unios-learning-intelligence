from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.schemas import (
    NormalizedLearningContext,
    GeneratedLesson,
    PedagogyDecision,
    AssessmentEvidence,
    MasteryUpdateResult,
    ClosedLoopResult,
    RelevantMemoryQuery,
    LearnerStage,
)
from app.services import (
    build_pedagogy_prompt,
    build_lesson_prompt,
    get_llm_provider,
    get_memory_engine,
    get_pedagogy_engine,
    get_mastery_engine,
    get_remediation_service,
    LLMProviderError,
    LLMParseError,
)

router = APIRouter()


class RemediationCycleRequest(BaseModel):
    """Request payload for executing a closed-loop remediation cycle (A2-S4-04)."""
    learner_id: str = Field(..., min_length=1, description="Target learner identifier")
    topic: str = Field(..., min_length=1, description="Topic undergoing remediation evaluation")
    initial_score: float = Field(..., ge=0.0, le=1.0, description="Initial assessment or practice score")
    reassessment_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Optional reassessment score after remedial lesson"
    )
    learner_stage: LearnerStage = Field(
        default=LearnerStage.BACHELOR, description="Academic career stage"
    )
    academic_domain: Optional[str] = Field(
        default=None, description="Disciplinary domain (engineering, commerce, humanities, science, etc.)"
    )


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
@router.post(
    "/generate-learning-intelligence",
    response_model=GeneratedLesson,
    status_code=status.HTTP_200_OK,
    summary="Generate Learning Intelligence (Canonical KIE Contract)",
    description="Universal KIE Section 4.2 / 4.5 endpoint returning structured learning decisions and blocks.",
    responses={
        200: {"description": "Structured learning intelligence returned."},
        422: {"description": "Validation error: Malformed learning context."},
    },
)
def plan_and_generate_lesson(context: NormalizedLearningContext) -> GeneratedLesson:
    """AI/ML-2 capability interface: Takes normalized context and generates a structured lesson (A2-S4-02)."""
    req_id = context.request_id or context.context_id
    print(f"\n>>> [API /api/v1/learning] Received Lesson Request:")
    print(f"    - Request ID: {req_id} (Context ID: {context.context_id})")
    print(f"    - Topic:      {context.topic}")
    print(f"    - Objective:  {context.objective}")
    print(f"    - Learner:    {context.learner_state.learner_id} (Stage: {context.learner_state.stage.value})")
    if context.academic_domain:
        print(f"    - Domain:     {context.academic_domain}")
    print(f"    - Confidence: {context.learner_state.confidence.value}")
    print(f"    - Weaknesses: {context.learner_state.weak_topics}")

    # 0. Enrich with memory context if not explicitly provided
    if context.memory_context is None and context.learner_state and context.learner_state.learner_id:
        clean_learner_id = context.learner_state.learner_id.strip()
        if clean_learner_id:
            try:
                engine = get_memory_engine()
                query = RelevantMemoryQuery(
                    learner_id=clean_learner_id,
                    topic=context.topic,
                    objective=context.objective,
                    target_role=context.learner_state.career_goal,
                    current_learner_state=context.learner_state,
                )
                context.memory_context = engine.retrieve_relevant_context(query)
                print(f"    - Memory:     Retrieved context for '{clean_learner_id}' (Friction count: {len(context.memory_context.relevant_friction)})")
            except Exception as mem_err:
                print(f"    - Memory:     Retrieval skipped gracefully: {mem_err}")
                context.memory_context = None
    elif context.memory_context is not None:
        print(f"    - Memory:     Explicit context provided (Friction count: {len(context.memory_context.relevant_friction)})")

    provider = get_llm_provider()
    pedagogy_engine = get_pedagogy_engine()

    try:
        # 1. Pedagogy decision (A2-S4-01)
        # Use PedagogyEngine for comprehensive domain & evidence calibration
        pedagogy_decision = pedagogy_engine.determine_pedagogy(context)

        # If live LLM provider is active and not mock, blend with live provider reasoning
        from app.core.config import settings
        if settings.LLM_PROVIDER.lower() != "mock":
            try:
                pedagogy_prompt = build_pedagogy_prompt(context)
                llm_decision = provider.generate_pedagogy_decision(pedagogy_prompt)
                # Retain enriched sequence and practice level from engine
                llm_decision.sequence = pedagogy_decision.sequence
                llm_decision.practice_level = pedagogy_decision.practice_level
                llm_decision.remediation = pedagogy_decision.remediation
                pedagogy_decision = llm_decision
            except Exception as e:
                print(f"    - Notice: LLM pedagogy failed ({e}), falling back to deterministic PedagogyEngine decision.")

        # 2. Structured lesson generation (A2-S4-02)
        lesson_prompt = build_lesson_prompt(context, pedagogy_decision)
        lesson = provider.generate_lesson(
            prompt=lesson_prompt,
            context_id=context.context_id,
            topic=context.topic,
            pedagogy_decision=pedagogy_decision,
        )
        # Ensure request_id is propagated
        lesson.request_id = req_id

    except LLMProviderError as err:
        raise HTTPException(status_code=err.status_code, detail=err.message)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected internal error during lesson generation: {str(err)}",
        )

    print(f">>> [API /api/v1/learning] Returning {len(lesson.blocks)} blocks to caller.\n")
    return lesson


@router.post(
    "/pedagogy-decision",
    response_model=PedagogyDecision,
    status_code=status.HTTP_200_OK,
    summary="Determine Pedagogy Strategy (A2-S4-01)",
    description="Evaluates learner evidence, academic domain, stage, and memory to produce structured pedagogy.",
    responses={
        200: {"description": "Structured pedagogy decision returned."},
        422: {"description": "Validation error: Malformed learning context."},
    },
)
def determine_pedagogy_decision(context: NormalizedLearningContext) -> PedagogyDecision:
    """AI/ML-2 interface: Answers how a topic should be taught given learner evidence (A2-S4-01)."""
    engine = get_pedagogy_engine()
    return engine.determine_pedagogy(context)


@router.post(
    "/evaluate-assessment",
    response_model=MasteryUpdateResult,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Assessment Evidence & Update Mastery (A2-S4-03)",
    description=(
        "Consumes assessment evidence from Backend, updates concept mastery, classifies into canonical BRD tiers "
        "(Weak < 50%, Developing 50-79%, Proficient 80-100%), and emits measurable remediation signals."
    ),
    responses={
        200: {"description": "Mastery successfully evaluated with remediation signal."},
        422: {"description": "Validation error: Malformed assessment evidence."},
    },
)
def evaluate_assessment_evidence(evidence: AssessmentEvidence) -> MasteryUpdateResult:
    """AI/ML-2 interface: Computes mastery and generates remediation signals from assessment evidence (A2-S4-03)."""
    engine = get_mastery_engine()
    return engine.evaluate_assessment(evidence)


@router.post(
    "/remediation-loop",
    response_model=ClosedLoopResult,
    status_code=status.HTTP_200_OK,
    summary="Execute Remediation Closed-Loop (A2-S4-04)",
    description=(
        "Executes complete closed loop: Initial Assessment -> Mastery Update -> Remediation Pedagogy -> "
        "Reassessment -> Updated State & Next Action."
    ),
    responses={
        200: {"description": "Closed-loop remediation cycle trace returned."},
        422: {"description": "Validation error: Malformed request payload."},
    },
)
def execute_remediation_loop(request: RemediationCycleRequest) -> ClosedLoopResult:
    """AI/ML-2 interface: Orchestrates the adaptive remediation loop (A2-S4-04)."""
    service = get_remediation_service()
    return service.run_remediation_cycle(
        learner_id=request.learner_id,
        topic=request.topic,
        initial_score=request.initial_score,
        reassessment_score=request.reassessment_score,
        learner_stage=request.learner_stage,
        academic_domain=request.academic_domain,
    )
