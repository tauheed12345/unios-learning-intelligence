from app.schemas import NormalizedLearningContext, PedagogyDecision, LearnerStage


def get_stage_guidelines(stage: LearnerStage) -> str:
    """Returns pedagogical guidelines based on the learner's academic stage (Section 17)."""
    if stage == LearnerStage.BACHELOR:
        return (
            "- Learner Stage: Bachelor's Degree\n"
            "- Guidelines: Provide foundational sequencing, step-by-step prerequisite reinforcement, "
            "and clear concept progression."
        )
    elif stage == LearnerStage.MASTER:
        return (
            "- Learner Stage: Master's Degree\n"
            "- Guidelines: Provide specialization depth, architectural trade-offs, and advanced theoretical insight."
        )
    else:  # GRADUATE
        return (
            "- Learner Stage: Recent Graduate\n"
            "- Guidelines: Focus on job/interview readiness, industry application, and practical execution."
        )


def build_pedagogy_prompt(context: NormalizedLearningContext) -> str:
    """Constructs the prompt for selecting the teaching strategy and presentation mode."""
    learner = context.learner_state
    stage_guide = get_stage_guidelines(learner.stage)

    return f"""You are the UniOS Pedagogy Engine. Determine how to teach the following topic to the student.

LEARNER CONTEXT:
{stage_guide}
- Current Confidence: {learner.confidence.value}
- Weak Topics: {', '.join(learner.weak_topics) if learner.weak_topics else 'None reported'}
- Known Concept Mastery: {learner.concept_mastery}
- Career Goal: {learner.career_goal or 'Not specified'}

TARGET TOPIC & OBJECTIVE:
- Topic: {context.topic}
- Objective: {context.objective}

INSTRUCTIONS:
1. Select the appropriate TeachingStrategy (foundational, reinforcement, advancement, remediation).
2. Select DifficultyLevel (beginner, intermediate, advanced) based on mastery.
3. Select PresentationMode (traditional, visual, story, simulation, animation, interactive).
4. Provide a clear pedagogical rationale for your choices.
5. Return strictly structured output adhering to the PedagogyDecision schema.
"""


def build_lesson_prompt(
    context: NormalizedLearningContext, pedagogy: PedagogyDecision
) -> str:
    """Constructs the prompt for generating structured lesson blocks."""
    learner = context.learner_state
    stage_guide = get_stage_guidelines(learner.stage)

    return f"""You are the UniOS Lesson Generation Engine. Generate structured lesson blocks for the student.

LEARNER CONTEXT:
{stage_guide}
- Preferred Mode: {learner.learning_preference}

SELECTED PEDAGOGICAL STRATEGY:
- Strategy: {pedagogy.strategy.value}
- Difficulty: {pedagogy.difficulty.value}
- Presentation Mode: {pedagogy.presentation_mode.value}
- Explanation Depth: {pedagogy.explanation_depth}

TOPIC & OBJECTIVE:
- Topic: {context.topic}
- Objective: {context.objective}
- Grounded References: {context.curriculum_references if context.curriculum_references else 'Standard curriculum'}

STRICT GENERATION RULES:
1. Generate an ordered sequence of structured LessonBlocks (objective, explanation, worked_example, visual_spec, practice_task).
2. Under no circumstances should you generate raw HTML, React, JSX, or frontend UI code.
3. For visual_spec blocks, provide structured layout/renderer metadata (e.g. renderer type, step instructions).
4. Ensure the content strictly matches the {pedagogy.difficulty.value} level.
"""
