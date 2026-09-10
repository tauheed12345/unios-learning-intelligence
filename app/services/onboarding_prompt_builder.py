from app.schemas.onboarding import OnboardingInputProfile
from app.services.prompt_builder import get_stage_guidelines


def build_onboarding_intelligence_prompt(profile: OnboardingInputProfile) -> str:
    """Builds a token-optimized, structured prompt instructing the LLM to generate

    multi-dimensional Learner Intelligence matching LearnerIntelligenceReport.
    Target output token budget: ~700-900 tokens.
    """
    stage_guide = get_stage_guidelines(profile.stage)

    skills_text = (
        "\n".join(
            [
                f"  - {s.skill_name} ({s.category.value}): Rating {s.self_rating}/5"
                for s in profile.declared_skills
            ]
        )
        if profile.declared_skills
        else "  - None declared"
    )

    interests_text = (
        ", ".join(profile.interests) if profile.interests else "General computer science"
    )

    return f"""You are the UniOS Learner Intelligence Engine (AI/ML-2).
Analyze the learner onboarding profile and output a comprehensive, structured Learner Intelligence Report.

LEARNER ONBOARDING PROFILE:
- Learner ID: {profile.learner_id}
- Academic Program: {profile.academic_program} (Semester {profile.current_semester or 'N/A'})
{stage_guide}
- Target Role: {profile.target_role}
- Declared Skills & Baseline Ratings:
{skills_text}
- Domain Interests: {interests_text}
- Learning Preferences:
  * Preferred Medium: {profile.preferences.preferred_medium}
  * Pace: {profile.preferences.pace}
  * Weekly Hours: {profile.preferences.weekly_hours}h/week
  * Practical vs Theory Ratio: {profile.preferences.practical_vs_theory_ratio:.2f} (1.0 = 100% practical)
- Prior Projects / Background: {profile.prior_projects_summary or 'None provided'}
- Motivation Statement: {profile.motivation_statement or 'None provided'}

EVALUATION OBJECTIVES (6 DIMENSIONS):
1. Skill Baseline & Gap Analysis:
   - Synthesize baseline proficiency (beginner, intermediate, advanced) in a concise narrative.
   - Extract core verified strengths and 2-4 critical skill gaps for '{profile.target_role}'.
2. Conceptual & Prerequisite Depth Analysis:
   - Evaluate conceptual depth and prerequisite health (solid, minor_gaps, needs_remediation).
   - Balance theoretical foundations vs applied coding.
   - Prescribe 2-4 prerequisite foundation topics.
3. Learning Style & Cognitive Modality:
   - Prescribe dominant & secondary cognitive modality, optimal pacing, feedback cadence, and content formats.
4. Career Role Alignment & Milestone Roadmap:
   - Calculate role alignment score (0.0 to 1.0) and estimate timeline in months.
   - Outline 3 sequenced roadmap milestones and high-priority competencies.
5. Motivation, Engagement & Frustration Triggers:
   - Identify primary driver, intrinsic/extrinsic orientation, engagement hooks, frustration triggers, and resilience advice.
6. Readiness Scoring & Entry-Level Determination:
   - Calculate overall readiness score (0.0 to 1.0) and readiness tier (high, moderate, needs_scaffolding).
   - Assign recommended entry level (foundational, intermediate, advanced) and 3-4 immediate onboarding recommendations.

OUTPUT JSON SCHEMA:
Output strictly valid JSON matching the following structure. Keep values concise (under 800 tokens total):
{{
  "learner_id": "{profile.learner_id}",
  "stage": "{profile.stage.value}",
  "target_role": "{profile.target_role}",
  "executive_summary": "<2-3 sentence synthesized learner overview>",
  "skill_analysis": {{
    "baseline_summary": "<concise baseline summary>",
    "proficiency_level": "beginner" | "intermediate" | "advanced",
    "core_strengths": ["<strength1>", "<strength2>"],
    "critical_skill_gaps": ["<gap1>", "<gap2>"]
  }},
  "knowledge_analysis": {{
    "conceptual_depth": "foundational" | "applied" | "theoretical",
    "prerequisite_health": "solid" | "minor_gaps" | "needs_remediation",
    "theoretical_vs_applied_balance": "<concise balance assessment>",
    "recommended_foundation_topics": ["<topic1>", "<topic2>"]
  }},
  "learning_style": {{
    "dominant_modality": "<modality>",
    "secondary_modality": "<modality>",
    "recommended_pacing": "accelerated" | "standard" | "scaffolded",
    "feedback_cadence": "<cadence description>",
    "content_format_priorities": ["<format1>", "<format2>"]
  }},
  "career_goals": {{
    "target_role": "{profile.target_role}",
    "role_alignment_score": <float 0.0-1.0>,
    "key_milestones": ["<milestone 1>", "<milestone 2>", "<milestone 3>"],
    "high_priority_competencies": ["<competency 1>", "<competency 2>"],
    "estimated_timeline_months": <integer>
  }},
  "motivation": {{
    "primary_driver": "<primary driver>",
    "intrinsic_vs_extrinsic": "mostly_intrinsic" | "balanced" | "mostly_extrinsic",
    "engagement_hooks": ["<hook 1>", "<hook 2>"],
    "potential_frustration_triggers": ["<trigger 1>", "<trigger 2>"],
    "resilience_advice": "<actionable advice>"
  }},
  "readiness": {{
    "overall_readiness_score": <float 0.0-1.0>,
    "readiness_tier": "high" | "moderate" | "needs_scaffolding",
    "recommended_entry_level": "foundational" | "intermediate" | "advanced",
    "onboarding_recommendations": ["<rec 1>", "<rec 2>", "<rec 3>"]
  }}
}}

CRITICAL BOUNDARY RULES:
- Output ONLY valid JSON.
- Under NO circumstances should you output raw HTML, JSX, or React UI code.
- Keep bullet points concise to maintain the safe token budget (700-900 tokens).
"""
