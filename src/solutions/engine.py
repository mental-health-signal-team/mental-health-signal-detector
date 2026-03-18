"""
Moteur de recommandation clinique — Python port de lib/solutionEngine.ts

Entrée  : DiagnosticProfileRequest
Sortie  : SolutionResponse

Logique : triage en 5 niveaux (0–4) → protocole adapté
"""

from src.solutions.schemas import (
    ClinicalProfile,
    DiagnosticProfileRequest,
    SolutionResponse,
    TherapeuticBrick,
)
from src.solutions.data import (
    RESOURCES,
    RESOURCES_BY_LEVEL,
    get_actions,
    get_closing,
    get_message,
)


def map_to_triage_level(profile: DiagnosticProfileRequest) -> int:
    """Mappe le DiagnosticProfile vers un niveau de triage 0–4."""

    # Niveau 4 — urgence critique
    if profile.clinicalProfile == "crisis":
        return 4

    # Niveau 3 — alerte clinique
    if profile.distressLevel == "critical":
        return 3
    if (
        profile.clinicalProfile == "burnout"
        and "burnout" in profile.clinicalDimensions
        and profile.emotionId in ("tiredness", "sadness")
    ):
        return 3

    # Niveau 2 — détresse modérée
    if profile.distressLevel == "elevated":
        return 2
    if profile.clinicalProfile in ("anxiety", "depression", "burnout"):
        return 2

    # Niveau 1 — inconfort léger
    if profile.clinicalProfile == "adjustment":
        return 1

    # Niveau 0 — bien-être
    return 0


def select_brick(clinical_profile: ClinicalProfile, level: int) -> TherapeuticBrick:
    if level >= 4:
        return "crisis"
    if level >= 3:
        return "professional"
    bricks: dict[str, TherapeuticBrick] = {
        "anxiety": "mindfulness",
        "burnout": "psychoeducation",
        "depression": "cbt_restructuring" if level >= 2 else "cbt_activation",
        "adjustment": "cbt_activation",
        "wellbeing": "cbt_activation",
    }
    return bricks.get(clinical_profile, "mindfulness")


def select_resources(profile: DiagnosticProfileRequest, level: int) -> list:
    """Enrichit la liste de base avec des ressources contextuelles (émotion + dimensions)."""
    base = list(RESOURCES_BY_LEVEL[level][profile.mode])
    seen = {r.id for r in base}

    def add(key: str) -> None:
        r = RESOURCES[key]
        if r.id not in seen:
            base.append(r)
            seen.add(r.id)

    if profile.mode == "adult" and level >= 2:
        if (
            profile.emotionId in ("anger", "fear")
            or "dysregulation" in profile.clinicalDimensions
        ):
            add("violencesFemmes")

    if profile.mode == "kids" and level >= 2:
        if profile.emotionId in ("anger", "fear"):
            add("enfanceEnDanger")
        if profile.emotionId in ("stress", "anger"):
            add("antiHarcelement")
            add("harcelementScolaire")

    return base


def compute_solution(profile: DiagnosticProfileRequest) -> SolutionResponse:
    level = map_to_triage_level(profile)
    brick = select_brick(profile.clinicalProfile, level)
    message = get_message(profile.emotionId, level, profile.mode)
    closing = get_closing(profile.emotionId, level, profile.mode)
    micro_actions = get_actions(profile.clinicalProfile, level, profile.mode)
    resources = select_resources(profile, level)

    return SolutionResponse(
        level=level,  # type: ignore[arg-type]
        clinicalProfile=profile.clinicalProfile,
        message=message,
        closing=closing,
        microActions=micro_actions,
        therapeuticBrick=brick,
        resources=resources,
        escalationRequired=level >= 4,
    )
