"""
Schémas Pydantic — endpoint POST /solutions

Miroir Python des types TypeScript :
  types/diagnostic.ts  → DiagnosticProfileRequest
  types/solutions.ts   → SolutionResponse
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ─── Types de base ────────────────────────────────────────────────────────────

DistressLevel = Literal["light", "elevated", "critical"]
ClinicalDimension = Literal["burnout", "anxiety", "depression_masked", "dysregulation"]
ClinicalProfile = Literal["wellbeing", "adjustment", "burnout", "anxiety", "depression", "crisis"]
TriageLevel = Literal[0, 1, 2, 3, 4]
TherapeuticBrick = Literal[
    "crisis", "professional", "mindfulness",
    "psychoeducation", "cbt_restructuring", "cbt_activation",
]
ResourceType = Literal["phone", "website", "app", "person"]
Mode = Literal["kids", "adult"]

# ─── Entrée ───────────────────────────────────────────────────────────────────

VALID_EMOTION_IDS = frozenset(["joy", "sadness", "anger", "fear", "stress", "calm", "tiredness", "pride"])


class DiagnosticProfileRequest(BaseModel):
    """DiagnosticProfile calculé par le frontend, soumis pour obtenir le contenu thérapeutique."""

    emotionId: str = Field(..., description="joy | sadness | anger | fear | stress | calm | tiredness | pride")
    mode: Mode
    userText: str = Field(..., min_length=1, max_length=5000)

    # Self-report (QuickCheck — null si émotion positive ou étape passée)
    selfScore: float | None = Field(None, ge=0.0, le=1.0)
    selfReportAnswers: list[int] | None = None

    # ML
    mlScore: float | None = Field(None, ge=0.0, le=1.0)
    finalScore: float | None = Field(None, ge=0.0, le=1.0)

    # Analyse clinique calculée côté frontend
    distressLevel: DistressLevel
    clinicalDimensions: list[ClinicalDimension] = Field(default_factory=list)
    clinicalProfile: ClinicalProfile

# ─── Sortie ───────────────────────────────────────────────────────────────────

class Resource(BaseModel):
    id: str
    label: str
    detail: str | None = None
    type: ResourceType
    href: str | None = None
    website: str | None = None
    urgent: bool = False


class MicroAction(BaseModel):
    id: str
    title: str
    description: str
    duration: str | None = None


class SolutionResponse(BaseModel):
    level: TriageLevel
    clinicalProfile: ClinicalProfile
    message: str
    closing: str
    microActions: list[MicroAction]
    therapeuticBrick: TherapeuticBrick
    resources: list[Resource]
    escalationRequired: bool
