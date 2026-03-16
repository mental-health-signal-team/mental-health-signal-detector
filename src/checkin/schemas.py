from pydantic import BaseModel, Field


class CheckInRequest(BaseModel):
    emoji: str | None = Field(default=None, description="Emoji choisi parmi 😄🙂😐😔😢")
    text: str | None = Field(default=None, max_length=1000, description="Texte libre optionnel")
    step: int = Field(default=1, ge=1, le=2, description="1=check-in initial, 2=réponse à la question de suivi")


class ResourceItem(BaseModel):
    title: str
    description: str
    action: str
    url: str


class CheckInResponse(BaseModel):
    level: str = Field(..., description="green | yellow | red")
    score: float = Field(..., ge=0.0, le=1.0)
    greeting: str
    message: str
    tip: str | None = None
    follow_up: str | None = Field(default=None, description="Question de suivi si pertinent")
    resources: list[ResourceItem] = Field(default_factory=list)
    detected_lang: str = "fr"
