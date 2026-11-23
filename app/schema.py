from pydantic import BaseModel, ConfigDict, Field


class GamePredictionRequest(BaseModel):
    price: float = Field(
        ..., ge=0.0, description="Game price in USD (0.0 for Free-to-Play)"
    )
    required_age: int = Field(
        default=0, ge=0, le=21, description="Minimum required age"
    )
    achievements: int = Field(
        default=0, ge=0, description="Total number of achievements"
    )
    screenshots_count: int = Field(default=0, ge=0, description="Number of screenshots")
    release_month: int = Field(..., ge=1, le=12, description="Release month (1-12)")

    windows: bool = Field(default=True, description="Windows support")
    mac: bool = Field(default=False, description="macOS support")
    linux: bool = Field(default=False, description="Linux support")
    has_header_image: bool = Field(
        default=True, description="Presence of a header image"
    )
    has_website: bool = Field(
        default=False, description="Presence of an official website"
    )
    has_support_url: bool = Field(
        default=False, description="Presence of a support URL"
    )
    has_support_email: bool = Field(
        default=False, description="Presence of a support email"
    )

    developers: list[str] = Field(
        default_factory=list, description="List of developer names"
    )
    publishers: list[str] = Field(
        default_factory=list, description="List of publisher names"
    )

    short_description: str = Field(
        default="", description="Short game description on Steam"
    )

    genres: list[str] = Field(default_factory=list, description="List of game genres")
    categories: list[str] = Field(
        default_factory=list, description="List of Steam categories"
    )
    supported_languages: list[str] = Field(
        default_factory=list, description="Supported interface/subtitle languages"
    )
    full_audio_languages: list[str] = Field(
        default_factory=list, description="Languages with full audio support"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "price": 19.99,
                "required_age": 16,
                "achievements": 35,
                "screenshots_count": 8,
                "release_month": 10,
                "windows": True,
                "mac": False,
                "linux": False,
                "has_header_image": True,
                "has_website": True,
                "has_support_url": True,
                "has_support_email": True,
                "developers": ["Supergiant Games"],
                "publishers": ["Supergiant Games"],
                "short_description": "Defy the god of the dead as you hack and slash out of the Underworld in this rogue-like dungeon crawler.",
                "genres": ["Action", "Indie", "RPG"],
                "categories": [
                    "Single-player",
                    "Steam Achievements",
                    "Full controller support",
                    "Steam Cloud",
                ],
                "supported_languages": [
                    "English",
                    "Russian",
                    "French",
                    "German",
                    "Spanish - Spain",
                    "Simplified Chinese",
                ],
                "full_audio_languages": ["English"],
            }
        }
    )


class GamePredictionResponse(BaseModel):
    success_tier: str = Field(
        ..., description="Most likely success tier (Low, Medium, High)"
    )
    prob_20k_plus: float = Field(
        ..., description="Combined probability of reaching 20k+ owners (Medium + High)"
    )
    probabilities: dict[str, float] = Field(
        ..., description="Calibrated probabilities for each success tier"
    )
    summary: str = Field(description="Human-readable summary of the prediction")