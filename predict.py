import json

import httpx

API_URL = "http://127.0.0.1:8000/api/v1/predict"

sample_game = {
    "price": 24.99,
    "required_age": 0,
    "achievements": 49,
    "screenshots_count": 21,
    "release_month": 9,
    "windows": True,
    "mac": True,
    "linux": False,
    "has_header_image": True,
    "has_website": True,
    "has_support_url": True,
    "has_support_email": True,
    "developers": ["Supergiant Games"],
    "publishers": ["Supergiant Games"],
    "short_description": (
        "Defy the god of the dead as you hack and slash out of the Underworld "
        "in this rogue-like dungeon crawler from the creators of Bastion, Transistor, and Pyre."
    ),
    "genres": ["Action", "Indie", "RPG"],
    "categories": [
        "Single-player",
        "Steam Achievements",
        "Full controller support",
        "Steam Cloud",
        "Remote Play on TV",
    ],
    "supported_languages": [
        "English",
        "French",
        "Italian",
        "German",
        "Spanish - Spain",
        "Korean",
        "Polish",
        "Portuguese - Brazil",
        "Russian",
        "Simplified Chinese",
        "Japanese",
    ],
    "full_audio_languages": ["English"],
}


def main() -> None:
    response = httpx.post(API_URL, json=sample_game, timeout=10.0)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:", json.dumps(response.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
