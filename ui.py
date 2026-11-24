import os

import httpx
import streamlit as st

BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000/steam-predictor/")

st.set_page_config(
    page_title="Steam Game Success Predictor",
    page_icon="🎮",
    layout="wide",
)

st.markdown(
    "<style>.block-container { max-width: 1000px; }</style>",
    unsafe_allow_html=True,
)
st.title("Steam Game Success Predictor")

@st.cache_data
def get_metadata() -> dict[str, list[str]]:
    response = httpx.get(f"{BASE_URL}metadata", timeout=5.0)
    response.raise_for_status()
    return response.json()

with st.form("game_form"):
    price = st.number_input("Price ($)", min_value=0.0, max_value=200.0, value=24.99, step=1.0)
    required_age = st.slider("Required age", 0, 21, 0)
    achievements = st.number_input("Number of achievements", min_value=0, value=49, step=1)
    screenshots_count = st.number_input("Number of screenshots", min_value=0, value=21, step=1)
    release_month = st.slider("Release month", 1, 12, 9)

    c1, c2, c3 = st.columns(3)
    with c1:
        windows = st.toggle("Windows", True)
    with c2:
        mac = st.toggle("macOS", False)
    with c3:
        linux = st.toggle("Linux", False)

    has_header_image = st.toggle("Presence of a header image", True)
    has_website = st.toggle("Presence of an official website", True)
    has_support_url = st.toggle("Presence of a support URL", True)
    has_support_email = st.toggle("Presence of a support email", True)

    developers = st.multiselect(
        "List of developers", 
        options=["Supergiant Games"], 
        default=["Supergiant Games"], 
        accept_new_options=True, 
        max_selections=5)
    publishers = st.multiselect(
        "List of publishers", 
        options=["Supergiant Games"], 
        default=["Supergiant Games"], 
        accept_new_options=True, 
        max_selections=5)

    short_description = st.text_area("Short description", value="Defy the god of the dead as you hack and slash out of the Underworld "
    "in this rogue-like dungeon crawler from the creators of Bastion, Transistor, and Pyre.")

    try:
        metadata = get_metadata()
        genres = st.multiselect(
            "List of game genres", 
            options=metadata["genres"],
            default=["Action", "Indie", "RPG"],)
        categories = st.multiselect(
            "List of Steam categories", 
            options=metadata["categories"],
            default=[
                "Single-player",
                "Steam Achievements",
                "Full controller support",
                "Steam Cloud",
                "Remote Play on TV",
            ])
        supported_languages = st.multiselect(
            "Supported interface/subtitle languages", 
            options=metadata["supported_languages"],
            default=[
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
            ],)
        full_audio_languages = st.multiselect(
            "Languages with full audio support", 
            options=metadata["full_audio_languages"],
            default=["English"])
    except httpx.HTTPError:
        st.error("Run FastApi first")
        st.stop()
    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    payload = {
        "price": price,
        "required_age": required_age,
        "achievements": achievements,
        "screenshots_count": screenshots_count,
        "release_month": release_month,
        "windows": windows,
        "mac": mac,
        "linux": linux,
        "has_header_image": has_header_image,
        "has_website": has_website,
        "has_support_url": has_support_url,
        "has_support_email": has_support_email,
        "developers": developers,
        "publishers": publishers,
        "short_description": short_description,
        "genres": genres,
        "categories": categories,
        "supported_languages": supported_languages,
        "full_audio_languages": full_audio_languages,
    }

    with st.spinner("Predicting..."):
        try: 
            response = httpx.post(BASE_URL+"predict", json=payload, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            m1, m2 = st.columns(2)
            m1.metric("Predicted Tier", data["success_tier"])
            m2.metric("Chance 20k+ owners", f"{data["prob_20k_plus"]:.1%}")

            tier = data["success_tier"]
            if tier == "High":
                st.success(data["summary"])
            elif tier == "Medium":
                st.info(data["summary"])
            else:
                st.warning(data["summary"])

            for tier in ("High", "Medium", "Low"):
                prob = data["probabilities"][tier]
                st.progress(prob, text=f"{tier}: {prob:.1%}")

            with st.expander("Raw JSON"):
                st.json(data)
        except httpx.HTTPError:
            st.error("Backend error")