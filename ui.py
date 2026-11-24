import httpx
import streamlit as st

st.title("Steam Game Success Predictor")

price = st.number_input("Price ($)", min_value=0.0, value=19.9, step=1.0)
required_age = st.slider("Required age", 0, 21, 0)
achievements = st.number_input("Number of achievements", min_value=0, value=0, step=1)
screenshots = st.number_input("Number of screenshots", min_value=0, value=0, step=1)
release_month = st.slider("Release month", 1, 12, 10)

windows = st.toggle("Windows support", True)
mac = st.toggle("Mac support", False)
linux = st.toggle("Linux support", False)
has_header_image = st.toggle("Presence of a header image", True)
has_website = st.toggle("Presence of an official website", True)
has_support_url = st.toggle("Presence of a support URL", True)
has_support_email = st.toggle("Presence of a support email", True)

developers = st.multiselect("List of developers", options=[], accept_new_options=True, max_selections=5)
publishers = st.multiselect("List of publishers", options=[], accept_new_options=True, max_selections=5)

short_description = st.text_area("Short description")

genres = st.multiselect(
    "List of game genres", 
    options=[], 
    accept_new_options=True, 
    max_selections=5)
categories = st.multiselect(
    "List of Steam categories", 
    options=[], 
    accept_new_options=True, 
    max_selections=5)
supported_languages = st.multiselect(
    "Supported interface/subtitle languages", 
    options=[], 
    accept_new_options=True, 
    max_selections=5)
full_audio_languages = st.multiselect(
    "Languages with full audio support", 
    options=[], 
    accept_new_options=True, 
    max_selections=5)

if st.button("Predict"):
    payload = [price, release_month, short_description]
    response = httpx.post("http://127.0.0.1:8000/steam-predictor/predict", json=payload)
    st.json(response.json())