import streamlit as st
import pandas as pd
import numpy as np

from predict import load_model, predict_record, MODEL_FEATURES

#
# Load Model (cached)

@st.cache_resource
def load():
    load_model()
    return True

load()

# Custom CSS Styling

st.markdown("""
    <style>

    /* Background image */
    .stApp {
        background-image: url("assets/bg.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }

    /* Hero Card */
    .hero {
        padding: 2rem;
        background: linear-gradient(135deg, #6C63FF 0%, #B49CFF 100%);
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }

    /* Content cards */
    .card {
        padding: 1.5rem;
        background: #ffffffcc;
        border-radius: 16px;
        margin-bottom: 1.6rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        backdrop-filter: blur(4px);
    }

    /* Section headers */
    h3 {
        color: #6C63FF;
    }

    </style>
""", unsafe_allow_html=True)



# Hero Header

st.markdown("""
<div class="hero">
    <h1>🏡 Immo Eliza House Price Predictor</h1>
    <p>Your sleek and elegant gateway to property insights.</p>
</div>
""", unsafe_allow_html=True)


# Field Categories

numeric_fields = [
    'build_year', 'facades', 'living_area', 'number_rooms',
    'postal_code', 'property_id'
]

bool_fields = ['garden', 'swimming_pool', 'terrace', 'has_garden']

text_fields = [
    'locality_name', 'property_type', 'property_type_name',
    'property_url', 'state', 'province', 'state_mapped', 'region'
]


# Form Card

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Enter Property Details")

with st.form("prediction_form"):

    # Numeric fields
    st.markdown("### 📏 Numeric Values")
    cols = st.columns(3)
    for i, field in enumerate(numeric_fields):
        with cols[i % 3]:
            user_input = st.number_input(
                field,
                value=None,
                placeholder="Leave empty if unknown"
            )
        st.session_state[field] = user_input

    # Boolean fields
    st.markdown("### 🔘 Boolean Fields")
    cols = st.columns(2)
    for i, field in enumerate(bool_fields):
        with cols[i % 2]:
            user_input = st.selectbox(
                field,
                [None, True, False],
                index=0
            )
        st.session_state[field] = user_input

    # Text fields
    st.markdown("### 📝 Text Fields")
    for field in text_fields:
        st.session_state[field] = st.text_input(field, value="")

    submitted = st.form_submit_button("✨ Predict Price")

st.markdown('</div>', unsafe_allow_html=True)



# Prediction Card

if submitted:
    cleaned = {
        k: (v if v != "" else None)
        for k, v in st.session_state.items()
        if k in numeric_fields + bool_fields + text_fields
    }

    try:
        result = predict_record(cleaned)
        prediction = result["predictions"][0]

        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <h2>💶 Estimated Price</h2>
            <p style="font-size: 2rem; font-weight: bold; color:#6C63FF;">
                € {prediction:,.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")



