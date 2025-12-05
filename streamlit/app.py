import streamlit as st
import pandas as pd
import numpy as np

from predict import load_model, predict_record, MODEL_FEATURES

# Load model

@st.cache_resource
def load():
    load_model()
    return True

load()

# Custom CSS

st.markdown("""
    <style>
    .stApp {
        background-image: url("streamlit/assets/bg.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    .hero {
        padding: 2rem;
        background: linear-gradient(135deg, #6C63FF 0%, #B49CFF 100%);
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    .card {
        padding: 1.5rem;
        background: #ffffffcc;
        border-radius: 16px;
        margin-bottom: 1.6rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        backdrop-filter: blur(4px);
    }
    h3 { color: #6C63FF; }
    </style>
""", unsafe_allow_html=True)


# Header

st.markdown("""
<div class="hero">
    <h1> Immo Eliza House Price Predictor</h1>
    <p>Easily estimate a property's selling price.</p>
</div>
""", unsafe_allow_html=True)


# Field definitions

numeric_fields = {
    "build_year": "Year the property was built",
    "facades": "Number of façades",
    "living_area": "Living area (m²)",
    "number_rooms": "Number of rooms",
    "postal_code": "Postal code"
}

bool_fields = {
    "garden": "Has a garden?",
    "swimming_pool": "Has a swimming pool?",
    "terrace": "Has a terrace?",
    "has_garden": "Official garden indicator"
}

text_fields = {
    "locality_name": "Locality",
    "property_type": "Property type",
    "property_type_name": "Detailed property type",
    "state": "State of the property",
    "province": "Province",
    "state_mapped": "Mapped property condition",
    "region": "Region"
}

# Input Form

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Enter Property Details")

with st.form("prediction_form"):

    # Numeric Fields
    st.markdown("### Property Measurements")
    cols = st.columns(3)

    for i, (field, label) in enumerate(numeric_fields.items()):
        with cols[i % 3]:
            value = st.number_input(
                label,
                value=None,
                step=1,          #  force integer stepping
                format="%d",     #  force integer formatting
                placeholder="Optional"
            )
        st.session_state[field] = int(value) if value not in (None, "") else None

    # Boolean Fields
    st.markdown("### 🔘 Features")
    cols = st.columns(2)

    for i, (field, label) in enumerate(bool_fields.items()):
        with cols[i % 2]:
            choice = st.selectbox(
                label,
                ["Choose an option", "Yes", "No"],
                index=0
            )
        st.session_state[field] = (
            True if choice == "Yes"
            else False if choice == "No"
            else None
        )

    # Text fields
    st.markdown("### Location & Description")
    for field, label in text_fields.items():
        txt = st.text_input(label, value="", placeholder="Optional")
        st.session_state[field] = txt if txt else None

    submitted = st.form_submit_button("Predict Price")

st.markdown('</div>', unsafe_allow_html=True)

# Prediction

if submitted:

    cleaned = {
        k: v
        for k, v in st.session_state.items()
        if k in numeric_fields or k in bool_fields or k in text_fields
    }

    try:
        # Model returns a float → convert to int so final output has no float
        result = predict_record(cleaned)
        prediction = int(result["predictions"][0])

        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <h2>Estimated Price</h2>
            <p style="font-size: 2rem; font-weight: bold; color:#6C63FF;">
                € {prediction:,}
            </p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
