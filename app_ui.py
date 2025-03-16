import streamlit as st
import requests
import os
import base64
import io

API_URL = "http://127.0.0.1:8000/query"

# Set Streamlit page config
st.set_page_config(page_title="IUIU-Support Assistant", page_icon="🤖")

# Load local background image
background_image_path = "./image/logo.png"  # Change this to your image path

# Function to convert image to base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convert image to base64
background_base64 = get_base64_of_image(background_image_path)

# Custom CSS for background image and button styling
st.markdown(
    f"""
    <style>
    /* Background Image */
    .stApp {{
        background: url("data:image/png;base64,{background_base64}") no-repeat;
        background-position: right bottom;  /* Position the image at the bottom-right */
        background-size: 10%;  /* Default size for smaller screens */
        margin-right: 20px;  /* Add space to the right of the image */
        margin-bottom: 20px;  /* Add space to the bottom of the image */
    }}

    /* Media Query for medium and larger screens */
    @media (min-width: 768px) {{
        .stApp {{
            background-size: 8%;  /* Smaller size for medium and larger screens */
        }}
    }}

    /* Media Query for large screens */
    @media (min-width: 1024px) {{
        .stApp {{
            background-size: 5%;  /* Even smaller for large screens */
        }}
    }}

    /* Custom Button */
    div.stButton > button {{
        color: #008000 !important;  /* Change this to your desired text color */
        border: 2px solid #008000 !important;  /* Change this to your desired border color */
        font-size: 18px !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
    }}

    /* Button Hover Effect */
    div.stButton > button:hover {{
        color: blue !important;
        border: 2px solid blue !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("📚 IUIU-Support Assistant")

# Chat-like UI
st.subheader("💬 Ask a Question")
user_query = st.text_input("Type your question:")

# Send query to API
if st.button("Ask IUIU"):
    if user_query:
        response = requests.post(API_URL, json={"query": user_query})
        response_data = response.json()

        query_text = response_data.get("query", "Unknown query")
        answer_text = response_data.get("response", "No response available.")

        # Format answer with proper newlines
        formatted_answer = answer_text.replace("\n", "\n\n")  # Adds spacing between paragraphs
        st.markdown(f"#### 📝 Answer:\n\n{formatted_answer}")

    else:
        st.warning("⚠️ Please enter a question.")
