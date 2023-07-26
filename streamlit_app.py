import base64, io
import json
import requests

import streamlit as st
from PIL import Image


from cp_app_st import pain_emotion_from_image

# URL = "http://127.0.0.1:5000/detect_skin_defect"

st.set_page_config(layout="wide", page_title="Cerabral Palsey Classifier")

st.write("## Classify emotion of cerabal palsey patient using pain scale.")
st.write(
            """
            Upload or take picture of a cerabral palsey patient to understand their emotion.
            """
        )

st.sidebar.write("## Upload or take picture from your camera :gear:")

if "uploaded_img_changed" not in st.session_state:
    st.session_state.uploaded_img_changed = 0
    # prev_state = 1


def uploaded_on_change_callback():
    # prev_state = st.session_state_uploaded_img_changed
    st.session_state.uploaded_img_changed ^= 1

def convert_image(img):
    buffer = io.BytesIO()
    img.save(buffer, format="png")
    byte_img = base64.b64encode(buffer.getvalue()).decode()


    return byte_img

def file_upload_callback(img_input): 
    # img_input = st.session_state.uploaded_img
    img = Image.open(img_input) 
    img_col.write("Uploaded Input Image :camera:")
    img_col.image(img)

    encoded_img = convert_image(img)
    classify_image(encoded_img)

def classify_image(encoded_img_str):
    body = {"image":encoded_img_str}

    result = pain_emotion_from_image(body)
    # response = requests.post(url=URL, json=body)
    # if response.status_code == 200:
    #     result = response.json()

    if result:
        st.info(
                    f"""
                        Top 5 predictions:
                    """
                )
        st.info(
                
                    "\n\n".join(f"{result['predicted_classes'][i]}" for i in range(5))
                )
    else:
        st.error("Could not perform classification.")

img_col = st.columns(1)[0]
uploaded_img = st.sidebar.file_uploader(
                label="Upload and image", type=["png", "jpg", "jpeg"],
                help="Ensure that the uploaded image is focused on the area of the affected skin.",
                on_change=uploaded_on_change_callback
            )
if uploaded_img:
    file_upload_callback(uploaded_img)
# if (prev_state != st.session_state.uploaded_img_changed) and uploaded_img:
# if (st.session_state.uploaded_img_changed) and uploaded_img:
#     prev_state = st.session_state.uploaded_img_changed
#     file_upload_callback(uploaded_img)
