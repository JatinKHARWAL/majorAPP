import streamlit as st
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
from PIL import Image

model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Emotion Based Music Recommender")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner('Processing...'):
        #####################################
        # Perform Emotion Analysis on the uploaded image
        # Replace this part with your emotion analysis code
        # For example, you can use OpenCV to detect faces and perform emotion analysis
        # Here's a simplified example:

        frm = np.array(image)
        frm = np.flip(frm, axis=-1)  # Convert RGB to BGR

        res = holis.process(frm)

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]

            print(pred)
            st.success(f"Predicted Emotion: {pred}")

        try:
            if res.face_landmarks:  # Check if face landmarks are available before drawing
                drawing.draw_landmarks(
                    frm, res.face_landmarks, holistic.FACEMESH_TESSELATION
                )
            if res.left_hand_landmarks:  # Check if left hand landmarks are available before drawing
                drawing.draw_landmarks(
                    frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS
                )

            if res.right_hand_landmarks:  # Check if right hand landmarks are available before drawing
                drawing.draw_landmarks(
                    frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS
                )
        except Exception as e:
            print("Error occurred while drawing landmarks:", e)

        st.image(frm, caption='Processed Image', use_column_width=True)
        #####################################

    btn = st.button("Recommend me songs")

    if btn:
        # Perform action based on emotion prediction
        # For example, open a web page with recommended songs
        if not pred:
            st.warning("Please let me capture your emotion first")
        else:
            # Perform action based on predicted emotion
            # For demonstration, open a YouTube search page with the predicted emotion
            webbrowser.open(f"https://www.youtube.com/results?search_query={pred}+songs")


