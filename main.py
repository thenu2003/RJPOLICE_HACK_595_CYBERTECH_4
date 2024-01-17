# main.py
import streamlit as st
from rajastan.components.sidebar import sidebar
from rajastan.ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    is_open_ai_key_valid,
    display_file_read_error,
)
from rajastan.core.caching import bootstrap_caching
from rajastan.core.parsing import read_file
from rajastan.core.chunking import chunk_file
from rajastan.core.embedding import embed_files
from rajastan.core.qa import query_folder
from rajastan.core.utils import get_llm
from typing_extensions import TypeAliasType
from IPC_ID_NLP import train_ipc_model, predict_ipc_section
from CRPC_ID_NLP import train_crpc_model, predict_crpc_section

from ocr import model_1
from ocr import model_1_history
from tensorflow.keras.models import load_model  # Add this line

from ocr import model_1
from ocr import model_1_history
import numpy as np
model_1_history = load_model("best_model.h5")

import os 
import streamlit as st
import cv2

model_1_history = load_model("best_model.h5")

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer

# IPC Model Training
ipc_data = pd.read_json('data\ipc.json').fillna('UNKNOWN')
label_encoder_ipc = LabelEncoder()
ipc_data['label'] = label_encoder_ipc.fit_transform(ipc_data['section_desc'])
train_data_ipc, _ = train_test_split(ipc_data, test_size=0.2, random_state=42)
vectorizer_ipc = TfidfVectorizer(max_features=1000)
X_train_ipc = vectorizer_ipc.fit_transform(train_data_ipc['section_desc']).toarray()
y_train_ipc = train_data_ipc['label'].values
model_ipc = Sequential([
    Dense(32, activation='relu', input_dim=1000),
    Dense(len(label_encoder_ipc.classes_), activation='softmax')
])
model_ipc.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_ipc.fit(X_train_ipc, y_train_ipc, epochs=100, batch_size=32)
#model_ipc.save('model_ipc.h5')

# CRPC Model Training
crpc_data = pd.read_json('data\crpc.json').fillna('UNKNOWN')
label_encoder_crpc = LabelEncoder()
crpc_data['label'] = label_encoder_crpc.fit_transform(crpc_data['section_desc'])
train_data_crpc, _ = train_test_split(crpc_data, test_size=0.2, random_state=42)
vectorizer_crpc = TfidfVectorizer(max_features=1000)
X_train_crpc = vectorizer_crpc.fit_transform(train_data_crpc['section_desc']).toarray()
y_train_crpc = train_data_crpc['label'].values
model_crpc = Sequential([
    Dense(32, activation='relu', input_dim=1000),
    Dense(len(label_encoder_crpc.classes_), activation='softmax')
])
model_crpc.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_crpc.fit(X_train_crpc, y_train_crpc, epochs=100, batch_size=32)
#model_crpc.save('model_crpc.h5')

# main.py
EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL_LIST = ["gpt-3.5-turbo", "gpt-4"]

# Uncomment to enable debug mode
# MODEL_LIST.insert(0, "debug")

st.set_page_config(page_title="LegalAssist", page_icon="🔍", layout="wide")
st.header("🔍 LegalAssist - AI for Legal Section Suggestions")

# Enable caching for expensive functions
bootstrap_caching()

sidebar()
# 1) Prompt entering section
with st.form(key="prompt_form"):
    st.subheader("Prompt Section")
    gen_inp = st.text_input("Enter some text:", "Default Text")
    st.write("You entered:", gen_inp)
    prompt_submit = st.form_submit_button("Submit Prompt")

# 2) Answer generation section
if st.button("Generate Answer"):
    st.spinner("Generating answer...")

    new_description_ipc = gen_inp
    new_description_vectorized_ipc = vectorizer_ipc.transform([new_description_ipc]).toarray()
    predicted_ipc_probs = model_ipc.predict(new_description_vectorized_ipc)
    predicted_ipc_label = label_encoder_ipc.inverse_transform([predicted_ipc_probs.argmax()])[0]

    new_description_crpc = gen_inp
    new_description_vectorized_crpc = vectorizer_crpc.transform([new_description_crpc]).toarray()
    predicted_crpc_probs = model_crpc.predict(new_description_vectorized_crpc)
    predicted_crpc_label = label_encoder_crpc.inverse_transform([predicted_crpc_probs.argmax()])[0]

    matching_row_ipc = ipc_data[ipc_data['section_desc'] == predicted_ipc_label].iloc[0]
    matching_row_crpc = crpc_data[crpc_data['section_desc'] == predicted_crpc_label].iloc[0]

    # Display the results
    st.write("Predicted IPC Section Information:\n", matching_row_ipc)
    st.write("Predicted CRPC Section Information:\n", matching_row_crpc)

# 4) File upload section
with st.form(key="file_upload_form"):
    st.subheader("File Upload Section")
    uploaded_file = st.file_uploader(
        "Upload a pdf, docx, or txt file",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png", "gif"],
        help="Supported image formats: JPG, JPEG, PNG, GIF",
    )
    file_submit = st.form_submit_button("Submit File")
    # File upload
    st.title("Hindi Character Recognition with Streamlit")
    uploaded_file_image = st.file_uploader("Choose an image...", type="jpg")

    # Generate button inside the block
    if st.button("Generate") and uploaded_file_image:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file_image.read(), np.uint8), 1)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        hindi_character = 'क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व ख श ष स ह ॠ त्र ज्ञ ० १ २ ३ ४ ५ ६ ७ ८ ९'.split()

        # Resize the image to match the model input shape
        resized_image = cv2.resize(image, (32, 32))

        # Prepare the image for prediction
        test_input = resized_image.reshape((1, 32, 32, 3))

        # Make predictions using Model 1
        predicted_probability_1 = model_1.predict(test_input)
        predicted_class_1 = predicted_probability_1.argmax(axis=1)
        class_number_1 = predicted_class_1[0]

        # Make predictions using Model 2
        # Assuming Model 2 is designed for grayscale images
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        test_input_2 = grayscale_image.reshape((1, 32, 32, 1))
        predicted_probability_2 = model_1_history.predict(test_input_2)
        predicted_class_2 = predicted_probability_2.argmax(axis=1)
        class_number_2 = predicted_class_2[0]

        # Display the predicted classes for both models
        st.subheader("Predictions:")
        st.write("Model 1 Predicted Class:", hindi_character[class_number_1])
        st.write("Model 2 Predicted Class:", hindi_character[class_number_2])


# 3) Advanced sections
with st.expander("Advanced Options"):
    st.subheader("Advanced Options Section")
    return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    show_full_doc = st.checkbox("Show parsed contents of the document")

