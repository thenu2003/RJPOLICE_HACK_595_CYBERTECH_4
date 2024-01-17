import streamlit as st
from pdf_extractor import process_uploaded_file
import speech_recognition as sr
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import tempfile
from IPC_ID_NLP import train_ipc_model, predict_ipc_section
from CRPC_ID_NLP import train_crpc_model, predict_crpc_section

# Function to detect language using langdetect library
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# Function to transcribe audio and get the transcribed text
def transcribe_audio():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("Speak something...")
        audio = recognizer.listen(source)
        st.success("Audio recorded successfully!")

    try:
        st.subheader("Transcription:")
        text = recognizer.recognize_google(audio)
        st.write(text)
        return text
    except sr.UnknownValueError:
        st.warning("Speech Recognition could not understand audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")

# Load IPC data
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

# Load CRPC data
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

# Streamlit UI
st.set_page_config(page_title="LegalAssist", page_icon="🔍", layout="wide")
st.header("🔍 LegalAssist - AI for Legal Section Suggestions")

from rajastan.components.sidebar import sidebar
from rajastan.core.caching import bootstrap_caching

# Enable caching for expensive functions
bootstrap_caching()

sidebar()

# 1) Prompt entering section with Audio icon
with st.form(key="prompt_form"):
    st.subheader("Prompt Section")

    # Create a microphone icon using Unicode
    audio_icon = "🎤"

    # Add an audio button to switch input mode
    audio_button_clicked = st.form_submit_button(audio_icon)
    
    # Use transcribed text as the default input for the prompt if the audio button is clicked
    transcribed_text = transcribe_audio() if audio_button_clicked else None
    gen_inp = st.text_input("Enter some text:", transcribed_text if transcribed_text else "Default Text")
    
    st.write("You entered:", gen_inp)
    prompt_submit = st.form_submit_button("Submit Prompt")

    # Display audio icon button status
    st.write("Audio Recording Status:", "Recording..." if audio_button_clicked else "Not Recording")

# Check if the audio icon button is clicked
#if audio_button_clicked:
    # Start transcribing the recorded audio
   # transcribed_text = transcribe_audio()
   # st.subheader("Transcription:")
   # st.write(transcribed_text)
   # st.text("You entered: " + transcribed_text)

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

    if file_submit and uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Process the uploaded file
        text_result = process_uploaded_file(temp_file_path)

        # Display extracted text
        if text_result:
            for i, text in enumerate(text_result, start=1):
                st.text(f"Page {i}:\n{text}\n")
        else:
            st.warning("Unsupported file type. Please upload a PDF.")

        # Clean up the temporary file after processing
        os.unlink(temp_file_path)

# 3) Advanced sections
with st.expander("Advanced Options"):
    st.subheader("Advanced Options Section")
    return_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    show_full_doc = st.checkbox("Show parsed contents of the document")