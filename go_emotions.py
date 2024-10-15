import streamlit as st
from transformers import pipeline

# Load the emotion classifier
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Sample abstracts
sample_dict = {
    "Abstract 1": "Many researchers believe that PTLDS symptoms actually have nothing to do with Lyme disease, attributing these symptoms instead to other underlying conditions such as chronic fatigue syndrome, fibromyalgia, or autoimmune disorders.",
    "Abstract 2": "PTLDS is caused by clusters of Lyme bacteria that resist traditional methods of antibiotic treatment, leading to persistent symptoms due to ongoing infection and inflammation.",
    "Abstract 3": "The hypothesis of PTLDS has been thoroughly refuted, with evidence suggesting that the symptoms often attributed to this condition are more likely caused by alternative diagnoses such as immune dysfunction, chronic fatigue syndrome, or psychological factors."
}

# Streamlit app layout
st.title("Text Emotion Analysis")

# Option for user to choose between sample abstracts or custom text input
analysis_type = st.radio("Choose input type:", ("Select from sample abstracts", "Enter your own text"))

# Variable to hold the text for analysis
input_text = ""

# If user selects a sample abstract
if analysis_type == "Select from sample abstracts":
    selected_abstract = st.selectbox("Choose an abstract to analyze:", list(sample_dict.keys()))
    input_text = sample_dict[selected_abstract]
    st.write(f"**Selected Abstract:** {input_text}")

# If user chooses to enter their own text
elif analysis_type == "Enter your own text":
    input_text = st.text_area("Enter your text here:")

# Button to submit the input
if st.button("Submit"):
    if input_text:
        # Analyze the input text
        result = emotion_classifier([input_text])
        
        # Extract label and score
        label = result[0]['label']
        score = result[0]['score']

        # Display results in a user-friendly format
        st.subheader("Analysis Result:")
        st.write(f"**Emotion Detected:** {label}")
        st.write(f"**Confidence Score:** {score:.2f}")  # Format score to 2 decimal places
    else:
        st.warning("Please enter some text or select a sample before submitting.")

