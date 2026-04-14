import streamlit as st
import pickle
import re
import PyPDF2

# -----------------------------
# Load model and vectorizer
# -----------------------------
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# -----------------------------
# Category Mapping (FIXED)
# -----------------------------
category_mapping = {
    0: 'Backend Developer',
    1: 'Cloud Engineer',
    2: 'Data Scientist',
    3: 'Frontend Developer',
    4: 'Full Stack Developer',
    5: 'Machine Learning Engineer',
    6: 'Mobile App Developer',
    7: 'Python Developer'
}

# -----------------------------
# Clean Resume Function
# -----------------------------
def CleanResume(txt):
    txt = re.sub(r'http\S+\s*', ' ', txt)
    txt = re.sub(r'RT|cc', ' ', txt)
    txt = re.sub(r'#\S+', '', txt)
    txt = re.sub(r'@\S+', '  ', txt)
    txt = re.sub(r'[^A-Za-z0-9 ]+', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt

# -----------------------------
# Extract Text from PDF
# -----------------------------
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():  # avoid None error
            text += page.extract_text()
    return text

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Resume Classifier", page_icon="📄")

st.title("📄 Resume Category Predictor")
st.write("Upload your resume or paste text to predict job role 🚀")

# File Upload
uploaded_file = st.file_uploader("Upload Resume", type=["txt", "pdf"])

# Text Input
resume_text = st.text_area("Or paste your resume here:")

# Button
if st.button("Predict Category"):

    # If file uploaded
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = uploaded_file.read().decode("utf-8")

    # Check empty
    if resume_text.strip() == "":
        st.warning("Please upload or enter resume ⚠️")

    else:
        # Clean text
        cleaned = CleanResume(resume_text)

        # Transform
        input_features = tfidf.transform([cleaned])

        # Predict
        prediction = clf.predict(input_features)[0]

        # Map to category name
        category_name = category_mapping.get(prediction, "Unknown")

        # Output
        st.success(f"🎯 Predicted Category: {category_name}")