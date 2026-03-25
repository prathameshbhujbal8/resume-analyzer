import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- UI -------------------- #
st.title("📄 AI Resume Analyzer")
st.write("Analyze how well your resume matches a job description using NLP")
st.markdown("---")

# Input fields
resume = st.text_area("📌 Paste your Resume")
jd = st.text_area("📌 Paste Job Description")

# -------------------- FUNCTIONS -------------------- #

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# Suggestion function
def generate_suggestions(missing_skills):
    suggestions = []

    if "sql" in missing_skills:
        suggestions.append("Learn SQL for data handling")

    if "deep" in missing_skills:
        suggestions.append("Add Deep Learning projects")

    if "machine" in missing_skills:
        suggestions.append("Include Machine Learning experience")

    if "python" in missing_skills:
        suggestions.append("Improve Python skills")

    return suggestions


# -------------------- MAIN LOGIC -------------------- #

if st.button("🔍 Analyze"):

    # Handle empty input
    if not resume or not jd:
        st.warning("⚠️ Please enter both Resume and Job Description")
        st.stop()

    # Clean text
    resume_clean = clean_text(resume)
    jd_clean = clean_text(jd)

    # TF-IDF Vectorization
    documents = [resume_clean, jd_clean]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Cosine Similarity
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    score = similarity[0][0] * 100

    # Tokenization
    resume_words = set(resume_clean.split())
    jd_words = set(jd_clean.split())

    # Missing skills
    missing_skills = jd_words - resume_words

    # Filter important skills
    important_words = {"python", "sql", "machine", "learning", "deep"}
    missing_filtered = [word for word in missing_skills if word in important_words]

    # Suggestions
    suggestions = generate_suggestions(missing_filtered)

    # -------------------- OUTPUT -------------------- #

    st.subheader("📊 Match Score")
    st.progress(int(score))
    st.success(f"Match Score: {round(score, 2)}%")

    # Feedback based on score
    if score > 75:
        st.success("🔥 Strong Match! You are a great fit.")
    elif score > 50:
        st.info("🙂 Moderate match — you can improve.")
    else:
        st.error("⚠️ Low match — improve your resume.")

    st.markdown("---")

    st.subheader("❌ Missing Skills")

    if missing_filtered:
        st.write(", ".join(missing_filtered))
    else:
        st.write("No major missing skills 🎉")

    st.markdown("---")

    st.subheader("💡 Suggestions")

    if suggestions:
        for s in suggestions:
            st.write("✔️ " + s)
    else:
        st.write("Your resume looks strong! 🚀")