import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

skills_list = [
    "python", "java", "sql", "excel", "machine learning",
    "deep learning", "data science", "pandas", "numpy",
    "matplotlib", "scikit-learn", "nlp"
]

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄")

st.title("📄 AI Resume Analyzer")
st.write("Analyze how well your resume matches a job description using NLP")

resume = st.text_area("Paste your Resume")
jd = st.text_area("Paste Job Description")

if st.button("Analyze"):

    if resume == "" or jd == "":
        st.warning("Please enter both Resume and Job Description")
    else:
        resume_clean = clean_text(resume)
        jd_clean = clean_text(jd)

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_clean, jd_clean])

        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        score = round(similarity * 100, 2)

        st.subheader("📊 Match Score")
        st.progress(int(score))

        if score > 70:
            st.success(f"Match Score: {score}% (Strong match)")
        elif score > 40:
            st.warning(f"Match Score: {score}% (Moderate match)")
        else:
            st.error(f"Match Score: {score}% (Low match)")

        resume_words = set(resume_clean.split())
        jd_words = set(jd_clean.split())

        matched_skills = []
        missing_skills = []

        for skill in skills_list:
            if skill in jd_clean:
                if skill in resume_clean:
                    matched_skills.append(skill)
                else:
                    missing_skills.append(skill)

        st.subheader("✅ Matched Skills")
        if matched_skills:
            st.write(", ".join(matched_skills))
        else:
            st.write("No strong matches found")

        st.subheader("❌ Missing Skills")
        if missing_skills:
            st.write(", ".join(missing_skills))
        else:
            st.write("No major missing skills")

        st.subheader("💡 Suggestions")

        if score > 70:
            st.success("Your resume is well aligned with the job description.")
        elif score > 40:
            st.warning("Improve your resume by adding more relevant skills and experience.")
        else:
            st.error("Consider adding key skills mentioned in the job description to improve your chances.")