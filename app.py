import streamlit as st
from src.sbert_model import SBERTModel

# Load model once
@st.cache_resource
def load_model():
    return SBERTModel()

model = load_model()

st.title("🧠 Paraphrase Detection & Semantic Similarity")

st.write("Enter two sentences to check similarity and paraphrase detection.")

# Input
sentence1 = st.text_input("Sentence 1")
sentence2 = st.text_input("Sentence 2")

if st.button("Check Similarity"):
    if sentence1 and sentence2:
        score = model.similarity(sentence1, sentence2)

        # Threshold for paraphrase
        threshold = 0.75
        is_para = score > threshold

        st.subheader("🔍 Results")

        st.write(f"**Similarity Score:** {score:.4f}")

        if is_para:
            st.success("✅ These sentences are paraphrases")
        else:
            st.error("❌ These sentences are NOT paraphrases")

    else:
        st.warning("Please enter both sentences!")