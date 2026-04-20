import streamlit as st
from src.sbert_model import SBERTModel

st.title("🧠 Paraphrase Detection & Semantic Similarity")

st.write("Enter two sentences to check similarity and paraphrase detection.")

# Model Selection
model_choice_ui = st.selectbox(
    "Choose the NLP Model",
    [
        "MiniLM (Fast & Lightweight - all-MiniLM-L6-v2)", 
        "MPNet (Balanced & High Quality - all-mpnet-base-v2)", 
        "DeBERTa-V3 / Elite (Maximum Precision Cross-Encoder)"
    ]
)

if "MiniLM" in model_choice_ui:
    model_name = "MiniLM"
elif "MPNet" in model_choice_ui:
    model_name = "MPNet"
else:
    model_name = "Elite"

# Load model based on selection
@st.cache_resource
def load_model(model_name):
    return SBERTModel(model_name)

model = load_model(model_name)

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