import streamlit as st
from src.sbert_model import SBERTModel
from datasets import load_dataset
import random

# Page configuration
st.set_page_config(
    page_title="Paraphrase Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🧠 Paraphrase Detection & Semantic Similarity")
st.markdown("""
Analyze semantic similarity and detect paraphrases between text pairs using 
state-of-the-art transformer models. Compare predictions across different architectures.
""")

# ============================================================================
# SIDEBAR: Model Selection with Enhanced UI
# ============================================================================
st.sidebar.header("⚙️ Model Configuration")

model_descriptions = {
    "MiniLM": "Lightweight & Fast - Best for real-time applications with lower latency requirements",
    "MPNet": "Balanced Quality - High accuracy with moderate computational cost",
    "Elite": "Maximum Precision - DeBERTa Cross-Encoder for deep semantic reasoning"
}

model_choice = st.sidebar.radio(
    "**Choose the NLP Model**",
    options=["MiniLM", "MPNet", "Elite"],
    format_func=lambda x: f"{x}\n└─ {model_descriptions[x]}",
    help="Select a model based on your accuracy/speed requirements"
)

# Display model details in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Selected Model Details")
with st.sidebar.container():
    if model_choice == "MiniLM":
        st.sidebar.info(
            "**Model:** all-MiniLM-L6-v2\n\n"
            "• 6 layers, 22M parameters\n"
            "• ~5-10ms per inference\n"
            "• Best for: Speed-critical applications"
        )
    elif model_choice == "MPNet":
        st.sidebar.info(
            "**Model:** all-mpnet-base-v2\n\n"
            "• 12 layers, 109M parameters\n"
            "• ~20-30ms per inference\n"
            "• Best for: Balanced applications"
        )
    else:  # Elite
        st.sidebar.info(
            "**Model:** cross-encoder/stsb-roberta-large\n\n"
            "• Cross-Encoder architecture, 355M parameters\n"
            "• ~50-100ms per inference\n"
            "• Best for: Highest accuracy requirements"
        )

# ============================================================================
# Load model
# ============================================================================
@st.cache_resource
def load_model(model_name):
    return SBERTModel(model_name)

model = load_model(model_choice)

# ============================================================================
# SAMPLE DATA SECTION
# ============================================================================
st.markdown("---")
st.subheader("📚 Sample Sentence Pairs")
st.markdown("Click on any sample below to auto-populate the input fields")

@st.cache_data
def load_sample_pairs():
    """Load sample pairs from GLUE datasets (MRPC and STSb)"""
    samples = []
    
    try:
        # Load MRPC (Paraphrase Detection)
        mrpc = load_dataset("glue", "mrpc", split="validation")
        mrpc_samples = random.sample(range(len(mrpc)), min(3, len(mrpc)))
        
        for idx in mrpc_samples:
            sample = mrpc[int(idx)]
            samples.append({
                "sentence1": sample["sentence1"],
                "sentence2": sample["sentence2"],
                "label": sample["label"],
                "score": 1.0 if sample["label"] == 1 else 0.0,
                "source": "MRPC (Paraphrase Detection)"
            })
        
        # Load STSb (Semantic Similarity)
        stsb = load_dataset("glue", "stsb", split="validation")
        stsb_samples = random.sample(range(len(stsb)), min(2, len(stsb)))
        
        for idx in stsb_samples:
            sample = stsb[int(idx)]
            samples.append({
                "sentence1": sample["sentence1"],
                "sentence2": sample["sentence2"],
                "label": 1 if sample["label"] >= 3.5 else 0,  # Convert score to label
                "score": sample["label"] / 5.0,  # Normalize to 0-1
                "source": "STSb (Semantic Similarity)"
            })
    except Exception as e:
        # Fallback to hardcoded examples if dataset loading fails
        st.warning(f"Could not load datasets: {e}. Using example samples instead.")
        samples = [
            {
                "sentence1": "A plane is taking off with a full load of people.",
                "sentence2": "An air plane is taking off with a full load of people.",
                "label": 1,
                "score": 0.95,
                "source": "Example (High Similarity)"
            },
            {
                "sentence1": "The cat sat on the mat.",
                "sentence2": "The feline rested on the rug.",
                "label": 1,
                "score": 0.78,
                "source": "Example (Paraphrase)"
            },
            {
                "sentence1": "A woman is playing the violin.",
                "sentence2": "A man is driving a motorcycle.",
                "label": 0,
                "score": 0.15,
                "source": "Example (Low Similarity)"
            },
            {
                "sentence1": "The price of gold went up.",
                "sentence2": "The cost of gold increased.",
                "label": 1,
                "score": 0.88,
                "source": "Example (Synonym Paraphrase)"
            },
            {
                "sentence1": "She loves cooking Italian food.",
                "sentence2": "He dislikes eating spicy dishes.",
                "label": 0,
                "score": 0.25,
                "source": "Example (Opposite Sentiment)"
            }
        ]
    
    return samples

samples = load_sample_pairs()

# Display sample pairs as clickable cards
sample_cols = st.columns(1)
with sample_cols[0]:
    for idx, sample in enumerate(samples):
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**Sample {idx + 1}** · *{sample['source']}*")
                st.text(f"S1: {sample['sentence1'][:80]}...")
                st.text(f"S2: {sample['sentence2'][:80]}...")
            
            with col2:
                similarity_badge = f"**Score:** {sample['score']:.2f}"
                st.markdown(similarity_badge)
            
            with col3:
                if st.button(
                    "Use Sample",
                    key=f"sample_{idx}",
                    use_container_width=True,
                    help="Populate input fields with this sample"
                ):
                    st.session_state.sentence1_input = sample["sentence1"]
                    st.session_state.sentence2_input = sample["sentence2"]
                    st.success("✅ Sample loaded! Scroll down to check similarity.")
                    st.rerun()

# ============================================================================
# USER INPUT SECTION
# ============================================================================
st.markdown("---")
st.subheader("🔍 Check Similarity")
st.markdown("Enter or paste your text pairs below:")

# Initialize session state for inputs
if "sentence1_input" not in st.session_state:
    st.session_state.sentence1_input = ""
if "sentence2_input" not in st.session_state:
    st.session_state.sentence2_input = ""

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    sentence1 = st.text_area(
        "Paragraph 1",
        value=st.session_state.sentence1_input,
        height=150,
        placeholder="Enter the first text...",
        help="Paste your first paragraph or sentence here"
    )
    st.session_state.sentence1_input = sentence1

with col2:
    sentence2 = st.text_area(
        "Paragraph 2",
        value=st.session_state.sentence2_input,
        height=150,
        placeholder="Enter the second text...",
        help="Paste your second paragraph or sentence here"
    )
    st.session_state.sentence2_input = sentence2

# Action buttons
button_col1, button_col2, button_col3 = st.columns([2, 1, 1])

with button_col1:
    if st.button("✨ Check Similarity", use_container_width=True, type="primary"):
        if sentence1.strip() and sentence2.strip():
            st.session_state.last_result = {
                "sentence1": sentence1,
                "sentence2": sentence2,
                "run": True
            }
        else:
            st.warning("⚠️ Please enter both paragraphs before checking!")

with button_col2:
    if st.button("🔄 Clear", use_container_width=True):
        st.session_state.sentence1_input = ""
        st.session_state.sentence2_input = ""
        st.rerun()

# ============================================================================
# RESULTS SECTION
# ============================================================================
if "last_result" in st.session_state and st.session_state.last_result.get("run"):
    result = st.session_state.last_result
    sentence1 = result["sentence1"]
    sentence2 = result["sentence2"]
    
    # Perform inference
    with st.spinner(f"🔄 Analyzing with {model_choice} model..."):
        score = model.similarity(sentence1, sentence2)
    
    # Determine paraphrase classification
    threshold = 0.75
    is_paraphrase = score > threshold
    
    # Display results in organized sections
    st.markdown("---")
    st.subheader("📊 Results")
    
    # Main result cards
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.metric(
            label="Similarity Score",
            value=f"{score:.4f}",
            help="Normalized similarity score from 0 (dissimilar) to 1 (identical)"
        )
    
    with result_col2:
        paraphrase_status = "✅ YES" if is_paraphrase else "❌ NO"
        st.metric(
            label="Paraphrase?",
            value=paraphrase_status,
            help=f"Classification based on threshold of {threshold}"
        )
    
    with result_col3:
        model_info = {
            "MiniLM": "Fast",
            "MPNet": "Balanced",
            "Elite": "Precise"
        }
        st.metric(
            label="Model Used",
            value=model_choice,
            delta=model_info[model_choice],
            help=model_descriptions[model_choice]
        )
    
    # Detailed interpretation
    st.markdown("---")
    st.subheader("💡 Interpretation")
    
    with st.container(border=True):
        if score >= 0.9:
            st.markdown("""
            **🎯 Nearly Identical:** The texts are nearly identical or very close paraphrases.
            They convey the same meaning with minimal lexical differences.
            """)
        elif score >= 0.75:
            st.markdown("""
            **✅ Strong Paraphrase:** The texts are clear paraphrases of each other.
            They convey essentially the same meaning despite different wording.
            """)
        elif score >= 0.5:
            st.markdown("""
            **⚠️ Partial Similarity:** The texts share some semantic concepts but differ in meaning or focus.
            They are related but not paraphrases.
            """)
        elif score >= 0.25:
            st.markdown("""
            **📍 Low Similarity:** The texts have minimal semantic overlap.
            They discuss different topics or perspectives.
            """)
        else:
            st.markdown("""
            **❌ Dissimilar:** The texts are semantically unrelated.
            They convey different meanings and concepts.
            """)
    
    # Input preview
    st.markdown("---")
    st.subheader("📝 Input Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("**Paragraph 1**")
        st.text(sentence1)
    
    with col2:
        st.caption("**Paragraph 2**")
        st.text(sentence2)
    
    st.session_state.last_result["run"] = False