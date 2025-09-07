import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="Análisis de Sentimientos con Qwen2", layout="centered")
st.title("🎭 Análisis de Sentimientos con Qwen2-0.5B-Instruct")

MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    return tokenizer, model

tokenizer, model = load_model()

default_examples = [
    ("Esta película fue increíble, la mejor que he visto.", "positivo"),
    ("No me gustó el final, esperaba más.", "negativo"),
    ("La cinematografía estuvo bien, pero la historia es regular.", "neutro"),
]

st.markdown("### Selecciona el modo de análisis:")
mode = st.radio("Modo", ["Zero-shot", "Few-shot"])

if mode == "Few-shot":
    st.markdown("#### Ejemplos de entrenamiento:")
    example_inputs = []
    for idx, (text, label) in enumerate(default_examples):
        col1, col2 = st.columns([3, 1])
        with col1:
            input_text = st.text_input(f"{idx+1}. Reseña", value=text, key=f"example_{idx}_text")
        with col2:
            input_label = st.selectbox(
                f"Sentimiento", 
                options=["positivo", "negativo", "neutro"], 
                index=["positivo", "negativo", "neutro"].index(label),
                key=f"example_{idx}_label"
            )
        example_inputs.append((input_text, input_label))

user_review = st.text_area("Introduce tu reseña de película aquí:")

if st.button("🔍 Analizar Sentimiento"):
    if not user_review.strip():
        st.warning("Por favor escribe una reseña para analizar.")
    else:
        if mode == "Few-shot":
            prompt = "Clasifica el sentimiento (positivo, negativo o neutro) de la reseña. Ejemplos:\n"
            for text, label in example_inputs:
                prompt += f"Reseña: {text}\nSentimiento: {label}\n"
            prompt += f"Reseña: {user_review}\nSentimiento:"
        else:
            prompt = f"Clasifica el sentimiento (positivo, negativo o neutro) de la siguiente reseña de película:\nReseña: {user_review}\nSentimiento:"

        with st.spinner("Analizando..."):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output = model.generate(**inputs, max_new_tokens=10)
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            sentiment = decoded.split("Sentimiento:")[-1].strip().split()[0].lower()

            if "positivo" in sentiment:
                emoji="😊"
                label="Positivo"
            elif "negativo" in sentiment:
                emoji="😞"
                label="Negativo"
            elif "neutro" in sentiment:
                emoji="😐"
                label="Neutro"
            else:
                emoji="❓"
                label="Impreciso"

            st.success(f"**Sentimiento detectado:** {emoji} {label}")
