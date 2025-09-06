import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Configurar la página de Streamlit
st.set_page_config(
    page_title="Clasificación de Sentimientos con Qwen2-0.5B",
    page_icon="🎭",
    layout="wide"
)

# Título principal
st.title("🎭 Clasificación de Sentimientos con Qwen2-0.5B")
st.markdown("**Análisis de sentimientos usando el modelo Qwen2-0.5B con enfoques zero-shot y few-shot**")

# Cache del modelo para mejorar el rendimiento
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None

def zero_shot_sentiment(text, tokenizer, model):
    """
    Clasificación zero-shot de sentimientos
    """
    prompt = f"""Analiza el sentimiento del siguiente texto y responde solo con una de estas opciones: POSITIVO, NEGATIVO, o NEUTRO.

Texto: "{text}"
Sentimiento:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=10,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extraer solo la respuesta después del prompt
    response = response[len(prompt):].strip()
    
    # Buscar palabras clave en la respuesta
    if re.search(r'\bPOSITIVO\b|\bpositivo\b|\bPOSITIVE\b|\bpositive\b', response, re.IGNORECASE):
        return "POSITIVO", response
    elif re.search(r'\bNEGATIVO\b|\bnegativo\b|\bNEGATIVE\b|\bnegative\b', response, re.IGNORECASE):
        return "NEGATIVO", response
    elif re.search(r'\bNEUTRO\b|\bneutro\b|\bNEUTRAL\b|\bneutral\b', response, re.IGNORECASE):
        return "NEUTRO", response
    else:
        return "INDETERMINADO", response

def few_shot_sentiment(text, tokenizer, model):
    """
    Clasificación few-shot de sentimientos con ejemplos
    """
    prompt = f"""Aquí hay algunos ejemplos de clasificación de sentimientos:

Ejemplo 1:
Texto: "Me encanta este producto, es fantástico!"
Sentimiento: POSITIVO

Ejemplo 2:
Texto: "Este servicio es terrible, muy decepcionante."
Sentimiento: NEGATIVO

Ejemplo 3:
Texto: "El clima está nublado hoy."
Sentimiento: NEUTRO

Ahora analiza este texto y responde solo con: POSITIVO, NEGATIVO, o NEUTRO.

Texto: "{text}"
Sentimiento:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=10,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extraer solo la respuesta después del prompt
    response = response[len(prompt):].strip()
    
    # Buscar palabras clave en la respuesta
    if re.search(r'\bPOSITIVO\b|\bpositivo\b|\bPOSITIVE\b|\bpositive\b', response, re.IGNORECASE):
        return "POSITIVO", response
    elif re.search(r'\bNEGATIVO\b|\bnegativo\b|\bNEGATIVE\b|\bnegative\b', response, re.IGNORECASE):
        return "NEGATIVO", response
    elif re.search(r'\bNEUTRO\b|\bneutro\b|\bNEUTRAL\b|\bneutral\b', response, re.IGNORECASE):
        return "NEUTRO", response
    else:
        return "INDETERMINADO", response

def main():
    # Cargar el modelo
    with st.spinner("Cargando modelo Qwen2-0.5B..."):
        tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("No se pudo cargar el modelo. Por favor, verifica la instalación.")
        return
    
    st.success("✅ Modelo cargado correctamente")
    
    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información")
        st.markdown("""
        **Modelo:** Qwen2-0.5B  
        **Desarrollado por:** Alibaba Cloud  
        **Capacidades:**
        - Zero-shot: Sin ejemplos previos
        - Few-shot: Con ejemplos de entrenamiento
        
        **Categorías de sentimiento:**
        - 🟢 POSITIVO
        - 🔴 NEGATIVO  
        - 🟡 NEUTRO
        """)
    
    # Crear dos columnas para los métodos
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🎯 Zero-Shot")
        st.markdown("*Análisis sin ejemplos previos*")
        
        text_zero = st.text_area(
            "Introduce el texto a analizar:",
            key="zero_text",
            height=100,
            placeholder="Ejemplo: 'Hoy ha sido un día maravilloso'"
        )
        
        if st.button("Analizar (Zero-Shot)", type="primary"):
            if text_zero.strip():
                with st.spinner("Analizando sentimiento..."):
                    sentiment, full_response = zero_shot_sentiment(text_zero, tokenizer, model)
                
                # Mostrar resultado con color
                if sentiment == "POSITIVO":
                    st.success(f"🟢 **Sentimiento:** {sentiment}")
                elif sentiment == "NEGATIVO":
                    st.error(f"🔴 **Sentimiento:** {sentiment}")
                elif sentiment == "NEUTRO":
                    st.warning(f"🟡 **Sentimiento:** {sentiment}")
                else:
                    st.info(f"❓ **Sentimiento:** {sentiment}")
                
                with st.expander("Ver respuesta completa del modelo"):
                    st.text(full_response)
            else:
                st.warning("Por favor, introduce un texto para analizar.")
    
    with col2:
        st.header("📚 Few-Shot")
        st.markdown("*Análisis con ejemplos de entrenamiento*")
        
        text_few = st.text_area(
            "Introduce el texto a analizar:",
            key="few_text",
            height=100,
            placeholder="Ejemplo: 'Este producto es decepcionante'"
        )
        
        if st.button("Analizar (Few-Shot)", type="primary"):
            if text_few.strip():
                with st.spinner("Analizando sentimiento..."):
                    sentiment, full_response = few_shot_sentiment(text_few, tokenizer, model)
                
                # Mostrar resultado con color
                if sentiment == "POSITIVO":
                    st.success(f"🟢 **Sentimiento:** {sentiment}")
                elif sentiment == "NEGATIVO":
                    st.error(f"🔴 **Sentimiento:** {sentiment}")
                elif sentiment == "NEUTRO":
                    st.warning(f"🟡 **Sentimiento:** {sentiment}")
                else:
                    st.info(f"❓ **Sentimiento:** {sentiment}")
                
                with st.expander("Ver respuesta completa del modelo"):
                    st.text(full_response)
            else:
                st.warning("Por favor, introduce un texto para analizar.")
    
    # Sección de comparación
    st.header("⚖️ Comparación de Métodos")
    
    compare_text = st.text_area(
        "Texto para comparar ambos métodos:",
        height=100,
        placeholder="Introduce un texto para ver cómo lo clasifica cada método"
    )
    
    if st.button("Comparar Métodos", type="secondary"):
        if compare_text.strip():
            col_comp1, col_comp2 = st.columns(2)
            
            with col_comp1:
                st.subheader("Zero-Shot")
                with st.spinner("Analizando..."):
                    zero_sentiment, zero_response = zero_shot_sentiment(compare_text, tokenizer, model)
                if zero_sentiment == "POSITIVO":
                    st.success(f"🟢 {zero_sentiment}")
                elif zero_sentiment == "NEGATIVO":
                    st.error(f"🔴 {zero_sentiment}")
                else:
                    st.warning(f"🟡 {zero_sentiment}")
            
            with col_comp2:
                st.subheader("Few-Shot")
                with st.spinner("Analizando..."):
                    few_sentiment, few_response = few_shot_sentiment(compare_text, tokenizer, model)
                if few_sentiment == "POSITIVO":
                    st.success(f"🟢 {few_sentiment}")
                elif few_sentiment == "NEGATIVO":
                    st.error(f"🔴 {few_sentiment}")
                else:
                    st.warning(f"🟡 {few_sentiment}")
            
            # Mostrar si coinciden
            if zero_sentiment == few_sentiment:
                st.success("✅ Ambos métodos coinciden en el resultado")
            else:
                st.warning("⚠️ Los métodos difieren en el resultado")
        else:
            st.warning("Por favor, introduce un texto para comparar.")
    
    # Ejemplos predefinidos
    st.header("💡 Ejemplos para Probar")
    
    examples = [
        "Me encanta este nuevo restaurante, la comida es excelente",
        "El servicio al cliente fue muy decepcionante y lento",
        "El informe del tiempo dice que llueve mañana",
        "¡Qué día tan fantástico! Todo ha salido perfecto",
        "No me gustó nada la película, fue aburrida"
    ]
    
    example_choice = st.selectbox("Selecciona un ejemplo:", ["Elige un ejemplo..."] + examples)
    
    if example_choice != "Elige un ejemplo...":
        st.text_area("Texto seleccionado:", value=example_choice, height=80, disabled=True)
        
        if st.button("Analizar Ejemplo"):
            col_ex1, col_ex2 = st.columns(2)
            
            with col_ex1:
                st.subheader("Zero-Shot")
                zero_sentiment, _ = zero_shot_sentiment(example_choice, tokenizer, model)
                if zero_sentiment == "POSITIVO":
                    st.success(f"🟢 {zero_sentiment}")
                elif zero_sentiment == "NEGATIVO":
                    st.error(f"🔴 {zero_sentiment}")
                else:
                    st.warning(f"🟡 {zero_sentiment}")
            
            with col_ex2:
                st.subheader("Few-Shot")
                few_sentiment, _ = few_shot_sentiment(example_choice, tokenizer, model)
                if few_sentiment == "POSITIVO":
                    st.success(f"🟢 {few_sentiment}")
                elif few_sentiment == "NEGATIVO":
                    st.error(f"🔴 {few_sentiment}")
                else:
                    st.warning(f"🟡 {few_sentiment}")

if __name__ == "__main__":
    main()
