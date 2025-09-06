# Text Generation Models

Este repositorio contiene notebooks de Jupyter y una aplicación Streamlit que demuestran varias capacidades de generación y procesamiento de texto usando modelos transformer.

## 🎆 Características

- **Generación de texto** con el modelo Qwen2-0.5B
- **Clasificación zero-shot** para análisis de sentimientos
- **Aprendizaje few-shot** para:
  - Traducción de texto (Inglés a Español)
  - Análisis de sentimientos mejorado
- **Aplicación web interactiva** con Streamlit

## 📱 Aplicación Streamlit - Clasificación de Sentimientos

La aplicación web `app.py` proporciona una interfaz fácil de usar para clasificar sentimientos usando el modelo Qwen2-0.5B con dos enfoques:

### 🎯 Zero-Shot
- Análisis de sentimientos sin ejemplos previos
- Clasificación directa en: POSITIVO, NEGATIVO, o NEUTRO

### 📚 Few-Shot  
- Análisis con ejemplos de entrenamiento
- Mayor precisión mediante ejemplos contextuales

### ⚖️ Funcionalidades
- Comparación lado a lado de ambos métodos
- Ejemplos predefinidos para probar
- Interfaz intuitiva con colores para cada tipo de sentimiento
- Visualización de respuestas completas del modelo

## 🛠️ Instalación

### Requisitos
- Python 3.11 o superior
- Al menos 4GB de RAM libre
- Conexión a Internet (para descargar el modelo la primera vez)

### Pasos de instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/RicardoAmano04/text-generation-models.git
cd text-generation-models
```

2. **Crear un entorno virtual (recomendado):**
```bash
# Con conda
conda create -n text_models python=3.11
conda activate text_models

# O con venv
python -m venv text_models
source text_models/bin/activate  # En Linux/Mac
# text_models\Scripts\activate  # En Windows
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

## 🚀 Uso

### Ejecutar la aplicación Streamlit

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

**Nota:** La primera ejecución puede tardar varios minutos mientras se descarga el modelo Qwen2-0.5B (≈500MB).

### Ejecutar los notebooks de Jupyter

```bash
jupyter notebook
```

Abre `text_models.ipynb` para explorar los ejemplos de generación de texto.

## 📚 Archivos del proyecto

- `app.py` - Aplicación principal de Streamlit para clasificación de sentimientos
- `requirements.txt` - Dependencias necesarias del proyecto
- `text_models.ipynb` - Notebook con ejemplos de generación de texto
- `README.md` - Este archivo con documentación del proyecto

## 🧪 Ejemplos de uso de la aplicación

1. **Análisis básico:** Introduce cualquier texto en español y obtén su clasificación de sentimiento
2. **Comparación de métodos:** Usa la sección de comparación para ver cómo difieren los enfoques zero-shot y few-shot
3. **Ejemplos predefinidos:** Selecciona ejemplos del menú desplegable para probar rápidamente

## 📊 Modelo utilizado

**Qwen2-0.5B** por Alibaba Cloud:
- Modelo de lenguaje pequeño pero eficiente
- Soporte multiidioma incluido español
- Optimizado para tareas de clasificación de texto
- Ejecución rápida en hardware convencional

## ⚠️ Consideraciones

- La primera ejecución requiere descarga del modelo (≈500MB)
- Los tiempos de respuesta pueden variar según el hardware
- Para mejor rendimiento, se recomienda usar una GPU si está disponible
- El modelo puede funcionar en CPU, pero será más lento

## 🔧 Solución de problemas

### Error de memoria insuficiente
```bash
# Reducir el uso de memoria limitando el tamaño del modelo
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Error de dependencias
```bash
# Reinstalar dependencias
pip install --upgrade -r requirements.txt
```

### Problemas con el modelo
```bash
# Limpiar caché de transformers
rm -rf ~/.cache/huggingface/
```

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si encuentras algún problema o tienes ideas para mejoras:

1. Abre un issue describiendo el problema o sugerencia
2. Haz fork del repositorio
3. Crea una nueva rama para tu feature
4. Envía un pull request

## 📋 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

---

💬 **¿Preguntas?** Abre un issue o contacta al desarrollador.

🌟 **¿Te gustó el proyecto?** ¡Dale una estrella al repositorio!
