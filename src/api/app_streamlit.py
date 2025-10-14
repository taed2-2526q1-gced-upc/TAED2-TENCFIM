import streamlit as st
import requests

# URL de tu API FastAPI (ajusta el puerto si lo corres distinto)
API_URL = "http://127.0.0.1:5000/prediction"

st.set_page_config(page_title="Emotion Classifier", page_icon="😊", layout="centered")

st.title("🎭 Emotion Classification App")
st.write("Introduce una o varias frases y la API predecirá la emoción asociada.")

# Entrada de texto multilinea
input_text = st.text_area(
    "Escribe tus frases (una por línea):",
    height=200,
    placeholder="Ejemplo:\nI am so happy today!\nThis is terrible...\nI love programming!",
)

# Botón para enviar a la API
if st.button("Predecir emociones"):
    if not input_text.strip():
        st.warning("⚠️ Por favor, introduce al menos una frase.")
    else:
        # Dividir en líneas y eliminar vacías
        texts = [line.strip() for line in input_text.splitlines() if line.strip()]
        
        payload = {"texts": [{"text": t} for t in texts]}
        
        try:
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                results = response.json()
                
                st.success("✅ Resultados de la prediccion")
                
                # Mostrar como tabla
                st.dataframe(results, use_container_width=True)
                
                # Opcional: mostrar cada resultado como tarjeta
                for res in results:
                    st.markdown(
                        f"""
                        **Texto:** {res['text']}  
                        **Emoción:** 🎭 {res['label']}  
                        **Score:** {res['score']:.2f}  
                        ---
                        """
                    )
            else:
                st.error(f"❌ Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"⚠️ No se pudo conectar con la API: {e}")
