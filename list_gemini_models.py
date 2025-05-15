from google import genai
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY no encontrada. Asegúrate de tenerla configurada en tu archivo .env")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)

print("Listando modelos de Google Gemini disponibles para tu clave API:")

try:
    # Listar todos los modelos disponibles
    for m in genai.list_models():
        # Imprimir solo los modelos que contienen "gemini" en su nombre para filtrar
        if 'gemini' in m.name:
            print(f"\nNombre: {m.name}")
            print(f"Descripción: {m.description}")
            print(f"Versión: {m.version}")
            print(f"Métodos soportados: {m.supported_generation_methods}")
            # Verifica si soporta 'generateContent'
            if 'generateContent' in m.supported_generation_methods:
                 print("  --> Soporta generateContent (útil para generación de texto/chat)")
            else:
                 print("  --> NO soporta generateContent")


except Exception as e:
    print(f"Error al listar modelos: {e}")
    print("Asegúrate de que tu GOOGLE_API_KEY es correcta y tienes conexión a internet.")

print("\n--- Fin de la lista ---")
print("Busca modelos con 'gemini' en el nombre que soporten 'generateContent'.")
print("El modelo 'gemini-pro' o 'gemini-1.0-pro' son opciones comunes.")