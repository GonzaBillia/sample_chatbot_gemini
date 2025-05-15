# src/app/cli/chat_cli.py
"""
Cliente de línea de comandos para interactuar con el bot RAG.
"""
import os
import sys

from app.core.config import get_settings
from app.langchain.chains import get_qa_chain

def main():
    # 1. Carga de settings (incluye la .env automáticamente)
    settings = get_settings()

    # 2. Aseguramos que GOOGLE_API_KEY está en el env para Gemini
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key

    # 3. Inicializa la cadena RAG
    qa_chain = get_qa_chain()

    # 4. Mensaje de bienvenida (puedes parametrizarlo también desde settings si quieres)
    print(f"\n¡Hola! Soy tu asistente. Escribe 'salir' para terminar.\n")

    # 5. Bucle de interacción
    try:
        while True:
            query = input("Tú: ")
            if query.strip().lower() in ("salir", "exit", "quit"):
                print("Adiós!")
                break
            if not query.strip():
                print("Por favor, ingresa una pregunta.")
                continue

            print("Procesando…")
            try:
                # Pasamos directamente la query; la chain se encarga de context, embedding, prompt, etc.
                answer = qa_chain.run(query)
            except Exception as e:
                print(f"Error al procesar la pregunta: {e}")
                continue

            print(f"Chatbot: {answer}\n")

    except (KeyboardInterrupt, EOFError):
        print("\nSesión terminada.")
        sys.exit(0)

if __name__ == "__main__":
    main()
