from langchain.schema.runnable import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import get_settings
from app.langchain.prompt_manager import load_prompt

# Cargar el prompt de clasificación desde un archivo .txt
defaults = get_settings()
prompt_str = load_prompt("classification_prompt.txt")
complexity_prompt = PromptTemplate(
    template=prompt_str,
    input_variables=["question"],
)

# Construir la secuencia Runnable (prompt | llm)
llm = ChatGoogleGenerativeAI(model=defaults.llm_model, temperature=0.0)
classification_chain: RunnableSequence = complexity_prompt | llm

# Para clasificar:
def classify_complexity(question: str) -> str:
    # Ejecuta la secuencia de forma sincrónica
    result = classification_chain.invoke({"question": question})
    # Si quieres manejarlo como LLMChain antiguo con 'run':
    # result = classification_chain.run(question=question)
    return result
