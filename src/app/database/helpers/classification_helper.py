from langchain.chains import LLMChain
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

# Cadena LLM determinista para clasificación de complejidad
classification_chain = LLMChain(
    llm=ChatGoogleGenerativeAI(
        model=defaults.llm_model,
        temperature=0.0,
    ),
    prompt=complexity_prompt,
)
