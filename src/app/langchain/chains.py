# src/app/langchain/chains.py

"""
Configuración de la cadena de RetrievalQA usando LangChain y Gemini,
con prompt externalizado en `src/app/prompts/base_prompt.txt`.
"""
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.core.config import get_settings
from app.embeddings.vectorstore import get_vectorstore
from app.langchain.prompt_manager import load_prompt

def get_qa_chain(
    top_k: int = 5,
    temperature: float = 0.0,
) -> RetrievalQA:
    """
    Devuelve un objeto RetrievalQA configurado con:
      - LLM Gemini para generación
      - PGVector retriever con top_k dinámico
      - PromptTemplate cargado desde base_prompt.txt
      - Temperatura del LLM dinámica
    """
    settings = get_settings()

    # 1) Embeddings & Vectorstore
    embeddings = GoogleGenerativeAIEmbeddings(model=settings.embedding_model)
    vectorstore = get_vectorstore(
        embedding_function=embeddings,
        collection_name=settings.collection_name
    )

    # 2) Retriever con top_k variable
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k}
    )

    # 3) LLM con temperatura variable
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=temperature
    )

    # 4) Carga y construcción del PromptTemplate
    template_str = load_prompt("base_prompt.txt")
    prompt = PromptTemplate(
        template=template_str,
        input_variables=["context", "question"],
    )

    # 5) Creación de la cadena RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain
