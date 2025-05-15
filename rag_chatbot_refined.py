import os
from dotenv import load_dotenv
from google import genai
from langchain_community.vectorstores import PGVector
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ——— Carga de entorno ———
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ——— Wrapper custom para embeddings ———
class GeminiEmbeddings:
    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for txt in texts:
            resp = genai.embeddings.embed_text(model=self.model, text=txt)
            # resp.embeddings puede ser un objeto ProtoType; convertimos a lista de floats
            raw = getattr(resp, "embeddings", None) or resp.get("embeddings")
            # si es ProtoType con atributo 'values'
            if hasattr(raw, "values"):
                raw = list(raw.values)
            vecs.append(raw)
        return vecs

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

# ——— Setup de DB y VectorStore ———
DB_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@" \
         f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
embeddings = GeminiEmbeddings(model="models/text-embedding-004")
vectorstore = PGVector(
    collection_name="knowledge_collection",
    connection_string=DB_URL,
    embedding_function=embeddings,
    use_jsonb=True
)

# ——— Retriever y LLM ———
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash-thinking-exp-1219",
    temperature=0.2,
    top_p=0.9,
    max_tokens=512
)

# ——— Prompts para chain map_reduce ———
map_prompt = PromptTemplate(
    template="""You are an expert technical assistant. Use the context to answer.

Context:
{context}

Question:
{question}

Answer:""",
    input_variables=["context","question"]
)
combine_prompt = PromptTemplate(
    template="""Merge the following partial answers into one concise, accurate response.
Cite sources in [brackets].

Question:
{question}

Partial Answers:
{summaries}

Final Answer:""",
    input_variables=["question","summaries"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "question_prompt": map_prompt,
        "combine_prompt": combine_prompt
    }
)

# ——— Función de consulta con reranking sencillo ———
def rerank_and_ask(query: str, top_n: int = 5):
    docs = retriever.invoke({"query": query})
    top_docs = docs[:top_n]
    return qa_chain.invoke({"query": query, "input_documents": top_docs})

# ——— Bucle principal ———
if __name__ == "__main__":
    print("Chatbot iniciado. (escribe 'salir' para terminar)")
    while True:
        q = input("Tú: ").strip()
        if q.lower() == "salir":
            break
        if not q:
            continue
        res = rerank_and_ask(q)
        print("\nChatbot:", res["result"])
        if docs := res.get("source_documents"):
            print("\nFuentes:")
            for d in docs:
                print("-", d.metadata.get("source", "N/A"))
