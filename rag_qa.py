from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load the same HuggingFace embeddings used in ingestion
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS index
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM for generation (Ollama only)
llm = OllamaLLM(model="llama3")  # DO NOT put all-MiniLM-L6-v2 here

# Prompt template
prompt = ChatPromptTemplate.from_template(
"""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
)

# LCEL RAG chain
rag_chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

# Chat loop
while True:
    query = input("\nAsk a question (or type exit): ")
    if query.lower() == "exit":
        break

    answer = rag_chain.invoke(query)
    print("\nAnswer:\n", answer)
