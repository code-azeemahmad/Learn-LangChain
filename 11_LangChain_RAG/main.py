from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
        RunnableLambda,
        RunnablePassthrough,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader(
    "fastapi.txt",
    encoding="utf-8",
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    base_url="http://localhost:11434"
)

model = ChatOllama(
    model="gemma4:26b",
    temperature=0
)

parser = StrOutputParser()

documents = loader.load()
chunks = splitter.split_documents(documents)

vector_store = QdrantVectorStore.from_documents(
    chunks,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="langchain-rag-pipeline"
)

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3,
    }
)

def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(
            f"Source [{doc.metadata.get('source', 'unknown')}]:\n{doc.page_content}"
            for doc in docs
        )

question = "How does FastAPI handle dependency injection?"


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful technical assistant.
            Answer the user's question using only the
            provided context.

            If the answer is not contained in the context,
            say that you do not have enough information.
            
            Context:
            {context}
            """,
        ),
        (
            "human",
            "{question}",
        ),
    ]
)

format_context = RunnableLambda(format_docs)

rag_chain = (
    {
        "context": retriever | format_context,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | parser

)

answer = rag_chain.invoke(question)
print(answer)