from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_vectorstore(file_paths: list):
    all_docs = []

    logger.info(f"Building vectorstore from {len(file_paths)} files.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50)


    for path in file_paths:
        loader = TextLoader(path)
        documents = loader.load()

        chunks = splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata["source_file"] = path

        all_docs.extend(chunks)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")


    vectorstore = FAISS.from_documents(all_docs, embeddings)

    return vectorstore

def query_rag(vectorstore, query, chat_history, alpha=0.7):

    results = vectorstore.similarity_search_with_score(query, k=6)
    hybrid_results = []

    query_terms = set(query.lower().split())

    for doc, vector_score in results:
        text = doc.page_content.lower()
        keyword_matches = sum(1 for term in query_terms if term in text)

        # Normalize vector score (FAISS L2 distance → similarity)
        vector_sim = 1 / (1 + vector_score)

        # Normalize keyword score
        keyword_score = keyword_matches / max(len(query_terms), 1)

        # 🔥 Weighted hybrid blending
        hybrid_score = alpha * vector_sim + (1 - alpha) * keyword_score

        hybrid_results.append((doc, hybrid_score))

        hybrid_results.sort(key=lambda x: x[1], reverse=True)

        top_results = hybrid_results[:3]

        docs = [item[0] for item in top_results]
        scores = [item[1] for item in top_results]


        context = "\n\n".join([doc.page_content for doc in docs])
        llm = Ollama(model="mistral")

        logger.info("Running similarity search.")

        conversation_context = ""
    for turn in chat_history[-3:]:
        conversation_context += f"""
       Previous Question: {turn['question']}
       Previous Answer: {turn['answer']}"""


    prompt = f"""
       You are a document analysis assistant.

       Only answer using the provided context.
       If the answer cannot be found in the context, say:
       "Answer not found in provided documents."

        Context:
        {context}

         Conversation History:
        {conversation_context}

        Question:
        {query}"""


    response = llm.invoke(prompt)

    return response, docs, scores