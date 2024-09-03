import os
from typing import Optional
from dotenv import load_dotenv
from operator import itemgetter
import psycopg

from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_postgres import PGVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_postgres import PostgresChatMessageHistory

load_dotenv()


def get_rag_assistant(
    vector_store,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
) :
    """Get a Local RAG Assistant."""
    system_template = """When a user asks a question, you will be provided with information about the question.
    Carefully read this information and provide a clear and concise answer to the user.
    Do not use phrases like 'based on my knowledge' or 'depending on the information'.
    {context} """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_template),
            ("placeholder", "{chat_history}"),
            ("human","{input}"),        
        ]
    )
    retriever = get_retriever(vector_store)
    llm = ChatOllama(model="llama3")

    return get_rag_chain(retriever, prompt, llm)

def get_vector_store(embeddings_model: str, collection_name:str):
    connection = "postgresql+psycopg://ai:ai@localhost:5532/ai"
    if embeddings_model == "embed-english-v3.0":
        embedding=CohereEmbeddings(model=embeddings_model)
    else:
        embedding=OllamaEmbeddings(model=embeddings_model)
    return PGVector(
        embeddings=embedding,
        collection_name=collection_name,
        connection=connection
        )
    
def get_retriever(vector_store):
    base_retrieval = vector_store.as_retriever(search_kwargs={"k":5})
    rerank = CohereRerank()
    return ContextualCompressionRetriever(
        base_retriever=base_retrieval,
        base_compressor=rerank
        )        
def get_rag_chain(retriever, prompt,llm):
    retriever_chain = RunnableParallel({
        "context":itemgetter("input") | retriever,
         "input": itemgetter("input"),
         "chat_history": itemgetter("chat_history")}
        )
    format_context = RunnablePassthrough.assign(context = lambda x: format_docs(x["context"]))
    rag_chain = format_context | prompt | llm  | StrOutputParser()
    return retriever_chain.assign(answer=rag_chain)
  
def get_rag_history_assistant(chain):
    # lambda session_id:history_store,
    return RunnableWithMessageHistory(
        chain,
        get_message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )  
    
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

conn_dict = psycopg.conninfo.conninfo_to_dict("postgresql://ai:ai@localhost:5532/ai")
sync_connection = psycopg.connect(**conn_dict)
table_name = "chat_history"
def get_message_history(session_id:str):
    return PostgresChatMessageHistory(
        table_name,
        session_id,
        sync_connection=sync_connection
    )
def get_session_ids():
    with psycopg.connect(**conn_dict) as conn:
        with conn.cursor() as cur:
            cur.execute(f"select session_id, min(created_at) from {table_name} group by session_id order by min(created_at) desc;")
            rows = cur.fetchall()
            return [str(row[0]) for row in rows]