from langchain_core.tools import tool
from app.rag.store import get_vector_store


@tool
async def search_knowledge_base(query: str) -> str:
    """Searches the indexed RAG knowledge base for context matching the query."""
    store = get_vector_store()
    retriever = store.as_retriever(search_kwargs={"k": 4})
    docs = await retriever.ainvoke(query)
    if not docs:
        return "No relevant documents found in knowledge base."
    return "\n\n---\n\n".join(
        f"[Source Chunk {i+1}]: {doc.page_content}" for i, doc in enumerate(docs)
    )


RAG_TOOLS = [search_knowledge_base]
