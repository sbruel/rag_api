from typing import Optional, List, Tuple, Dict, Any
import asyncio
import contextvars
from langchain_core.documents import Document
from langchain_core.runnables.config import run_in_executor
from .extended_pg_vector import ExtendedPgVector

def context_aware_run_in_executor(executor, func, *args, **kwargs):
    """Run function in executor while preserving context variables"""
    # Capture current context
    ctx = contextvars.copy_context()
    
    # Create a wrapper that runs the function with the captured context
    def wrapper():
        return func(*args, **kwargs)
    
    # Run the wrapper in the captured context within the executor
    return run_in_executor(executor, ctx.run, wrapper)

class AsyncPgVector(ExtendedPgVector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_pool = None
    
    def _get_thread_pool(self):
        if self._thread_pool is None:
            try:
                # Try to get the thread pool from FastAPI app state
                import contextvars
                from fastapi import Request
                # This is a fallback - in practice, we'll pass the executor explicitly
                loop = asyncio.get_running_loop()
                self._thread_pool = getattr(loop, '_default_executor', None)
            except:
                pass
        return self._thread_pool
    
    async def get_all_ids(self, executor=None) -> list[str]:
        executor = executor or self._get_thread_pool()
        return await context_aware_run_in_executor(executor, super().get_all_ids)
    
    async def get_filtered_ids(self, ids: list[str], executor=None) -> list[str]:
        executor = executor or self._get_thread_pool()
        return await context_aware_run_in_executor(executor, super().get_filtered_ids, ids)

    async def get_documents_by_ids(self, ids: list[str], executor=None) -> list[Document]:
        executor = executor or self._get_thread_pool()
        return await context_aware_run_in_executor(executor, super().get_documents_by_ids, ids)

    async def delete(
        self, ids: Optional[list[str]] = None, collection_only: bool = False, executor=None
    ) -> None:
        executor = executor or self._get_thread_pool()
        await context_aware_run_in_executor(executor, self._delete_multiple, ids, collection_only)
    
    async def asimilarity_search_with_score_by_vector(
        self, 
        embedding: List[float], 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None,
        executor=None
    ) -> List[Tuple[Document, float]]:
        """Async version of similarity_search_with_score_by_vector"""
        executor = executor or self._get_thread_pool()
        return await context_aware_run_in_executor(
            executor, 
            super().similarity_search_with_score_by_vector, 
            embedding, 
            k, 
            filter
        )
    
    async def aadd_documents(
        self, 
        documents: List[Document], 
        ids: Optional[List[str]] = None,
        executor=None,
        **kwargs
    ) -> List[str]:
        """Async version of add_documents"""
        executor = executor or self._get_thread_pool()
        return await context_aware_run_in_executor(
            executor, 
            super().add_documents, 
            documents, 
            ids=ids,
            **kwargs
        )