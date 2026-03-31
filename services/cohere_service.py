# Cohere Service - Business logic for RAG operations using Cohere models

import logging
import os
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class CohereService:
    """RAG Service using Cohere models from Hugging Face"""
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 100
    MAX_INDEX_NODES = 350
    EMBED_BATCH_SIZE = 16
    RATE_LIMIT_RETRIES = 3
    
    def __init__(self, uploads_dir: Path, storage_dir: Path):
        self.uploads_dir = uploads_dir
        self.storage_dir = storage_dir
        self.api_key: Optional[str] = None
    
    def set_api_key(self, api_key: str):
        """Set Cohere API key"""
        self.api_key = api_key
        logger.info("✅ Cohere API key configured")

    def _ensure_api_key(self) -> str:
        """
        Return an API key from runtime config or environment.
        Keeps the service usable when keys are provided after startup.
        """
        api_key = self.api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not configured. Please set COHERE_API_KEY environment variable."
            )
        self.api_key = api_key
        return api_key
    
    def has_index(self) -> bool:
        """Check if index exists in storage"""
        return any(self.storage_dir.iterdir())
    
    async def index_document(self, pdf_path: Path) -> int:
        """
        Index a PDF document for RAG using Cohere models.
        
        Steps:
        1. Configure Cohere embedding model (embed-english-v3.0)
        2. Configure Cohere LLM (command-r-plus or command-r)
        3. Set both in LlamaIndex Settings
        4. Load PDF using SimpleDirectoryReader
        5. Split into chunks (chunk_size=512, overlap=50)
        6. Create FAISS index (dimension=1024 for Cohere)
        7. Wrap with FaissVectorStore
        8. Build VectorStoreIndex
        9. Persist to ./storage
        10. Return number of chunks
        """
        api_key = self._ensure_api_key()
        
        from llama_index.core import Settings
        from llama_index.llms.cohere import Cohere
        from llama_index.embeddings.cohere import CohereEmbedding
        from llama_index.core.readers import SimpleDirectoryReader
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.vector_stores.faiss import FaissVectorStore
        from llama_index.core import VectorStoreIndex, StorageContext
        import faiss
        
        logger.info("🔧 Configuring Cohere models...")
        logger.info(f"API Key preview: {api_key[:10]}...{api_key[-5:]}")
        
        # Configure Cohere embedding model
        embed_model = CohereEmbedding(
            model_name="embed-english-v3.0",
            api_key=api_key,
            embed_batch_size=self.EMBED_BATCH_SIZE,
        )
        
        logger.info("✅ Embedding model configured: embed-english-v3.0")
        
        # Configure Cohere LLM - Try multiple models for compatibility
        llm = None
        model_errors = []
        
        # List of Cohere models to try in order.
        # Aliases like `command-r-plus` and `command-r` may be retired.
        models_to_try = [
            "command-a-03-2025",
            "command-r-plus-08-2024",
            "command-r-08-2024",
            "command",
        ]
        
        for model_name in models_to_try:
            try:
                logger.info(f"Trying LLM model: {model_name}")
                llm = Cohere(
                    model=model_name,
                    api_key=api_key
                )
                logger.info(f"✅ Successfully configured LLM: {model_name}")
                break  # Success! Exit the loop
            except Exception as e:
                error_msg = f"{model_name} failed: {str(e)}"
                model_errors.append(error_msg)
                logger.warning(f"❌ {error_msg}")
                continue  # Try next model
        
        if not llm:
            logger.error("All Cohere LLM models failed!")
            for err in model_errors:
                logger.error(f"  - {err}")
            raise ValueError(f"No compatible Cohere model found. Check your API key and permissions. Errors: {'; '.join(model_errors)}")
        
        # Set models in LlamaIndex Settings
        Settings.embed_model = embed_model
        Settings.llm = llm
        
        logger.info(f"📄 Loading PDF document from {pdf_path}...")
        
        # Load PDF using SimpleDirectoryReader
        pdf_dir = pdf_path.parent
        documents = SimpleDirectoryReader(input_dir=str(pdf_dir)).load_data()
        
        logger.info(f"📊 Document loaded: {len(documents)} pages")
        
        # Split into chunks
        node_parser = SentenceSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP
        )
        
        logger.info("✂️ Splitting document into chunks...")
        nodes = node_parser.get_nodes_from_documents(documents)
        original_node_count = len(nodes)
        if original_node_count > self.MAX_INDEX_NODES:
            nodes = nodes[:self.MAX_INDEX_NODES]
            logger.warning(
                f"⚠️ Chunk count reduced from {original_node_count} to {len(nodes)} "
                f"to avoid embedding rate limits."
            )
        else:
            logger.info(f"✅ Created {len(nodes)} chunks")
        
        # Create FAISS index
        logger.info("🔍 Creating FAISS vector store...")
        d = 1024  # Embedding dimension for Cohere embed-english-v3.0
        faiss_index = faiss.IndexFlatL2(d)
        
        # Wrap with FaissVectorStore
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        # Build StorageContext and VectorStoreIndex with retry on API rate limits.
        index = None
        for attempt in range(self.RATE_LIMIT_RETRIES):
            try:
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex(
                    nodes,
                    storage_context=storage_context,
                    embed_model=embed_model
                )
                break
            except Exception as e:
                err = str(e)
                is_rate_limited = "429" in err or "Too Many Requests" in err
                if is_rate_limited and attempt < self.RATE_LIMIT_RETRIES - 1:
                    wait_s = 3 * (2 ** attempt)
                    logger.warning(
                        f"⚠️ Cohere rate limit during indexing. Retrying in {wait_s}s "
                        f"(attempt {attempt + 1}/{self.RATE_LIMIT_RETRIES})..."
                    )
                    await asyncio.sleep(wait_s)
                    continue
                raise
        if index is None:
            raise ValueError("Failed to build index due to repeated embedding failures.")
        
        # Persist to ./storage
        logger.info(f"💾 Persisting index to {self.storage_dir}...")
        index.storage_context.persist(persist_dir=str(self.storage_dir))
        
        logger.info("✅ Index created successfully!")
        
        return len(nodes)
    
    async def query_index(self, question: str) -> Tuple[str, List[str]]:
        """
        Query the indexed document using Cohere models.
        
        Steps:
        1. Configure Cohere models again
        2. Load FAISS index from ./storage
        3. Build query engine (top_k=3)
        4. Retrieve top chunks
        5. Generate answer using LLM
        6. Return answer and top 3 chunks as sources
        """
        api_key = self._ensure_api_key()
        
        from llama_index.core import Settings
        from llama_index.llms.cohere import Cohere
        from llama_index.embeddings.cohere import CohereEmbedding
        from llama_index.vector_stores.faiss import FaissVectorStore
        from llama_index.core import StorageContext
        from llama_index.core.indices.loading import load_index_from_storage
        
        logger.info(f"❓ Processing query: '{question}'")
        
        # Configure Cohere models
        embed_model = CohereEmbedding(
            model_name="embed-english-v3.0",
            api_key=api_key,
            embed_batch_size=self.EMBED_BATCH_SIZE,
        )
        
        # Use same model selection logic as index_document
        llm = None
        models_to_try = [
            "command-a-03-2025",
            "command-r-plus-08-2024",
            "command-r-08-2024",
            "command",
        ]
        
        for model_name in models_to_try:
            try:
                logger.info(f"Trying LLM for query: {model_name}")
                llm = Cohere(
                    model=model_name,
                    api_key=api_key
                )
                logger.info(f"✅ Query LLM configured: {model_name}")
                break
            except Exception as e:
                logger.warning(f"❌ Query LLM {model_name} failed: {str(e)}")
                continue
        
        if not llm:
            raise ValueError("No compatible Cohere LLM found for queries. Check your API key and permissions.")
        
        Settings.embed_model = embed_model
        Settings.llm = llm
        
        # Load persisted index from storage
        logger.info("📂 Loading index from storage...")

        vector_store = FaissVectorStore.from_persist_dir(
            persist_dir=str(self.storage_dir)
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(self.storage_dir)
        )
        
        index = load_index_from_storage(
            storage_context,
            embed_model=embed_model
        )
        
        # Build query engine (top_k=3)
        query_engine = index.as_query_engine(similarity_top_k=3)
        
        logger.info("🔍 Retrieving relevant chunks...")
        
        # Retrieve top chunks (retry once if query embedding hits temporary rate limit)
        response = None
        for attempt in range(self.RATE_LIMIT_RETRIES):
            try:
                response = query_engine.query(question)
                break
            except Exception as e:
                err = str(e)
                is_rate_limited = "429" in err or "Too Many Requests" in err
                if is_rate_limited and attempt < self.RATE_LIMIT_RETRIES - 1:
                    wait_s = 2 * (2 ** attempt)
                    logger.warning(
                        f"⚠️ Cohere rate limit during query. Retrying in {wait_s}s "
                        f"(attempt {attempt + 1}/{self.RATE_LIMIT_RETRIES})..."
                    )
                    await asyncio.sleep(wait_s)
                    continue
                raise
        if response is None:
            raise ValueError("Query failed due to repeated API rate limits.")
        
        # Get source chunks
        sources = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes[:3]:
                chunk_text = node.node.get_content()
                sources.append(chunk_text)
        
        answer = str(response)
        
        logger.info("✅ Response generated")
        logger.info(f"📝 Retrieved {len(sources)} source chunks")
        
        return answer, sources
