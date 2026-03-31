# RAG Service - Business logic for RAG operations

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import asyncio

logger = logging.getLogger(__name__)


class RagService:
    def __init__(self, uploads_dir: Path, storage_dir: Path):
        self.uploads_dir = uploads_dir
        self.storage_dir = storage_dir
        self.api_key: Optional[str] = None
    
    def set_api_key(self, api_key: str):
        """Set Gemini API key"""
        self.api_key = api_key
        logger.info("✅ API key configured")
    
    def has_index(self) -> bool:
        """Check if index exists in storage"""
        return any(self.storage_dir.iterdir())
    
    async def index_document(self, pdf_path: Path) -> int:
        """
        Index a PDF document for RAG.
        
        Steps:
        1. Configure Gemini embedding model
        2. Configure Gemini LLM
        3. Set both in LlamaIndex Settings
        4. Load PDF using SimpleDirectoryReader
        5. Split into chunks (chunk_size=512, overlap=50)
        6. Create FAISS index (dimension=768)
        7. Wrap with FaissVectorStore
        8. Build VectorStoreIndex
        9. Persist to ./storage
        10. Return number of chunks
        """
        # Check if API key is set
        if not self.api_key:
            raise ValueError(
                "API key not configured. Please set GEMINI_API_KEY environment variable."
            )
        
        from llama_index.core import Settings
        from llama_index.llms.gemini import Gemini
        from llama_index.embeddings.gemini import GeminiEmbedding
        from llama_index.core.readers import SimpleDirectoryReader
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.vector_stores.faiss import FaissVectorStore
        from llama_index.core import VectorStoreIndex, StorageContext
        import faiss
        
        # Import google.generativeai for model testing
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        
        logger.info("🔧 Configuring Gemini models...")
        logger.info(f"API Key preview: {self.api_key[:15]}...{self.api_key[-5:]}")
        
        # Configure Gemini embedding model
        embed_model = GeminiEmbedding(
            model_name="models/gemini-embedding-001",
            api_key=self.api_key
        )
        
        # Configure Gemini LLM using direct Google GenAI configuration
        # This bypasses LlamaIndex's automatic "models/" prefix addition
        logger.info("🔧 Configuring Gemini LLM...")
        
        # Initialize Google Generative AI directly
        genai.configure(api_key=self.api_key)  # type: ignore
        
        llm = None
        model_errors = []
        
        # List of models to try in order (without "models/" prefix)
        models_to_try = [
            "gemini-2.5-flash",      # Best for RAG: fast, accurate, latest
            "gemini-2.0-flash",       # Previous stable version
            "gemini-pro-latest",      # Legacy support
            "gemini-pro",             # Fallback
        ]
        
        for model_name in models_to_try:
            try:
                logger.info(f"Trying model: {model_name}")
                # Test if model is accessible
                model = genai.GenerativeModel(model_name=model_name)  # type: ignore
                # Try a simple generation to verify access
                model.generate_content("Hello")
                
                # If successful, create LlamaIndex wrapper
                llm = Gemini(
                    model_name=model_name,
                    api_key=self.api_key,
                )
                logger.info(f"✅ Successfully configured: {model_name}")
                break  # Success! Exit the loop
            except Exception as e:
                error_msg = f"{model_name} failed: {str(e)}"
                model_errors.append(error_msg)
                logger.warning(f"❌ {error_msg}")
                continue  # Try next model
        
        if not llm:
            logger.error("All Gemini models failed!")
            for err in model_errors:
                logger.error(f"  - {err}")
            raise ValueError(f"No compatible Gemini model found. Check your API key and permissions. Errors: {'; '.join(model_errors)}")
        
        logger.info(f"✅ Using model for indexing: {llm.model}")
        
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
            chunk_size=512,
            chunk_overlap=50
        )
        
        logger.info("✂️ Splitting document into chunks...")
        nodes = node_parser.get_nodes_from_documents(documents)
        
        logger.info(f"✅ Created {len(nodes)} chunks")
        
        # Create FAISS index
        logger.info("🔍 Creating FAISS vector store...")
        d = 768  # Embedding dimension for Gemini
        faiss_index = faiss.IndexFlatL2(d)
        
        # Wrap with FaissVectorStore
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        # Build StorageContext and VectorStoreIndex
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        # Persist to ./storage
        logger.info(f"💾 Persisting index to {self.storage_dir}...")
        index.storage_context.persist(persist_dir=str(self.storage_dir))
        
        logger.info("✅ Index created successfully!")
        
        return len(nodes)
    
    async def query_index(self, question: str) -> Tuple[str, List[str]]:
        """
        Query the indexed document.
        
        Steps:
        1. Configure Gemini models again
        2. Load FAISS index from ./storage
        3. Build query engine (top_k=3)
        4. Retrieve top chunks
        5. Generate answer using LLM
        6. Return answer and top 3 chunks as sources
        """
        # Check if API key is set
        if not self.api_key:
            raise ValueError(
                "API key not configured. Please set GEMINI_API_KEY environment variable."
            )
        
        from llama_index.core import Settings
        from llama_index.llms.gemini import Gemini
        from llama_index.embeddings.gemini import GeminiEmbedding
        from llama_index.vector_stores.faiss import FaissVectorStore
        from llama_index.core import StorageContext
        from llama_index.core.indices.loading import load_index_from_storage
        
        # Import google.generativeai for model testing
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        
        logger.info(f"❓ Processing query: '{question}'")
        
        # Configure Gemini models with same logic as index_document
        embed_model = GeminiEmbedding(
            model_name="models/gemini-embedding-001",
            api_key=self.api_key
        )
        
        logger.info("🔧 Configuring Gemini LLM for query...")
        genai.configure(api_key=self.api_key)  # type: ignore
        
        llm = None
        models_to_try = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-pro-latest",
            "gemini-pro",
        ]
        
        for model_name in models_to_try:
            try:
                logger.info(f"Trying model for query: {model_name}")
                model = genai.GenerativeModel(model_name=model_name)  # type: ignore
                model.generate_content("Hello")
                
                llm = Gemini(
                    model_name=model_name,
                    api_key=self.api_key,
                )
                logger.info(f"✅ Query model configured: {model_name}")
                break
            except Exception as e:
                logger.warning(f"❌ Query model {model_name} failed: {str(e)}")
                continue
        
        if not llm:
            raise ValueError("No compatible Gemini model found for queries. Check your API key and permissions.")
        
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
        
        # Retrieve top chunks
        response = query_engine.query(question)
        
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
