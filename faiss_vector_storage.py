import faiss, os
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore

from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core.response_synthesizers import ResponseMode

# Setup for text processing
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text"
)
postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
rerank = SentenceTransformerRerank(top_n=2, model="maidalun1020/bce-reranker-base_v1")

class FaissEmbeddingStorage:

    def __init__(self, data_dir, persist_dir, dimension=768):
        self.d = dimension
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.index = self.initialize_index()

    def initialize_index(self):
        documents = SimpleDirectoryReader(self.data_dir).load_data()
        faiss_index = faiss.IndexFlatL2(self.d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        index.storage_context.persist(persist_dir = self.persist_dir)
        return index

    def get_query_engine(self):
        return self.index.as_query_engine(
            alpha=0.5, 
            similarity_top_k=6, 
            node_postprocessors=[rerank], 
            response_synthesizer_mode=ResponseMode.REFINE)

