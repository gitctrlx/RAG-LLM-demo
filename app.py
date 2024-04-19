import os
import shutil
import gradio as gr
import json

from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
)
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core.response_synthesizers import ResponseMode

from chatglm import ChatGML

# Initialize directories
PERSIST_DIR = "./storage"
DATA_DIR = "./dataset"
os.makedirs(DATA_DIR, exist_ok=True)

# Configuration for Language Models
# Settings.llm = OpenAI(api_key="", model="gpt-3.5-turbo", temperature=0.1)
Settings.llm = ChatGML()
Settings.embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

# Setup for text processing
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text"
)
postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
rerank = SentenceTransformerRerank(top_n=2, model="BAAI/bge-reranker-base")

# Global variable for the query engine
query_engine = None

def initialize_index(directory):
    """ Initialize index from a given directory and handle errors properly. """
    persist_dir = os.path.join(PERSIST_DIR, os.path.basename(directory))
    os.makedirs(persist_dir, exist_ok=True)

    if not os.listdir(directory):
        print("No files found in the directory:", directory)
        return None

    try:
        documents = SimpleDirectoryReader(directory).load_data()
        if not documents:
            raise ValueError("No documents to index.")

        # storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = VectorStoreIndex.from_documents(documents)
        # storage_context.clear_persisted()
        index.storage_context.persist(persist_dir=persist_dir)

        return index.as_query_engine(
            # streaming=True, 
            alpha=0.5, 
            # node_postprocessors=[postproc], 
            similarity_top_k = 6,
            node_postprocessors = [rerank],
            # vector_store_query_mode="hybrid",
            response_synthesizer_mode=ResponseMode.REFINE)
    except Exception as e:
        print(f"[ERROR] Failed to initialize index: {e}")
        return None

def update_index(folder_name):
    """ Update the index with the new folder and handle missing folders. """
    new_data_dir = os.path.join(DATA_DIR, folder_name)
    global query_engine
    query_engine = initialize_index(new_data_dir)
    if query_engine is None:
        print("[ERROR] Initialization of the query engine failed. Please check the logs for more details.")

def create_folder(folder_name):
    """ Create a new folder for storing documents and initialize index. """
    if not folder_name:
        return "Folder name cannot be empty."
    new_data_dir = os.path.join(DATA_DIR, folder_name)
    new_persist_dir = os.path.join(PERSIST_DIR, folder_name)
    os.makedirs(new_data_dir, exist_ok=True)
    os.makedirs(new_persist_dir, exist_ok=True)
    return gr.Dropdown(choices=get_folders(), label="Select Folder", interactive=True)

def upload_files(files, folder_name):
    """ Handle file uploads and update index accordingly. """
    if not folder_name:
        return "Please select a folder first."
    target_dir = os.path.join(DATA_DIR, folder_name)
    if not os.path.exists(target_dir):
        return "Selected folder does not exist."

    file_paths = []
    duplicate_files = []
    for file_path in files:
        filename = os.path.basename(file_path)
        target_path = os.path.join(target_dir, filename)
        if not os.path.exists(target_path):
            shutil.copy(file_path, target_path)
            file_paths.append(target_path)
        else:
            duplicate_files.append(filename)
    update_index(folder_name)
    uploaded_files_msg = "\n[INFO] Files uploaded to: " + ", ".join(file_paths) if file_paths else "No new files uploaded."
    duplicate_files_msg = "\n[INFO] Duplicate files not uploaded: " + ", ".join(duplicate_files) if duplicate_files else ""
    return uploaded_files_msg + duplicate_files_msg

def folder_changed(folder_name):
    """ Function to be triggered when folder selection changes. """
    try:
        # Construct the storage context from the persist directory
        storage_context = StorageContext.from_defaults(persist_dir=os.path.join(PERSIST_DIR, folder_name))
        # Load the index from the existing storage
        index = load_index_from_storage(storage_context)
        global query_engine
        query_engine = index.as_query_engine(
            # streaming=True, 
            alpha=0.5, 
            # node_postprocessors=[postproc], 
            similarity_top_k = 6,
            node_postprocessors = [rerank],
            # vector_store_query_mode="hybrid",
            response_synthesizer_mode=ResponseMode.REFINE)
        if query_engine is None:
            message = "Failed to load index. Please check the storage or logs for more details."
        else:
            message = "Index loaded successfully from storage."
    except Exception as e:
        # Handle exceptions that could occur during index loading
        message = f"An error occurred while loading the index: {e}"
    return folder_name, message

def get_folders():
    """ Get list of folders in the data directory. """
    return [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]

def chatbot_knowledge_base(query, files=None):
    """ Handle chat queries and ensure the query engine is initialized. """
    if query_engine is None:
        return "Query engine is not available. Please ensure the knowledge base is properly initialized."
    response = query_engine.query(query + ", output in chinese")
    return str(response)

# Gradio interface setup
with gr.Blocks() as interface:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Manage Knowledge Base")
            folder_name_input = gr.Textbox(label="Create New Folder")
            folder_selection = gr.Dropdown(choices=get_folders(), label="Select Folder", interactive=True)
            message_output = gr.Text(label="Status Messages")
            folder_selection.change(folder_changed, inputs=folder_selection, outputs=[folder_selection, message_output])
            
            create_folder_button = gr.Button("Create Folder")
            create_folder_button.click(fn=create_folder, inputs=folder_name_input, outputs=folder_selection)

            file_input = gr.File(label="Select Files", file_count="multiple", file_types=['.pdf', '.docx', '.txt', ".csv", ".ppt", ".xlsx", ".doc", ".docx", ".md"], type="filepath")
            file_output = gr.Textbox(label="File Upload Status")
            upload_button = gr.Button("Upload Files")
            upload_button.click(fn=upload_files, inputs=[file_input, folder_selection], outputs=file_output)

        with gr.Column(scale=2):
            gr.Markdown("### Chat with AI")
            chat = gr.ChatInterface(fn=chatbot_knowledge_base, title="AI Chat", theme="default", fill_height=True)

tabbed_interface = gr.TabbedInterface([interface], ["知识库交流"])
tabbed_interface.launch(server_name="0.0.0.0", server_port=6006)
