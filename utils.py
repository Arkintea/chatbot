# utils.py
import os
import json
import time
import re
import uuid
import logging
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_community.docstore import InMemoryDocstore
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import (
    DATA_DIR, SESSIONS_METADATA_FILE, DEFAULT_CHAT_TITLE,
    BACKEND_RAG_INDEX_DIR, HISTORY_INDEX_DIR, BACKEND_RAG_SOURCE_DIR
)

logger = logging.getLogger(__name__)

def load_sessions_metadata_util():
    """Loads session metadata from a JSON file."""
    if os.path.exists(SESSIONS_METADATA_FILE):
        try:
            with open(SESSIONS_METADATA_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error("Error decoding sessions_metadata.json. Starting with empty metadata.")
            return {}
    return {}

def save_sessions_metadata_util(metadata):
    """Saves session metadata to a JSON file."""
    with open(SESSIONS_METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)

def load_chat_history_util(session_id: str):
    """Loads chat history for a given session ID."""
    history_file = os.path.join(DATA_DIR, f"chat_history_{session_id}.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f: 
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error decoding chat_history_{session_id}.json. Returning empty history.")
            return []
        except Exception as e:
            logger.error(f"Error reading chat_history_{session_id}.json: {e}", exc_info=True)
            return []
    return []

def save_chat_history_util(session_id: str, chat_history: List[Dict[str, str]], sessions_metadata: Dict[str, Any]):
    """Saves chat history for a given session and updates session metadata."""
    history_file = os.path.join(DATA_DIR, f"chat_history_{session_id}.json")
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=4)

    if session_id in sessions_metadata:
        sessions_metadata[session_id]['last_active'] = time.time()
        sessions_metadata[session_id]['name'] = get_session_title(chat_history)
        save_sessions_metadata_util(sessions_metadata)

def generate_session_id() -> str:
    """Generates a unique session ID."""
    return f"chat_{int(time.time())}_{uuid.uuid4().hex[:6]}"

def sanitize_text_for_title(text: str) -> str:
    """Sanitizes text to create a suitable session title."""
    text = re.sub(r'[^\w\s-]', '', text).strip()
    text = re.sub(r'[-\s]+', ' ', text)
    return text[:50]

def get_session_title(history: list) -> str:
    """Derives a session title from the first user message in the chat history."""
    if history:
        user_messages = [m['content'] for m in history if m['role'] == 'user']
        if user_messages:
            first_user_message_content = user_messages[0].strip()
            if first_user_message_content:
                title = sanitize_text_for_title(first_user_message_content)
                return title if title else DEFAULT_CHAT_TITLE
    return DEFAULT_CHAT_TITLE

def create_new_session_util(sessions_metadata: Dict[str, Any]):
    """Creates a new session and updates metadata."""
    new_session_id = generate_session_id()
    sessions_metadata[new_session_id] = {
        "id": new_session_id,
        "name": DEFAULT_CHAT_TITLE,
        "created_at": time.time(),
        "last_active": time.time(),
        "chat_history_file": f"chat_history_{new_session_id}.json"
    }
    save_sessions_metadata_util(sessions_metadata)
    return new_session_id

def delete_session_util(session_id: str, sessions_metadata: Dict[str, Any]):
    """Deletes a chat session and its history file."""
    history_file = os.path.join(DATA_DIR, f"chat_history_{session_id}.json")
    if os.path.exists(history_file):
        os.remove(history_file)
    if session_id in sessions_metadata:
        del sessions_metadata[session_id]
        save_sessions_metadata_util(sessions_metadata)

def load_faiss_store_util(index_dir: str, embeddings_model):
    """Loads a FAISS vector store from a local directory."""
    if os.path.exists(index_dir) and len(os.listdir(index_dir)) > 0:
        try:
            logger.info(f"Loading FAISS index from {index_dir}...")
            faiss_index = FAISS.load_local(index_dir, embeddings_model, allow_dangerous_deserialization=True)
            logger.info("FAISS index loaded successfully.")
            return faiss_index
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {index_dir}: {e}", exc_info=True)
            return None
    logger.info(f"No existing FAISS index found at {index_dir}.")
    return None

def save_faiss_store_util(faiss_index, index_dir: str):
    """Saves a FAISS vector store to a local directory."""
    try:
        if faiss_index:
            logger.info(f"Saving FAISS index to {index_dir}...")
            os.makedirs(index_dir, exist_ok=True)
            faiss_index.save_local(index_dir)
            logger.info("FAISS index saved successfully.")
            return True
    except Exception as e:
        logger.error(f"Failed to save FAISS index to {index_dir}: {e}", exc_info=True)
    return False

def build_backend_rag_index_util(source_dir: str, index_dir: str, embeddings_model, text_splitter):
    """Builds a FAISS RAG index from PDF documents in a source directory."""
    documents = []
    for filename in os.listdir(source_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(source_dir, filename)
            logger.info(f"Loading document: {filename}")
            try:
                loader = UnstructuredPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")

    if documents:
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Creating FAISS index with {len(chunks)} chunks...")
        new_index = FAISS.from_documents(chunks, embeddings_model)
        save_faiss_store_util(new_index, index_dir)
        return new_index
    logger.warning("No documents found to build backend RAG index.")
    return None

def clean_llm_output(text: str) -> str:
    """Removes specific XML-like tags from LLM output."""
    cleaned_text = re.sub(r"<tool_code>.*?</tool_code>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r"<execute_result>.*?</execute_result>", "", cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r"<(?:think)>.*?</(?:think)>", "", cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text.strip()