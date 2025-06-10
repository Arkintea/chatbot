# app.py
import streamlit as st
import os
import time
import json
import logging
from typing import List, Dict, Any

# Import modular components
from config import (
    DEFAULT_CHAT_TITLE, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, OLLAMA_BASE_URL,
    TEMPERATURE, HISTORY_INDEX_DIR, SESSIONS_METADATA_FILE, logger
)
from utils import (
    load_sessions_metadata_util, save_sessions_metadata_util, load_chat_history_util,
    save_chat_history_util, generate_session_id, get_session_title,
    create_new_session_util, delete_session_util, save_faiss_store_util, clean_llm_output
)
from core import ChatbotCore
from langchain_core.documents import Document # Needed for adding to history vector store

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Intelligent Cobot Assistant", layout="centered", initial_sidebar_state="expanded")

# --- Session State Initialization ---
# Initialize chatbot_core to None. It will be created on first run or if not existing.
if "chatbot_core" not in st.session_state:
    st.session_state.chatbot_core = None
# Load all session metadata
if "sessions_metadata" not in st.session_state:
    st.session_state.sessions_metadata = load_sessions_metadata_util()
# Active session ID
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
# Current chat history for the active session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Current interaction mode (Conversational, Knowledge Base, Database Search)
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "Conversational 💬"
# Helper to track if the current session was loaded on startup
if "current_session_loaded" not in st.session_state:
    st.session_state.current_session_loaded = None

# --- Functions for UI interactions ---
def switch_session(session_id: str):
    """Switches the active chat session."""
    st.session_state.active_session_id = session_id
    st.session_state.chat_history = load_chat_history_util(st.session_state.active_session_id)
    # Always reset to conversational mode when switching sessions for a clean start
    st.session_state.current_mode = "Conversational 💬"

def delete_chat_session(session_id: str):
    """Deletes a chat session and handles active session redirection."""
    delete_session_util(session_id, st.session_state.sessions_metadata)
    if st.session_state.active_session_id == session_id:
        # If the deleted session was the active one, try to load the most recent one
        sessions_meta = load_sessions_metadata_util()
        if sessions_meta:
            most_recent_session_id = None
            latest_timestamp = 0
            for sid, meta in sessions_meta.items():
                if isinstance(meta, dict) and meta.get('last_active', 0) > latest_timestamp:
                    latest_timestamp = meta['last_active']
                    most_recent_session_id = sid
            
            if most_recent_session_id:
                switch_session(most_recent_session_id) # Use switch_session to update history and mode
                logger.info(f"Deleted active session, loaded most recent: {most_recent_session_id}")
            else:
                # If no other sessions exist, create a new one
                st.session_state.active_session_id = create_new_session_util(st.session_state.sessions_metadata)
                st.session_state.chat_history = [{"role": "assistant", "content": "Hi! I'm your AI Companion, how can I help you today? 💖"}]
                save_chat_history_util(st.session_state.active_session_id, st.session_state.chat_history, st.session_state.sessions_metadata)
                st.session_state.current_mode = "Conversational 💬"
                logger.info("Deleted last session, created a new one with initial greeting.")
        else:
            # If no sessions at all after deletion, create a fresh one
            st.session_state.active_session_id = create_new_session_util(st.session_state.sessions_metadata)
            st.session_state.chat_history = [{"role": "assistant", "content": "Hi! I'm your AI Companion, how can I help you today? 💖"}]
            save_chat_history_util(st.session_state.active_session_id, st.session_state.chat_history, st.session_state.sessions_metadata)
            st.session_state.current_mode = "Conversational 💬"
            logger.info("No sessions found after deletion, created a new one with initial greeting.")
    st.rerun() # Rerun Streamlit to reflect changes

def init_chatbot_core():
    """Initializes the ChatbotCore instance if it's not already initialized."""
    if "chatbot_core" not in st.session_state or st.session_state.chatbot_core is None:
        with st.spinner("Initializing Chatbot Core (LLM, Embeddings, DB connections)... This may take a moment."):
            try:
                st.session_state.chatbot_core = ChatbotCore(
                    embedding_model_name=EMBEDDING_MODEL_NAME,
                    llm_model_name=LLM_MODEL_NAME,
                    ollama_base_url=OLLAMA_BASE_URL,
                    temperature=TEMPERATURE
                )
                st.success("Chatbot Core initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize Chatbot Core: {e}. Please check your Ollama server and model configuration.")
                st.session_state.chatbot_core = None
                logger.exception("Failed to initialize ChatbotCore")

# --- Load most recent session on application startup if no session is active ---
if st.session_state.active_session_id is None:
    sessions_meta = load_sessions_metadata_util()
    if sessions_meta:
        most_recent_session_id = None
        latest_timestamp = 0
        for session_id, meta in sessions_meta.items():
            # Ensure meta is a dictionary before accessing its items
            if isinstance(meta, dict) and meta.get('last_active', 0) > latest_timestamp:
                latest_timestamp = meta['last_active']
                most_recent_session_id = session_id
        
        if most_recent_session_id:
            st.session_state.active_session_id = most_recent_session_id
            st.session_state.chat_history = load_chat_history_util(st.session_state.active_session_id)
            st.session_state.current_mode = "Conversational 💬" # Default mode on load
            st.session_state.current_session_loaded = most_recent_session_id
            logger.info(f"Loaded most recent session: {most_recent_session_id}")
        else:
            # If no valid recent session found (e.g., all metadata is empty/corrupt), create a new one
            st.session_state.active_session_id = create_new_session_util(st.session_state.sessions_metadata)
            st.session_state.chat_history = [{"role": "assistant", "content": "Hi! I'm your AI Companion, how can I help you today? 💖"}]
            save_chat_history_util(st.session_state.active_session_id, st.session_state.chat_history, st.session_state.sessions_metadata)
            st.session_state.current_mode = "Conversational 💬"
            st.session_state.current_session_loaded = st.session_state.active_session_id
            logger.warning("No valid recent session found, created a new one with initial greeting.")
    else:
        # No sessions exist at all, create the very first one
        st.session_state.active_session_id = create_new_session_util(st.session_state.sessions_metadata)
        st.session_state.chat_history = [{"role": "assistant", "content": "Hi! I'm your AI Companion, how can I help you today? 💖"}]
        save_chat_history_util(st.session_state.active_session_id, st.session_state.chat_history, st.session_state.sessions_metadata)
        st.session_state.current_mode = "Conversational 💬"
        st.session_state.current_session_loaded = st.session_state.active_session_id
        logger.info("No sessions found, created a new one with initial greeting.")
# --- End Load most recent session on startup ---

# Initialize the chatbot core after session state setup
init_chatbot_core()

# --- Streamlit Sidebar ---
with st.sidebar:
    try:
        st.image("wmg_logo.png", width=300)
    except Exception as e:
        st.error(f"Error loading logo: {e}")

    # "New Chat" button
    if st.button("➕ New Chat", key="new_chat_button", use_container_width=True, type="primary"):
        new_id = generate_session_id()
        st.session_state.active_session_id = new_id
        st.session_state.chat_history = [{"role": "assistant", "content": "Hi! I'm your AI Companion, how can I help you today? 💖"}]
        
        # Add new session to metadata
        st.session_state.sessions_metadata[new_id] = {
            'id': new_id,
            'name': DEFAULT_CHAT_TITLE,
            'created_at': time.time(),
            'last_active': time.time(),
            'chat_history_file': f"chat_history_{new_id}.json"
        }
        
        st.session_state.current_session_loaded = new_id
        save_chat_history_util(new_id, st.session_state.chat_history, st.session_state.sessions_metadata)
        logging.info(f"New chat started: {new_id}")
        st.rerun() # Rerun to reflect the new chat

    st.markdown("--- \n#### Chat History")
    # Ensure sessions_metadata is a dictionary
    if not isinstance(st.session_state.sessions_metadata, dict):
        st.session_state.sessions_metadata = {}

    # Sort sessions by last_active time
    try:
        sorted_sessions = sorted(st.session_state.sessions_metadata.items(), 
                                 key=lambda item: item[1].get('last_active', 0) if isinstance(item[1], dict) else 0, 
                                 reverse=True)
    except Exception:
        sorted_sessions = [] # Fallback if sorting fails

    # Display chat history entries
    history_container = st.container(height=200, border=False)
    with history_container:
        if not sorted_sessions:
            st.caption("No past chats yet.")
        for session_id, meta in sorted_sessions:
            if not isinstance(meta, dict): # Skip malformed metadata entries
                continue
            title = meta.get('name', session_id)
            display_title = (title[:20] + '...' if len(title) > 20 else title) if title else DEFAULT_CHAT_TITLE
            is_active = (session_id == st.session_state.active_session_id)
            
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                # Button to switch to a session
                if col1.button(f"**{display_title}**" if is_active else display_title, 
                               key=f"session_btn_{session_id}", 
                               use_container_width=True, 
                               help=title, 
                               disabled=is_active, 
                               on_click=lambda s_id=session_id: switch_session(s_id)):
                    pass # The on_click handles the action, button content itself doesn't need
            with col2:
                # Button to delete a session
                if col2.button("🗑️", 
                               key=f"del_btn_{session_id}", 
                               help=f"Delete chat: {title}", 
                               type="secondary", 
                               use_container_width=True, 
                               on_click=lambda s_id=session_id: delete_chat_session(s_id)):
                    pass
    st.divider()

    # Mode Selection Radio Buttons
    available_modes = ["Conversational 💬", "Knowledge Base 📚", "Database Search 🗃️"]
    st.markdown("### ⚡ Mode Selection ⚡")
    default_mode_val = "Conversational 💬"
    # Ensure current_mode is one of the available modes, reset if not
    if 'current_mode' not in st.session_state or st.session_state.current_mode not in available_modes:
        st.session_state.current_mode = default_mode_val
    
    try:
        current_mode_idx = available_modes.index(st.session_state.current_mode)
    except ValueError:
        current_mode_idx = 0
        st.session_state.current_mode = available_modes[0] # Fallback to first mode if not found
    
    selected_mode_radio = st.radio("Choose interaction mode:", available_modes, key='mode_radio', index=current_mode_idx)
    
    # If mode changes, update session state and rerun
    if selected_mode_radio != st.session_state.current_mode:
        st.session_state.current_mode = selected_mode_radio
        st.rerun()
    st.divider()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("💖 WMG SME Digital Team", unsafe_allow_html=True)

# --- Main Chat Interface ---
st.title("Intelligent Assistant")
st.caption("Your friendly AI assistant with a knowledge base and database query capabilities! 📚")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        avatar = "🤖" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# --- Main Logic Loop for Chat Input ---
user_input_data = st.chat_input(
    "Let's chat or ask me anything!!!",
)

if user_input_data:
    active_session_id = st.session_state.active_session_id
    user_query_text = user_input_data
    display_message = user_query_text

    # Add user message to chat history immediately for display
    st.session_state.chat_history.append({"role": "user", "content": display_message})
    
    with st.chat_message("user", avatar="👤"):
            st.markdown(display_message)
        
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""

        # --- Display Response with Streaming ---
        with st.spinner("🧠 Thinking..."):

            try:
                response_stream = st.session_state.chatbot_core.process_query(
                    user_query_text,
                    st.session_state.active_session_id,
                    st.session_state.chat_history,
                    st.session_state.current_mode
                )
                
                for chunk in response_stream:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                    final_display_response = clean_llm_output(full_response)
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                error_msg = f"An error occurred: {e}"
                logger.error(error_msg, exc_info=True)
                message_placeholder.markdown(error_msg)
                final_display_response = error_msg

        # Save the AI's response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": final_display_response})
        save_chat_history_util(st.session_state.active_session_id, st.session_state.chat_history, st.session_state.sessions_metadata)
        
        # Add conversation turn to history vector store for RAG
        if st.session_state.chatbot_core and \
           st.session_state.chatbot_core.history_vector_store and \
           st.session_state.chatbot_core.embedding_model:
            try:
                user_turn_text = user_query_text
                ai_turn_text = final_display_response
                turn_docs = [
                    Document(page_content=user_turn_text, metadata={"role": "user", "session_id": st.session_state.active_session_id, "timestamp": time.time()}), 
                    Document(page_content=ai_turn_text, metadata={"role": "assistant", "session_id": st.session_state.active_session_id, "timestamp": time.time()})
                ]
                st.session_state.chatbot_core.history_vector_store.add_texts([doc.page_content for doc in turn_docs], metadatas=[doc.metadata for doc in turn_docs])
                if not save_faiss_store_util(st.session_state.chatbot_core.history_vector_store, HISTORY_INDEX_DIR): 
                    logger.warning("Failed to save history FAISS store.")
            except Exception as e: 
                logger.error(f"Error saving chat turn to history vector store: {e}", exc_info=True)

        st.rerun() # Rerun to update the chat interface with the assistant's response

