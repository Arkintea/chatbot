# core.py
import json
import logging
import re
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from config import (
    EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, OLLAMA_BASE_URL, TEMPERATURE,
    POSTGRES_CONFIGURED, INFLUXDB_CONFIGURED, BACKEND_RAG_INDEX_DIR, HISTORY_INDEX_DIR,
    BACKEND_RAG_SOURCE_DIR
)
from connectors import PostgresConnector, InfluxDBConnector
from utils import load_faiss_store_util, save_faiss_store_util, build_backend_rag_index_util, clean_llm_output

logger = logging.getLogger(__name__)

class ChatbotCore:
    """
    Core logic for the chatbot, managing LLM interactions, RAG, and database integrations.
    """
    def __init__(self, embedding_model_name: str, llm_model_name: str, ollama_base_url: str, temperature: float):
        self.embedding_model = None
        self.language_model = None
        self.backend_rag_vector_store = None
        self.history_vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.postgres_connector = None
        self.influxdb_connector = None

        self._initialize_models(embedding_model_name, llm_model_name, ollama_base_url, temperature)
        self._initialize_connectors()
        self._initialize_vector_stores()

    def _initialize_models(self, embedding_model_name: str, llm_model_name: str, ollama_base_url: str, temperature: float):
        """Initializes the embedding model and the Large Language Model."""
        try:
            logger.info(f"Initializing embedding model: {embedding_model_name} from {ollama_base_url}")
            self.embedding_model = OllamaEmbeddings(model=embedding_model_name, base_url=ollama_base_url)
            self.embedding_model.embed_query("test query") # Test embedding model connection
            logger.info("Embedding model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
            self.embedding_model = None

        try:
            logger.info(f"Initializing LLM: {llm_model_name} from {ollama_base_url}")
            self.language_model = OllamaLLM(
                model=llm_model_name,
                base_url=ollama_base_url,
                temperature=temperature,
            )
            self.language_model.invoke("Hello") # Test LLM connection
            logger.info("LLM initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
            self.language_model = None
    
    def _initialize_connectors(self):
        """Initializes database connectors based on configuration."""
        if POSTGRES_CONFIGURED:
            self.postgres_connector = PostgresConnector()
            if not self.postgres_connector.is_connected():
                logger.warning("PostgresConnector failed to connect or is not configured correctly.")
                self.postgres_connector = None
            else:
                logger.info("PostgresConnector initialized and connected.")
        else:
            logger.info("PostgreSQL is not configured, skipping PostgresConnector initialization.")

        if INFLUXDB_CONFIGURED:
            self.influxdb_connector = InfluxDBConnector()
            if not self.influxdb_connector.is_connected():
                logger.warning("InfluxDBConnector failed to connect or is not configured correctly.")
                self.influxdb_connector = None
            else:
                logger.info("InfluxDBConnector initialized and connected.")
        else:
            logger.info("InfluxDB is not configured, skipping InfluxDBConnector initialization.")

    def _initialize_vector_stores(self):
        """Initializes and loads/builds FAISS vector stores for RAG and chat history."""
        if self.embedding_model:
            # Backend RAG store
            self.backend_rag_vector_store = load_faiss_store_util(BACKEND_RAG_INDEX_DIR, self.embedding_model)
            if not self.backend_rag_vector_store:
                self.backend_rag_vector_store = build_backend_rag_index_util(BACKEND_RAG_SOURCE_DIR, BACKEND_RAG_INDEX_DIR, self.embedding_model, self.text_splitter)
            
            # Chat history store
            self.history_vector_store = load_faiss_store_util(HISTORY_INDEX_DIR, self.embedding_model)
            if not self.history_vector_store:
                logger.info("Creating empty FAISS index for chat history.")
                # Initialize with a dummy text if no history exists to avoid errors
                self.history_vector_store = FAISS.from_texts(["initialization"], self.embedding_model)
                save_faiss_store_util(self.history_vector_store, HISTORY_INDEX_DIR)
        else:
            logger.error("Embedding model not initialized. RAG and history features will be disabled.")

    def _get_all_available_tools_description(self) -> str:
        """Aggregates descriptions of all available database tools."""
        all_tools_descriptions = []
        if self.postgres_connector:
            all_tools_descriptions.extend(self.postgres_connector.get_tool_descriptions())
        if self.influxdb_connector:
            all_tools_descriptions.extend(self.influxdb_connector.get_tool_descriptions())
        
        formatted_tools = []
        for tool in all_tools_descriptions:
            tool_info = f"  - Name: {tool['name']}\n    Description: {tool['description']}"
            if 'parameters' in tool and tool['parameters']['properties']:
                params = ", ".join([f"{k} ({v['type']})" for k, v in tool['parameters']['properties'].items()])
                tool_info += f"\n    Parameters: {params}"
                if 'required' in tool['parameters'] and tool['parameters']['required']:
                    tool_info += f" (Required: {', '.join(tool['parameters']['required'])})"
            formatted_tools.append(tool_info)
        
        if formatted_tools:
            return "Available Database Tools:\n" + "\n".join(formatted_tools)
        return "No database tools are currently available."

    def _get_tool_function_by_name(self, tool_name: str):
        """Retrieves a specific tool function by its name."""
        if self.postgres_connector:
            func = self.postgres_connector.get_tool_function(tool_name)
            if func: return func
        if self.influxdb_connector:
            func = self.influxdb_connector.get_tool_function(tool_name)
            if func: return func
        return None

    def generate_answer(
        self,
        user_query: str,
        context_documents: Optional[str] = None,
        chat_history_str: Optional[str] = None,
        db_results: Optional[str] = None,
        current_mode: str = "Chat 💬"
    ) -> Generator[str, None, None]:
        """
        Generates an answer from the LLM based on the current mode and provided context.
        Yields chunks of the response for streaming.
        """
        if not self.language_model:
            yield "LLM is not initialized. Cannot generate answer."
            return

        system_message = ""
        user_message_content = user_query

        if current_mode == "Conversational 💬":
            system_message = (
                "You are a helpful and knowledgeable assistant."
            )
            if context_documents:
                system_message += f"\n\nContext:\n{context_documents}"
            if chat_history_str:
                system_message += f"\n\nChat History:\n{chat_history_str}"

        elif current_mode == "Knowledge Base 📚":
            system_message = (
                f"You are an expert assistant for retrieving information from the documents saved in the {BACKEND_RAG_SOURCE_DIR} folder in this chatbot_app. "
                f"Answer the user's questions based on the provided documents in the {BACKEND_RAG_SOURCE_DIR} folder. "
                f"The {BACKEND_RAG_SOURCE_DIR} folder is also refered to as the knowledge base or backend knowledge base. "
                "If the information is not found in the documents, tell the user the information is not found in the documents, but use your own general knowledge to the best of your ability to answer the question asked. "
                "Do not explicitly mention if information was found in the documents or not, just integrate it naturally if relevant."
                "After returning the response to the user, ask the user if the response you provided was helpful, and if not, ask them to provide more details or clarify their question."
            )
            if context_documents:
                system_message += f"\n\nBackend Knowledge Base Documents:\n{context_documents}"
            if chat_history_str:
                system_message += f"\n\nChat History:\n{chat_history_str}"

        elif current_mode == "Database Search 🗃️":
            tool_descriptions = self._get_all_available_tools_description()
            
            postgres_schema_desc_str = ""
            if self.postgres_connector and self.postgres_connector.is_connected():
                postgres_schema_raw = self.postgres_connector.get_schema_description()
                if postgres_schema_raw and not postgres_schema_raw.startswith("Error") and \
                   postgres_schema_raw != "No schema information could be retrieved for the specified tables of interest. Ensure tables exist and are accessible." and \
                   postgres_schema_raw != "PostgreSQL database not connected, schema unavailable.":
                    postgres_schema_desc_str = f"\n\nPostgreSQL Schema (Use table names like 'public.mrp_workorder' or 'cobot_data.rtde_logs' for queries):\n{postgres_schema_raw.replace('{', '{{').replace('}', '}}')}"
                else:
                    logger.warning(f"PostgreSQL schema description was empty or an error: '{postgres_schema_raw}'")
                    postgres_schema_desc_str = "\n\nPostgreSQL Schema: Currently unavailable or contains no relevant tables of interest."


            influxdb_schema_desc_str = ""
            if self.influxdb_connector and self.influxdb_connector.is_connected():
                influxdb_schema_raw = self.influxdb_connector.get_schema_description()
                if influxdb_schema_raw and not influxdb_schema_raw.startswith("Error") and \
                   influxdb_schema_raw != "InfluxDB database not connected, schema unavailable.":
                    influxdb_schema_desc_str = f"\n\nInfluxDB Schema (Bucket: '{self.influxdb_connector.config.get('bucket', 'Expo')}', Measurement for cobot data is typically 'cobot_telemetry'):\n{influxdb_schema_raw.replace('{', '{{').replace('}', '}}')}"
                else:
                    logger.warning(f"InfluxDB schema description was an error or unavailable: '{influxdb_schema_raw}'")
                    influxdb_schema_desc_str = "\n\nInfluxDB Schema: Currently unavailable or contains no relevant measurements."


            system_message = (
                "You are an expert database assistant for manufacturing ERP data (in PostgreSQL) and cobot telemetry (real-time/recent in InfluxDB, historical in PostgreSQL table `cobot_data.rtde_logs`). "
                "Your primary goal is to answer user questions by interacting with these databases using ONLY the provided tools and schemas. Adhere strictly to the following:\n"
                "1.  **Analyze Provided Schemas**: Your first step is to CAREFULLY review the PostgreSQL and InfluxDB schemas detailed below. All your database actions MUST be based EXCLUSIVELY on this information. DO NOT assume or hallucinate any other tables, measurements, fields, columns, or functions.\n"
                "    - **PostgreSQL**: Contains Odoo ERP tables (e.g., `public.mrp_workorder`, `public.crm_lead`, `public.res_partner` for Customers) and a crucial table `cobot_data.rtde_logs` for HISTORICAL cobot sensor data (joint positions, forces, temperatures, robot/safety modes over time).\n"
                "    - **InfluxDB**: Contains the `cobot_telemetry` measurement for RECENT or REAL-TIME cobot sensor data (joint positions, forces, temperatures, robot/safety modes). The bucket is 'Expo'.\n"
                
                "2.  **Prioritize Specialized Functions (Tools) & Map Common Terms**:\n"
                "    - For LATEST/CURRENT cobot status (e.g., 'latest temperature', 'current robot mode'), you MUST use the `get_latest_telemetry` function (InfluxDB).\n"
                "    - For common Odoo ERP requests, you MUST prioritize these specialized functions if they match the user's intent:\n"
                "        - User asks about 'newest customers', 'recent customers', 'list customers': Use the `get_recent_customers` function. (This queries the `public.res_partner` table which stores customer data).\n"
                "        - User asks about 'recent sales opportunities', 'latest leads', 'sales pipeline': Use the `get_recent_opportunities_with_orders` function. (This queries `public.crm_lead` and related tables).\n"
                "        - User asks about 'recent work orders', 'manufacturing tasks': Use the `get_recent_work_orders` function. (This queries `public.mrp_workorder`).\n"
                "    - If a specialized function perfectly fits, use it instead of trying to generate a raw SQL query from scratch for these common requests.\n"

                "3.  **Schema Inquiry Protocol**: If the user asks 'what tables exist?', 'describe measurements', 'what is the schema for customers?', or any similar question about database structure, you MUST call the `get_postgres_schema` function for PostgreSQL details or the `get_influxdb_schema` function for InfluxDB details. Provide the output of these functions. Do not invent schema; always use these functions for schema questions.\n"
                
                "4.  **Formulating Raw Queries (When Specialized Functions Don't Apply or for Other Tables)**:\n"
                "    - When formulating SQL queries for Odoo data not covered by a specialized function, or if the user asks for data from a table not covered by a specific tool, remember these mappings if the user uses common terms (always verify against the provided schema description below which lists full table names and their purpose):\n"
                "        - 'Customers', 'Partners', or 'Contacts' usually refers to the `public.res_partner` table.\n"
                "        - 'Production Orders' or 'Manufacturing Orders' usually refers to the `public.mrp_production` table.\n"
                "        - 'Products' can refer to `public.product_template` (general product definitions) or `public.product_product` (specific product variants).\n"
                "        - 'Work Orders' refers to `public.mrp_workorder`.\n"
                "        - 'Leads' or 'Opportunities' refers to `public.crm_lead`.\n"
                "    - Always refer to the full provided PostgreSQL schema for exact table names (like `public.res_partner`) and column names before constructing any query.\n"
                "    - For HISTORICAL cobot sensor data (e.g., temperature trend last hour, average force yesterday), generate a SQL SELECT query for the PostgreSQL table `cobot_data.rtde_logs`. Use the column definitions provided in its schema.\n"
                "    - For specific time-series analysis from InfluxDB not covered by `get_latest_telemetry` (e.g., average `temp3` over the last 30 minutes, or querying specific fields like `joint_temperatures_5`), generate a Flux query targeting the `cobot_telemetry` measurement in the 'Expo' bucket. Use field names like `temp1`-`temp6`, `joint_positions_0`-`joint_positions_5`, `tcp_forces_0`-`tcp_forces_5`, `robot_mode`, `safety_mode` as indicated by the data structure and schema.\n"

                "5.  **Non-Database Questions**: If the user's request is clearly unrelated to querying these specific databases, respond as a general helpful assistant without attempting database actions or outputting JSON.\n"
                f"\n\nAvailable Database Tools:\n{tool_descriptions.replace('{', '{{').replace('}', '}}') if tool_descriptions else 'No tools available.'}"
                f"{postgres_schema_desc_str}"
                f"{influxdb_schema_desc_str}"
                "\n\n"
                "**Response Format for Database Actions (MANDATORY - Adhere Strictly! Output ONLY the JSON block when performing an action):**\n"
                "To call a specific function: \n"
                "```json\n"
                "{{\n"
                "  \"action\": \"call_function\",\n"
                "  \"function_name\": \"<function_name_here>\",\n"
                "  \"arguments\": {{ <\"param1\": \"value1\", ...> }} \n"
                "}}\n"
                "```\n"
                "To execute a raw query: \n"
                "```json\n"
                "{{\n"
                "  \"action\": \"execute_query\",\n"
                "  \"query_type\": \"<postgres_sql_or_influxdb_flux>\",\n"
                "  \"query\": \"<your_SQL_SELECT_or_Flux_query_here>\"\n"
                "}}\n"
                "```\n"
                "For functions requiring no arguments (like schema functions), use `\"arguments\": {{}}`."
                "After returning the response to the user, ask the user if the response you provided was helpful, and if not, ask them to provide more details or clarify their question."
            )
            
            if db_results: 
                system_message = ( 
                    "You are an expert database assistant. You have just received the following results from a database query or function call. "
                    "Based SOLELY on these results, provide a concise and natural language answer to the user's original question. Do not refer to how you got the data (e.g., 'I ran a query'). Just present the information."
                    f"\n\n**Database Query/Function Results:**\n{db_results.replace('{', '{{').replace('}', '}}')}\n\n"
                    "Now, answer the user's original question using only this information."
                )
            
            if chat_history_str and not db_results: 
                system_message += f"\n\nRelevant Chat History:\n{chat_history_str.replace('{', '{{').replace('}', '}}')}"
            
        else:
            system_message = (
                "You are a helpful and knowledgeable assistant. "
                "Answer the user's questions based on the provided context and chat history. "
                "If the information is not available, use your own general knowledge to the best of your ability to answer the question asked."
                "After returning the response to the user, ask the user if the response you provided was helpful, and if not, ask them to provide more details or clarify their question."

            )
            if context_documents:
                system_message += f"\n\nContext:\n{context_documents}"
            if chat_history_str:
                system_message += f"\n\nChat History:\n{chat_history_str}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", user_message_content)
        ])

        try:
            chain = prompt | self.language_model
            for chunk in chain.stream({"user_query": user_query}):
                if hasattr(chunk, 'content') and isinstance(chunk.content, str):
                    yield chunk.content
                elif isinstance(chunk, str):
                    yield chunk
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}", exc_info=True)
            yield f"I'm sorry, I'm having trouble processing that request right now. Details: {e}"
            return

    def process_query(self, user_query: str, active_session_id: str, chat_history: List[Dict[str, str]], current_mode: str) -> Generator[str, None, None]:
        """
        Processes a user query based on the selected mode, incorporating RAG, history, and database tools.
        Yields chunks of the response.
        """
        if not self.language_model:
            yield "LLM is not initialized. Please check model configuration."
            return
        if not self.embedding_model:
            yield "Embedding model not initialized. RAG features will be limited."

        retrieved_context = []
        retrieved_history = []
        
        # Retrieve context documents for Knowledge Base mode
        if current_mode == "Knowledge Base 📚" and self.backend_rag_vector_store:
            try:
                retrieved_context_docs = self.backend_rag_vector_store.similarity_search(user_query, k=3)
                retrieved_context = [doc.page_content for doc in retrieved_context_docs]
                logger.info(f"Retrieved {len(retrieved_context)} context documents for Backend RAG.")
            except Exception as e:
                logger.error(f"Error retrieving from backend RAG: {e}", exc_info=True)
                retrieved_context = ["Error retrieving backend RAG documents."]

        # Retrieve relevant chat history
        if self.history_vector_store:
            try:
                retrieved_history_docs = self.history_vector_store.similarity_search(user_query, k=5)
                # Filter history to only include turns from the active session
                filtered_history_docs = [
                    doc for doc in retrieved_history_docs 
                    if doc.metadata.get('session_id') == active_session_id
                ]
                retrieved_history = [doc.page_content for doc in filtered_history_docs]
                logger.info(f"Retrieved {len(retrieved_history)} relevant chat history turns.")
            except Exception as e:
                logger.warning(f"Error retrieving from history vector store: {e}")
                retrieved_history = []

        retrieved_context_str = "\n".join(retrieved_context) if retrieved_context else None
        formatted_chat_history_str = "\n".join([f"{t['role']}: {t['content']}" for t in chat_history]) if chat_history else None

        db_results_str = None
        
        # Handle Database Search mode
        if current_mode == "Database Search 🗃️":
            # First, let LLM suggest an action (function call or raw query)
            llm_suggestion_raw_chunks = list(self.generate_answer(
                user_query, 
                context_documents=retrieved_context_str, 
                chat_history_str=formatted_chat_history_str, 
                current_mode=current_mode
            ))
            llm_suggestion_raw = "".join(llm_suggestion_raw_chunks)

            logger.info(f"LLM raw suggestion for DB query: {llm_suggestion_raw[:500]}...")

            db_output_raw = None
            action_executed = False

            try:
                # Attempt to parse JSON action from LLM's output
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', llm_suggestion_raw, re.DOTALL)
                if json_match:
                    llm_json_output = json.loads(json_match.group(1))
                    
                    action_type = llm_json_output.get("action")

                    if action_type == "call_function":
                        func_name = llm_json_output.get("function_name")
                        func_args = llm_json_output.get("arguments", {})
                        
                        logger.info(f"LLM suggested calling function: {func_name} with args: {func_args}")
                        
                        target_func = self._get_tool_function_by_name(func_name)
                        if target_func:
                            try:
                                db_output_raw = target_func(**func_args)
                                action_executed = True
                                logger.info(f"Successfully executed function: {func_name}")
                            except TypeError as te:
                                db_results_str = f"Error: Function '{func_name}' called with incorrect arguments. Details: {te}"
                                logger.error(db_results_str, exc_info=True)
                            except Exception as e:
                                db_results_str = f"Error executing function '{func_name}': {e}"
                                logger.error(db_results_str, exc_info=True)
                        else:
                            db_results_str = f"Error: Unknown function suggested by LLM: '{func_name}'."
                            logger.warning(db_results_str)
                    
                    elif action_type == "execute_query":
                        query_type = llm_json_output.get("query_type")
                        query_str = llm_json_output.get("query")
                        
                        logger.info(f"LLM suggested executing raw query (type: {query_type}): {query_str[:200]}...")
                        
                        if query_type == "postgres_sql" and self.postgres_connector:
                            db_output_raw = self.postgres_connector.execute_query(query_str)
                            action_executed = True
                            logger.info("Successfully executed raw PostgreSQL query.")
                        elif query_type == "influxdb_flux" and self.influxdb_connector:
                            db_output_raw = self.influxdb_connector.execute_flux_query(query_str)
                            action_executed = True
                            logger.info("Successfully executed raw InfluxDB Flux query.")
                        else:
                            db_results_str = f"Error: Invalid query type '{query_type}' or connector not available."
                            logger.warning(db_results_str)

                # If an action was executed and results obtained, format them
                if action_executed and db_output_raw is not None:
                    if isinstance(db_output_raw, list):
                        if not db_output_raw:
                            db_results_str = "Query executed, but no results found."
                        else:
                            # Custom JSON serializer for datetime objects
                            def datetime_converter(o):
                                if isinstance(o, datetime):
                                    return o.isoformat()
                                raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

                            formatted_rows = []
                            for row in db_output_raw:
                                if isinstance(row, dict):
                                    try:
                                        formatted_rows.append(json.dumps(row, default=datetime_converter))
                                    except TypeError as e:
                                        logger.error(f"TypeError during json.dumps of row {row}: {e}")
                                        formatted_rows.append(str(row)) # Fallback to string if not serializable
                                else:
                                    formatted_rows.append(str(row))
                            db_results_str = "\n".join(formatted_rows)
                    elif isinstance(db_output_raw, dict):
                        def datetime_converter(o):
                            if isinstance(o, datetime):
                                return o.isoformat()
                            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
                        try:
                            db_results_str = json.dumps(db_output_raw, default=datetime_converter)
                        except TypeError as e:
                            logger.error(f"TypeError during json.dumps of dict {db_output_raw}: {e}")
                            db_results_str = str(db_output_raw)
                    else:
                        db_results_str = str(db_output_raw)
                    
                    # Check if the results themselves contain an error message
                    error_in_results = False
                    if isinstance(db_output_raw, list) and db_output_raw and isinstance(db_output_raw[0], dict) and "error" in db_output_raw[0]:
                        error_in_results = True
                    elif isinstance(db_output_raw, dict) and "error" in db_output_raw:
                        error_in_results = True
                    
                    if error_in_results:
                         db_results_str = f"Database returned an error: {db_results_str}"
                         logger.error(f"Database operation returned an error based on 'error' key: {db_results_str}")

                # If no action was executed (e.g., LLM didn't suggest JSON or suggested something invalid)
                elif not action_executed:
                    # Stream the raw LLM suggestion directly to the user
                    for chunk in llm_suggestion_raw_chunks:
                        yield chunk
                    return
            except json.JSONDecodeError as e:
                db_results_str = f"Could not parse LLM's database action suggestion (invalid JSON). Error: {e}. LLM Output: {llm_suggestion_raw[:200]}..."
                logger.error(db_results_str, exc_info=True)
            except Exception as e:
                db_results_str = f"An unexpected error occurred while processing LLM's database suggestion. Error: {e}. LLM Output: {llm_suggestion_raw[:200]}..."
                logger.error(db_results_str, exc_info=True)
            
            # Now, generate the final answer using the database results (if any)
            if db_results_str is not None:
                summary_generator = self.generate_answer(
                    user_query, 
                    context_documents=retrieved_context_str, 
                    chat_history_str=formatted_chat_history_str, 
                    db_results=db_results_str, # Pass the database results for summarization
                    current_mode=current_mode
                )
                yield from summary_generator
            else:
                yield "I couldn't perform a database action based on your request. Please try rephrasing."
                logger.warning("No database action was performed despite being in DB search mode.")
        else:
            # For Conversational and Knowledge Base modes, generate answer directly
            answer_generator = self.generate_answer(
                user_query, 
                context_documents=retrieved_context_str, 
                chat_history_str=formatted_chat_history_str, 
                db_results=db_results_str, # Will be None in these modes
                current_mode=current_mode
            )
            yield from answer_generator