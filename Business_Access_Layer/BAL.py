import os
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class BusinessFunc:
    """Handles RAG processing and Gemini-based Q&A with conversation memory from .md files."""

    def get_text_chunks(self, md_file_paths: List[str]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """Split Markdown file content into chunks.

        Args:
            md_file_paths: List of paths to Markdown files.

        Returns:
            Tuple of (chunks, metadatas, errors).
        """
        if not md_file_paths:
            st.warning("No Markdown files provided for chunking.")
            logging.warning("No Markdown files provided for chunking.")
            return [], [], []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300
        )

        chunks, metadatas, errors = [], [], []
        for md_path in md_file_paths:
            if not os.path.isfile(md_path):
                errors.append(f"File not found: {md_path}")
                logging.warning(f"File not found: {md_path}")
                continue
            try:
                with open(md_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if content.strip().startswith("[Error"):
                        errors.append(f"Skipping invalid content in {md_path}")
                        logging.warning(f"Skipping invalid content in {md_path}")
                        continue
                    if not content.strip():
                        errors.append(f"Empty file: {md_path}")
                        logging.warning(f"Empty file: {md_path}")
                        continue
                    splits = text_splitter.split_text(content)
                    chunks.extend(splits)
                    metadatas.extend([{"source": md_path}] * len(splits))
            except Exception as e:
                errors.append(f"Error reading {md_path}: {str(e)}")
                logging.error(f"Error reading {md_path}: {str(e)}")

        return chunks, metadatas, errors

    def get_vector_store(self, md_file_paths: List[str], vector_store_path: str = "faiss_index") -> Optional[FAISS]:
        """Create or load an FAISS vector store from .md files.

        Args:
            md_file_paths: List of paths to Markdown files.
            vector_store_path: Path to save/load FAISS index.

        Returns:
            FAISS vector store or None if failed.
        """
        if not GOOGLE_API_KEY:
            st.error("Google API key is missing.")
            logging.error("Google API key is missing.")
            return None

        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=GOOGLE_API_KEY,
                model="models/embedding-001"
            )

            if os.path.exists(vector_store_path):
                try:
                    vector_store = FAISS.load_local(
                        vector_store_path, embeddings, allow_dangerous_deserialization=True
                    )
                    st.info("Loaded existing vector store.")
                    logging.info("Loaded existing vector store.")
                    return vector_store
                except Exception as e:
                    st.error(f"Error loading vector store: {str(e)}")
                    logging.error(f"Error loading vector store: {str(e)}")
                    return None

            chunks, metadatas, errors = self.get_text_chunks(md_file_paths)
            if errors:
                for error in errors:
                    st.warning(error)
                    logging.warning(error)
            if not chunks:
                st.error("No valid chunks available to create vector store.")
                logging.error("No valid chunks available to create vector store.")
                return None

            documents = [Document(page_content=text, metadata=meta) for text, meta in zip(chunks, metadatas)]
            vector_store = FAISS.from_documents(documents, embedding=embeddings)
            vector_store.save_local(vector_store_path)
            st.success("Vector store created successfully.")
            logging.info("Vector store created successfully.")
            return vector_store

        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            logging.error(f"Error creating vector store: {str(e)}")
            return None

    def get_conversational_chain(self, selected_model: str) -> Optional[Any]:
        """Create a conversational chain for Q&A.

        Args:
            selected_model: Name of the model (e.g., 'Gemini').

        Returns:
            Conversational chain or None if failed.
        """
        prompt_template = '''
        You are a helpful assistant. Use the provided context extracted from Markdown files to answer the question as accurately as possible.

        Instructions:
        - Base your answer strictly on the context below.
        - Do NOT guess or fabricate any information.
        - If the answer is not explicitly mentioned in the context, reply: "The answer is not available in the provided context."

        Context:
        {context}

        Question:
        {question}

        Answer:
        '''
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        try:
            if selected_model == "Gemini":
                if not GOOGLE_API_KEY:
                    st.error("Google API key is missing.")
                    logging.error("Google API key is missing.")
                    return None
                model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
            else:
                st.error(f"Invalid model selected: {selected_model}")
                logging.error(f"Invalid model selected: {selected_model}")
                return None

            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            return chain
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            logging.error(f"Error initializing model: {str(e)}")
            return None

    def user_input(self, user_question: str, vector_store_path: str = "faiss_index") -> Optional[str]:
        """Process user question and return answer using RAG.

        Args:
            user_question: The question to answer.
            vector_store_path: Path to the FAISS index.

        Returns:
            Answer as a string or None if failed.
        """
        try:
            if not GOOGLE_API_KEY:
                st.error("Google API key is missing.")
                logging.error("Google API key is missing.")
                return None

            embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=GOOGLE_API_KEY, model="models/embedding-001"
            )

            if not os.path.exists(vector_store_path):
                st.error("Vector store not found. Please process Markdown files first.")
                logging.error("Vector store not found.")
                return None

            new_db = FAISS.load_local(
                vector_store_path, embeddings, allow_dangerous_deserialization=True
            )
            docs = new_db.similarity_search(user_question)

            chain = self.get_conversational_chain("Gemini")
            if chain is None:
                return None

            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            return response["output_text"]

        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            logging.error(f"Error processing question: {str(e)}")
            return None