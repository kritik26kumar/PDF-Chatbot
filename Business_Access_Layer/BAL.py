from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class business_func:
    """Class to handle RAG processing and question-answering."""
    def get_text_chunks(self, extracted_content):
        """Split extracted content into chunks."""
        if not extracted_content:
            st.error("No content provided for chunking.")
            return []

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = []
        metadatas = []
        for item in extracted_content:
            splits = text_splitter.split_text(item["content"])
            chunks.extend(splits)
            metadatas.extend([{"type": item["type"], "source": item["source"]}] * len(splits))
        return chunks, metadatas

    def get_vector_store(self, extracted_content):
        """Create and save vector store from extracted content."""
        try:
            chunks, metadatas = self.get_text_chunks(extracted_content)
            if not chunks:
                st.error("No chunks available to create vector store.")
                return None

            embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=GOOGLE_API_KEY, model="models/embedding-001"
            )
            vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas)
            vector_store.save_local("faiss_index")
            return vector_store
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None

    def get_conversational_chain(self):
        """Create conversational chain using Gemini model."""
        prompt_template = '''
        You are a helpful assistant. Use the provided context extracted from PDF, Excel, or image files to answer the question as accurately as possible.

        Instructions:
        - Base your answer strictly on the context below.
        - Include the source file and content type (text, table, image) when relevant.
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
            if not GOOGLE_API_KEY:
                st.error("Google API key is missing.")
                return None
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            return chain
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            return None

    def user_input(self, user_question):
        """Process user question and generate response."""
        try:
            if not user_question:
                st.error("Please provide a question.")
                return None

            embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=GOOGLE_API_KEY, model="models/embedding-001"
            )
            if not os.path.exists("faiss_index"):
                st.error("Vector store not found. Please process files first.")
                return None

            new_db = FAISS.load_local(
                "faiss_index", embeddings, allow_dangerous_deserialization=True
            )
            docs = new_db.similarity_search(user_question, k=3)

            chain = self.get_conversational_chain()
            if chain is None:
                return None

            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            response_text = response["output_text"]
            return response_text
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            return None