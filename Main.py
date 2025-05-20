import streamlit as st
from Business_Access_Layer.BAL import BusinessFunc as bf
from Data_Access_Layer.DAL import DataLoader as dl
import os
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize classes
obj_bal = bf()
obj_dal = dl()

def main():
    st.set_page_config(page_title="Chat with PDFs(LLamaParse)", layout="wide")
    st.header("📄 Chat with PDFs(LLamaParse)")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "debug_log" not in st.session_state:
        st.session_state["debug_log"] = []
    if "processed" not in st.session_state:
        st.session_state["processed"] = False

    # Display previous messages
    for message in st.session_state["messages"]:
        if message["role"] in ["user", "assistant"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # User input
    user_question = st.chat_input("Ask a question about the uploaded files:")
    if user_question:
        st.session_state["messages"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Include last 5 messages for context
        context_messages = st.session_state["messages"][-6:-1]  # 5 most recent before current
        context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in context_messages])
        full_question = f"Context:\n{context}\n\nCurrent Question:\n{user_question}" if context else user_question

        st.session_state["debug_log"].append(f"[Debug] Sending question: {full_question}")
        logging.info(f"[Debug] Sending question: {full_question}")

        try:
            response = obj_bal.user_input(full_question)
            if response and response.strip():
                st.session_state["messages"].append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            else:
                error_msg = "⚠️ Empty response. Try rephrasing your question."
                st.session_state["messages"].append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.warning(error_msg)
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.session_state["messages"].append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)

    # Sidebar - File Upload
    with st.sidebar:
        st.title("📂 Menu")
        MAX_FILES = 10
        uploaded_files = st.file_uploader(
            "Upload PDF files", accept_multiple_files=True, type=["pdf"]
        )

        if st.button("Submit & Process"):
            if not uploaded_files:
                st.error("❗ Please upload at least one PDF file.")
                return

            if len(uploaded_files) > MAX_FILES:
                st.error(f"❗ Too many files. Max allowed is {MAX_FILES}.")
                return

            if "processed" not in st.session_state or not st.session_state["processed"]:
                with st.spinner("🔄 Extracting and processing content..."):
                    # Save uploaded PDFs to disk as temp .md files
                    temp_dir = "temp_md"
                    os.makedirs(temp_dir, exist_ok=True)
                    md_file_paths = []

                    # Process PDFs with DataLoader
                    documents, errors = obj_dal.extract_text_from_pdf(uploaded_files, temp_dir)
                    if errors:
                        for error in errors:
                            st.error(error)
                            st.session_state["debug_log"].append(f"[Debug] DataLoader error: {error}")
                            logging.error(error)

                    for i, doc in enumerate(documents):
                        content = doc.text if hasattr(doc, "text") else ""
                        if content and not content.strip().startswith("[Error"):
                            md_path = os.path.join(temp_dir, f"file_{i}.md")
                            with open(md_path, "w", encoding="utf-8") as f:
                                f.write(content)
                            md_file_paths.append(md_path)
                        else:
                            st.session_state["debug_log"].append(f"[Debug] Skipping invalid document {i}: {content}")
                            logging.warning(f"Skipping invalid document {i}: {content}")

                    # Process .md files through BusinessFunc
                    vector_store = obj_bal.get_vector_store(md_file_paths)
                    if vector_store:
                        st.success(f"✅ {len(md_file_paths)} file(s) processed and vector store created!")
                        st.session_state["processed"] = True
                    else:
                        st.error("❌ Failed to create vector store.")
                        st.session_state["debug_log"].append("[Debug] Failed to create vector store.")

        # Show debug logs
        st.subheader("Debug Log")
        for log in st.session_state["debug_log"]:
            st.write(log)

        # Reset system
        if st.button("Reset"):
            st.session_state["processed"] = False
            st.session_state["messages"] = []
            st.session_state["debug_log"] = []
            temp_dir = "temp_md"
            vector_store_path = "faiss_index"
            for path in [temp_dir, vector_store_path]:
                if os.path.exists(path):
                    try:
                        shutil.rmtree(path)
                        st.session_state["debug_log"].append(f"[Debug] Deleted {path}")
                        logging.info(f"Deleted {path}")
                    except Exception as e:
                        st.session_state["debug_log"].append(f"[Debug] Delete error for {path}: {str(e)}")
                        logging.error(f"Delete error for {path}: {str(e)}")
            st.success("🔁 System reset. Upload new files to start again.")

if __name__ == "__main__":
    main()