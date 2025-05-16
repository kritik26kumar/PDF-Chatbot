import streamlit as st
from Business_Access_Layer.BAL import business_func as bf
from Data_Access_Layer.DAL import data_loader as dl

obj_bal = bf()
obj_dal = dl()

# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat with Files", layout="wide")
    st.header("📄 Chat with PDFs, Images, and Excel Files")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "debug_log" not in st.session_state:
        st.session_state["debug_log"] = []
    if "processed" not in st.session_state:
        st.session_state["processed"] = False
    if "last_uploaded_files" not in st.session_state:
        st.session_state["last_uploaded_files"] = []

    # Display chat history (only user and assistant messages)
    for message in st.session_state["messages"]:
        if message["role"] in ["user", "assistant"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input for user questions
    user_question = st.chat_input("Ask a question about the uploaded files:")
    if user_question:
        # Append user question to chat history
        st.session_state["messages"].append({"role": "user", "content": user_question})
        
        # Display user question immediately
        with st.chat_message("user"):
            st.markdown(user_question)

        # Build context from previous messages (limit to last 5 for brevity)
        context = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" 
             for msg in st.session_state["messages"] 
             if msg["role"] in ["user", "assistant"][-5:-1]]  # Exclude current question
        )
        # Combine context and current question
        full_question = f"Context:\n{context}\n\nCurrent Question:\n{user_question}" if context else user_question

        # Debug: Log question to debug_log and terminal
        st.session_state["debug_log"].append(f"Debug: Sending question: {full_question}")
        print(f"Debug: Sending question: {full_question}")  # Log to terminal

        # Process question with business_func
        try:
            response = obj_bal.user_input(full_question)
            # Handle different response cases
            if response is not None and response.strip():
                # Valid response: store and display
                st.session_state["messages"].append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            elif response == "":
                # Empty response: store and display warning
                error_msg = "Empty response received. The model may not have generated an answer."
                st.session_state["messages"].append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.warning(error_msg)
            else:
                # None response: store and display error
                error_msg = "No response generated. Please check if files are processed or try a different question."
                st.session_state["messages"].append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)
        except Exception as e:
            # Exception: store and display error
            error_msg = f"Error processing question: {str(e)}"
            st.session_state["messages"].append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)

    # Sidebar for file upload
    with st.sidebar:
        st.title("📂 Menu")
        MAX_FILES = 10  # Set maximum number of files allowed
        uploaded_files = st.file_uploader(
            "Upload your PDF, image, or Excel files",
            accept_multiple_files=True,
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "xlsx", "xls"]
        )

        if st.button("Submit & Process"):
            if not uploaded_files:
                st.error("Please upload at least one PDF, image, or Excel file.")
                return

            # Check if the number of files exceeds the limit
            if len(uploaded_files) > MAX_FILES:
                st.error(f"Too many files uploaded. Maximum allowed is {MAX_FILES}. Please upload fewer files.")
                return

            # Check if uploaded files have changed
            file_names = sorted([f.name for f in uploaded_files])
            if file_names != st.session_state["last_uploaded_files"] or not st.session_state["processed"]:
                with st.spinner("Processing files..."):
                    # Process all files using DataLoader.load_files
                    extracted_content = obj_dal.load_files(uploaded_files)

                    if extracted_content:
                        # Pass extracted content directly to BusinessFunc
                        vector_store = obj_bal.get_vector_store(extracted_content)
                        if vector_store:
                            st.success("✅ Files processed successfully!")
                            st.session_state["processed"] = True
                            st.session_state["last_uploaded_files"] = file_names
                            st.write(f"📚 Total chunks created: {len(obj_bal.get_text_chunks(extracted_content)[0])}")
                        else:
                            st.error("Failed to create vector store.")
                    else:
                        st.error("No content extracted from the uploaded files.")
            else:
                st.info("📌 Files already processed. You can ask questions now.")

        # Reset button to clear files, chat history, and debug log
        if st.button("Reset"):
            st.session_state["processed"] = False
            st.session_state["messages"] = []  # Clear chat history
            st.session_state["debug_log"] = []  # Clear debug log
            st.session_state["last_uploaded_files"] = []  # Clear file tracking
            st.success("🌀 Processing state, chat history, and debug log reset. You can re-upload files.")

if __name__ == "__main__":
    main()