import streamlit as st
from pymilvus import connections, Collection
import logging
from second import process_query, initialize_milvus
from first import process_pdf_directory
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open('config.json') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    logger.warning("config.json not found. Using default values.")
    config = {
        'pdf_directory': 'data',
        'milvus_host': 'localhost',
        'milvus_port': '19530',
        'collection_name': 'medical_qa'
    }

def main():
    # Create a centered div with a logo
    st.markdown("""
        <div style="display: flex; justify-content: center;">
            <img src="https://www.freepnglogos.com/uploads/medicine-logo-png-1.png" alt="Logo" width="100">
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>MediSage</h1>", unsafe_allow_html=True)

    collection = initialize_milvus()
    if collection is None:
        st.error("Failed to connect to Milvus. Please check the logs for more details.")
        return

    # Process existing PDFs
    pdf_directory = config['pdf_directory']
    if not os.path.exists(pdf_directory):
        st.error(f"PDF directory {pdf_directory} does not exist.")
        return


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your medical question?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Searching and generating answer..."):
            answer, sources = process_query(prompt, collection)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer)
                st.subheader("Sources:")
                for source in sources[:3]:  # Display top 3 sources
                    st.write(f"- {os.path.basename(source['pdf_path'])} (Relevance: {1 / (1 + source['distance']):.2f})")

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})

    collection.release()

if __name__ == "__main__":
    main()