# MediSaga: RAG Chatbot Application
## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Medical Textbooks](#medical-textbooks)
5. [Setup Instructions](#setup-instructions)
   - [Prerequisites](#prerequisites)
   - [Steps](#steps)
6. [Running the System](#running-the-system)
7. [Configuration](#configuration)
8. [Key Components](#key-components)
   - [PDF Processing, Indexing, and Query Processing (first.py)](#pdf-processing-indexing-and-query-processing-firstpy)
   - [User Interface and Integration (third.py)](#user-interface-and-integration-thirdpy)
9. [Dependencies](#dependencies)
10. [Milvus Integration](#milvus-integration)
11. [Ollama Integration](#ollama-integration)
12. [User Interface](#user-interface)
13. [Evaluation and Performance](#evaluation-and-performance)
14. [Future Improvements](#future-improvements)
15. [Troubleshooting](#troubleshooting)
16. [Contact](#contact)

## Project Overview
MediSaga is a Retrieval-Augmented Generation (RAG) chatbot application designed for medical question answering. It leverages RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) indexing with Milvus vector database to process and index medical textbooks, and then uses this information to answer user queries.

## Features
- PDF text extraction and processing
- RAPTOR indexing using Milvus and Gaussian Mixture Models
- Query expansion and retrieval using Sentence Transformers
- Integration with Llama2 for question answering
- Milvus vector database for efficient similarity search
- Streamlit-based user interface for easy interaction

## Project Structure
- `first.py`: Contains functions for PDF processing, text chunking, and RAPTOR indexing
- `second.py`: Implements the query processing and answer generation logic using Milvus
- `third.py`: Streamlit-based user interface
- `config.json`: Configuration file for various parameters
- `requirements.txt`: List of required Python packages

## Medical Textbooks
The following medical textbooks are used as the knowledge base for this application:
1. Encyclopedia of Infectious Diseases
2. Gale Encyclopedia of Medicine, Vol. 1, 2nd Edition
3. Oxford Handbook of Oncology, 4th Edition

These books are stored in the `data` directory and are processed to create the knowledge base for the chatbot. Please ensure you have the rights to use these books for this purpose.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Docker and Docker Compose (for running Milvus)

### Steps

1. Clone the repository:
git clone [Your Repository URL]
cd [Your Repository Name]
Copy
2. Create a virtual environment and activate it:
``` 
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```  
4. Install the required packages:
```
pip install -r requirements.txt
```
5. Set up Milvus using Docker:
- Ensure Docker and Docker Compose are installed on your system
- Run the following command to start Milvus:
  ```
  docker-compose up -d
  ```
- This will start Milvus on the default port 19530

7. Configure the application:
- Edit `config.json` to set your desired parameters, including the Milvus host and port.

8. Prepare your data:
- Place your PDF textbooks in the `data` directory.

## Running the System
1. Process and index the PDFs:
   ```
   python first.py
   ```
2. Creating the raptor index and stored in Milvus Vectorbase:
   ```
   python second.py
   ```
3. Start the Streamlit interface:
   ```
   streamlit run third.py
    ```
4. Open a web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Configuration
The `config.json` file contains various configuration parameters:
- `sbert_model`: The Sentence-BERT model to use for embeddings (default: 'all-MiniLM-L6-v2')
- `llm_model`: The language model to use for question answering (default: 'llama2')
- `chunk_size`: The size of text chunks for processing (default: 100)
- `pdf_directory`: The directory containing the PDF files (default: 'data')
- `milvus_host`: Milvus server host (default: 'localhost')
- `milvus_port`: Milvus server port (default: '19530')
- `collection_name`: The name of the Milvus collection (default: 'medical_qa')
- `vector_dim`: Dimension of the vector embeddings (default: 384)
- `top_k`: Number of top results to retrieve (default: 5)
- `batch_size`: Batch size for processing (default: 10)

## Key Components

### PDF Processing and Indexing (`first.py`)
- Extracts text from PDF files using `pdfplumber`
- Chunks text into segments of approximately 100 tokens
- Embeds chunks using Sentence-BERT
- Implements RAPTOR indexing using Gaussian Mixture Models
- Stores embeddings and metadata in Milvus

### Query Processing (`second.py`)
- Connects to Milvus collection
- Performs similarity search using Milvus
- Re-ranks retrieved results
- Generates answers using Llama3 model via Ollama

### User Interface (`third.py`)
- Implements a Streamlit-based chat interface
- Displays query results and sources
- Provides a sidebar with information about the application and source documents



https://github.com/user-attachments/assets/9c3843a6-b164-44b6-9742-677d2fe61be3



## Dependencies
Major dependencies include:
- Python 3.8+
- Milvus (via Docker)
- Sentence-Transformers
- PyTorch
- NLTK
- scikit-learn
- pdfplumber
- Streamlit
- Ollama

For a complete list, see `requirements.txt`.

## Milvus Integration
The project uses Milvus, an open-source vector database, for efficient similarity search:
- Milvus is run using Docker for easy setup and management
- The system creates a collection named 'medical_qa' in Milvus
- Embeddings and metadata are stored in this collection
- Similarity search is performed using the L2 distance metric

## Ollama Integration
The project uses Ollama to interact with the Llama2 model:
- Ollama is used for text summarization in the RAPTOR indexing process
- It's also used for generating final answers to user queries

## User Interface
The Streamlit-based user interface provides:
- A chat-like interface for asking medical questions
- Display of generated answers
- Information about the top sources used to generate the answer
- A sidebar with application information and a list of source documents

## Evaluation and Performance
The system's performance depends on:
- The quality and relevance of the indexed textbooks
- The effectiveness of the RAPTOR indexing and retrieval process
- The capabilities of the Llama3 model
- The perfomance of the device

Users can assess the relevance of answers based on the provided source information.

## Future Improvements
- Implement more advanced re-ranking algorithms
- Enhance the user interface with additional features like history saving
- Integrate with medical ontologies for improved query understanding
- Implement a feedback mechanism for continuous improvement
- Optimize Milvus indexing parameters for better performance

## Troubleshooting
- If you encounter issues with Milvus connection, ensure Docker is running and the Milvus container is healthy
- For Ollama-related issues, check that the Llama3 model is properly installed and accessible
  ```
  ollama pull llama3
  ```

## Contact
For any questions or clarifications, please contact [KF SURYA] at [suryakf04@gmail.com ].
