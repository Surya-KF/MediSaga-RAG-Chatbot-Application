import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.mixture import GaussianMixture
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import json
import os
import ollama
import logging
from functools import lru_cache

# Adjust logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Reduce Ollama library logging

# Load configuration
try:
    with open('config.json') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    logger.warning("config.json not found. Using default values.")
    config = {
        'sbert_model': 'all-MiniLM-L6-v2',
        'llm_model': 'llama2',
        'chunk_size': 100,
        'pdf_directory': 'data',
        'milvus_host': 'localhost',
        'milvus_port': '19530',
        'collection_name': 'medical_qa',
        'vector_dim': 384,
        'top_k': 5,
        'batch_size': 10
    }

# Initialize SentenceTransformer model
sbert_model = SentenceTransformer(config['sbert_model'])

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text += f"\n\nPage {i + 1}\n\n"
            text += page.extract_text()
    return text

def chunk_text(text, chunk_size=config['chunk_size']):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        sentence_length = len(word_tokenize(sentence))
        if current_length + sentence_length > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = sentence_length
        else:
            current_chunk += " " + sentence
            current_length += sentence_length

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def embed_chunks(chunks):
    return sbert_model.encode(chunks)

@lru_cache(maxsize=None)
def summarize_single_text(text):
    response = ollama.generate(
        model=config['llm_model'],
        prompt=f"Summarize the following text:\n\n{text}\n\nSummary:"
    )
    return response['response'].strip()

def summarize_texts_ollama(texts):
    summaries = []
    for i in range(0, len(texts), config['batch_size']):
        batch = texts[i:i+config['batch_size']]
        batch_summaries = [summarize_single_text(text) for text in batch]
        summaries.extend(batch_summaries)
    return summaries

def create_raptor_index(embeddings, texts, depth=3):
    if depth == 0 or len(texts) <= 1:
        return texts[0] if texts else ""

    n_components = min(5, len(texts))
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(embeddings)
    labels = gmm.predict(embeddings)

    clusters = {i: [] for i in range(n_components)}
    for i, label in enumerate(labels):
        clusters[label].append(texts[i])

    summaries = []
    for cluster in clusters.values():
        if cluster:
            cluster_summary = summarize_texts_ollama(cluster)[0]
            summaries.append(cluster_summary)

    summary_embeddings = embed_chunks(summaries)
    return create_raptor_index(summary_embeddings, summaries, depth - 1)

def initialize_milvus():
    try:
        connections.connect("default", host=config['milvus_host'], port=config['milvus_port'])
        if not utility.has_collection(config['collection_name']):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config['vector_dim']),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            schema = CollectionSchema(fields, "Medical QA collection")
            collection = Collection(name=config['collection_name'], schema=schema)
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 1024}
            }
            collection.create_index("embedding", index_params)
        else:
            collection = Collection(config['collection_name'])
        collection.load()
        return collection
    except Exception as e:
        logger.error(f"Error initializing Milvus: {e}")
        return None

def process_pdf(pdf_path, collection):
    try:
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)

        raptor_index = create_raptor_index(embeddings, chunks)

        entities = [
            {
                "embedding": embedding.tolist(),
                "metadata": json.dumps({
                    "text": chunk,
                    "pdf_path": pdf_path,
                    "raptor_index": raptor_index
                })
            }
            for embedding, chunk in zip(embeddings, chunks)
        ]

        collection.insert(entities)
        logger.info(f"Processed and indexed PDF: {pdf_path}")
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {e}")

def process_pdf_directory(pdf_directory, collection):
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            process_pdf(pdf_path, collection)

def main():
    nltk.download('punkt', quiet=True)
    collection = initialize_milvus()
    if collection is None:
        logger.error("Failed to initialize Milvus. Exiting.")
        return

    pdf_directory = config['pdf_directory']
    if not os.path.exists(pdf_directory):
        logger.error(f"PDF directory {pdf_directory} does not exist.")
        return

    process_pdf_directory(pdf_directory, collection)
    collection.flush()
    logger.info("PDF processing and indexing completed.")

if __name__ == "__main__":
    main()