import numpy as np
from pymilvus import connections, Collection
import json
import ollama
import logging
from sentence_transformers import SentenceTransformer

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
        'sbert_model': 'all-MiniLM-L6-v2',
        'llm_model': 'llama3',
        'milvus_host': 'localhost',
        'milvus_port': '19530',
        'collection_name': 'medical_qa',
        'top_k': 5
    }

# Initialize SentenceTransformer model
sbert_model = SentenceTransformer(config['sbert_model'])

def search_milvus(collection, query_vector, top_k=config['top_k']):
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["metadata"]
    )
    return results

def summarize_texts_ollama(texts):
    summaries = []
    for text in texts:
        response = ollama.generate(
            model=config['llm_model'],
            prompt=f"Summarize the following text:\n\n{text}\n\nSummary:"
        )
        summary = response['response'].strip()
        summaries.append(summary)
    return summaries

def retrieve_and_rank(query, collection):
    query_vector = sbert_model.encode([query])[0].tolist()
    search_results = search_milvus(collection, query_vector)

    retrieved_texts = []
    for hits in search_results:
        for hit in hits:
            metadata = json.loads(hit.entity.get('metadata'))
            retrieved_texts.append({
                "content": metadata['text'],
                "raptor_index": metadata['raptor_index'],
                "pdf_path": metadata['pdf_path'],
                "distance": hit.distance
            })

    # Re-rank based on relevance (you can implement more sophisticated re-ranking here)
    retrieved_texts.sort(key=lambda x: x['distance'])

    return retrieved_texts

def generate_answer(query, context):
    response = ollama.generate(
        model=config['llm_model'],
        prompt=f"Question: {query}\n\nContext: {context}\n\nAnswer:"
    )
    return response['response'].strip()

def process_query(query, collection):
    retrieved_texts = retrieve_and_rank(query, collection)
    context = " ".join([text['content'] for text in retrieved_texts])
    answer = generate_answer(query, context)
    return answer, retrieved_texts

def initialize_milvus():
    try:
        connections.connect("default", host=config['milvus_host'], port=config['milvus_port'])
        collection = Collection(config['collection_name'])
        collection.load()
        return collection
    except Exception as e:
        logger.error(f"Error connecting to Milvus: {e}")
        return None

if __name__ == "__main__":
    collection = initialize_milvus()
    if collection:
        # Example usage
        query = "What are the symptoms of COVID-19?"
        answer, sources = process_query(query, collection)
        print(f"Answer: {answer}")
        print("Sources:")
        for source in sources[:3]:  # Print top 3 sources
            print(f"- {source['pdf_path']} (Relevance: {1 / (1 + source['distance']):.2f})")
    else:
        print("Failed to initialize Milvus connection.")