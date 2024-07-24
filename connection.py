from pymilvus import connections

# Attempt to connect to Milvus
try:
    connections.connect("default", host="localhost", port="19530")
    print("Successfully connected to Milvus.")
except Exception as e:
    print(f"Failed to connect to Milvus: {e}")
