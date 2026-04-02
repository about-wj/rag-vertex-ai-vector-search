import os
import uuid
from typing import List, Dict
import vertexai
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
from google.cloud.aiplatform_v1.types import IndexDatapoint
from google import genai
from google.genai.types import EmbedContentConfig
from vertexai.generative_models import GenerativeModel
import json

# ---------------------------
# CONFIG
# ---------------------------
PROJECT_ID = "bd-host-2026-002"  # Replace with your GCP project ID
LOCATION = "us-central1"     # Google Cloud region
MODEL_NAME = "gemini-embedding-001"
EMBED_DIM = 1408
INDEX_DNAME = "gemini-vector-search-index-version"
ENDPT_DNAME = "gemini-vector-search-endpoint-version"
DEPLOYED_ID = "gemini_vector_search_deployed"

# Route Google GenAI SDK to Vertex AI
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

vertexai.init(project=PROJECT_ID, location=LOCATION)
aiplatform.init(project=PROJECT_ID, location=LOCATION)
genai_client = genai.Client()

# ---------------------------
# 1) Embed helpers (Gemini 001)
# ---------------------------
def embed_texts(texts: List[str], dim: int = EMBED_DIM) -> List[List[float]]:
    resp = genai_client.models.embed_content(
        model=MODEL_NAME,
        contents=texts,
        config=EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",  # good default for RAG docs/queries
            output_dimensionality=dim       # 3072 / 1536 / 768
        ),
    )
    return [e.values for e in resp.embeddings]

# ---------------------------
# 2) Create & deploy Vector Search index (STREAM_UPDATE)
# ---------------------------
def create_index_and_endpoint():
    print("Creating Vector Search index...")
    index = MatchingEngineIndex.create_tree_ah_index(
        display_name=INDEX_DNAME,
        dimensions=EMBED_DIM,                   # MUST match embeddings
        index_update_method="STREAM_UPDATE",    # enables upsert
        distance_measure_type="DOT_PRODUCT_DISTANCE",
        description="Index for Mixture of Experts content",
        approximate_neighbors_count=150,        # Required for tree-AH algorithm
    )
    print(f"Created index: {index.resource_name}")
    
    print("Creating Vector Search endpoint...")
    endpoint = MatchingEngineIndexEndpoint.create(
        display_name=ENDPT_DNAME,
        public_endpoint_enabled=True,          # easiest to start
        description="Endpoint for Mixture of Experts index",
    )
    print(f"Created endpoint: {endpoint.resource_name}")
    
    print("Deploying index to endpoint...")
    endpoint.deploy_index(index=index, deployed_index_id=DEPLOYED_ID)
    print("Index deployed successfully")
    
    return index.resource_name, endpoint.resource_name

# ---------------------------
# 3) Upsert datapoints with content in restricts
# ---------------------------
def upsert_docs(index_name: str, chunks: List[str], ids: List[str] = None):
    print(f"Embedding {len(chunks)} chunks...")
    vecs = embed_texts(chunks)
    
    if not ids:
        ids = [str(uuid.uuid4()) for _ in chunks]
    
    print("Creating datapoints with content in restricts...")
    dps = []
    for i, (id, vec, chunk) in enumerate(zip(ids, vecs, chunks)):
        # Create restrict with content
        restrict = IndexDatapoint.Restriction(
            namespace="content",
            allow_list=[chunk]
        )
        
        # Create datapoint with ID, vector, and restrict
        dp = IndexDatapoint(
            datapoint_id=id, 
            feature_vector=vec,
            restricts=[restrict]
        )
        dps.append(dp)
    
    print(f"Upserting {len(dps)} datapoints to Vector Search...")
    idx = MatchingEngineIndex(index_name=index_name)
    idx.upsert_datapoints(datapoints=dps)
    
    # Return mapping of IDs to chunks for later retrieval
    chunk_map = dict(zip(ids, chunks))
    
    # Save mapping to file as backup
    with open("new_vector_search_mapping.json", 'w') as f:
        json.dump(chunk_map, f, indent=2)
    print(f"Saved mapping to new_vector_search_mapping.json")
    
    return chunk_map

# ---------------------------
# 4) Query (embed query → nearest neighbors)
# ---------------------------
def nearest_neighbors(endpoint_name: str, query: str, k: int = 5):
    print(f"Querying Vector Search for: {query}")
    qvec = embed_texts([query])[0]
    ep = MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_name)
    
    # find_neighbors is the public kNN call on the endpoint
    res = ep.find_neighbors(
        deployed_index_id=DEPLOYED_ID,
        queries=[qvec],
        num_neighbors=k,
    )
    
    # Handle both possible response formats
    if hasattr(res[0], 'neighbors'):
        return res[0].neighbors
    else:
        return res

# ---------------------------
# 5) Generate with Gemini (grounded)
# ---------------------------
def answer_with_gemini(context_snippets: List[str], question: str) -> str:
    model = GenerativeModel("gemini-2.5-flash")  # or 2.5-flash for speed
    context = "\n\n".join(context_snippets)
    prompt = f"""Use only the context to answer. Provide a detailed and comprehensive response with thorough explanations. Include all relevant information from the context. If the context doesn't contain relevant information, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}
Answer (provide a detailed explanation with at least 3-4 paragraphs):"""
    return model.generate_content(prompt).text

# ---------------------------
# PDF Processing
# ---------------------------
def download_from_gcs(gcs_url):
    """Download file from GCS using gsutil"""
    import subprocess
    import tempfile
    
    print(f"Downloading PDF from {gcs_url}...")
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.close()
    
    # Use gsutil to download the file
    subprocess.run(["gsutil", "cp", gcs_url, temp_file.name], check=True)
    print(f"Downloaded to {temp_file.name}")
    
    return temp_file.name

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    import PyPDF2
    
    print("Extracting text from PDF...")
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    
    print(f"Extracted {len(text)} characters of text")
    return text

def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks"""
    print("Chunking text...")
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len and text[end] != ' ':
            # Find the last space within the chunk to avoid cutting words
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
        
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    
    print(f"Created {len(chunks)} chunks")
    return chunks

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == "__main__":
    # Process PDF from GCS
    PDF_URL = "gs://about_test/Full catalog 4-1 duct fan251010.pdf"  # Replace with your GCS PDF path
    pdf_path = download_from_gcs(PDF_URL)
    pdf_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(pdf_text)
    
    try:
        # Step 1: Create index and endpoint
        index_name, endpoint_name = create_index_and_endpoint()
        
        # Step 2: Upsert documents
        chunk_map = upsert_docs(index_name, chunks)
        
        # Clean up temporary file
        import os
        os.unlink(pdf_path)
        
        # Step 3: Test query
        query = "What is Mixture of Experts (MoE) in large language models?"
        print(f"\nTesting query: {query}")
        
        # Step 4: Retrieve relevant chunks
        neighbors = nearest_neighbors(endpoint_name, query, k=3)
        retrieved_chunks = []
        
        print("\nRetrieved documents:")
        for i, nb in enumerate(neighbors):
            # Handle different response formats
            if hasattr(nb, 'datapoint'):
                datapoint_id = nb.datapoint.datapoint_id
                distance = nb.distance
                
                # Try to get content from restricts
                content = None
                if hasattr(nb.datapoint, 'restricts') and nb.datapoint.restricts:
                    for restrict in nb.datapoint.restricts:
                        if restrict.namespace == "content" and restrict.allow_list:
                            content = restrict.allow_list[0]
                            break
                
                # If not found in restricts, try the mapping
                if not content:
                    content = chunk_map.get(datapoint_id, "Content not found")
            else:
                # Alternative format - handle list response
                if isinstance(nb, list):
                    datapoint_id = str(i)
                    distance = 0.0
                    content = "Content format not recognized"
                else:
                    # Dictionary format
                    datapoint_id = nb.get('datapoint_id', str(i))
                    distance = nb.get('distance', 0.0)
                    content = chunk_map.get(datapoint_id, "Content not found")
            
            retrieved_chunks.append(content)
            print(f"Document {i+1} - ID: {datapoint_id}, Distance: {distance:.4f}")
            print(f"Content: {content[:150]}...\n")
        
        # Step 5: Generate answer
        print("\nGenerating answer...")
        answer = answer_with_gemini(retrieved_chunks, query)
        print(f"Answer: {answer}")
        
        print("\nVector Search index created and tested successfully!")
        print(f"Index name: {index_name}")
        print(f"Endpoint name: {endpoint_name}")
        
    except Exception as e:
        print(f"Error: {e}")
