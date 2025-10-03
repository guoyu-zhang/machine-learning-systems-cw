import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI, BackgroundTasks
import uvicorn
from pydantic import BaseModel
import pandas as pd
import regex as re
import os
import tqdm
from concurrent.futures import Future
import time
import queue
import threading

DATA_PATH = "task-2/data/movies.csv"
EMBEDDING_PATH = "task-2/data/embeddings.npy"
MAX_BATCH_SIZE = 8
MAX_WAITING_TIME = 1

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Initialize FastAPI
app = FastAPI()
request_queue = queue.Queue()

# Example documents in memory
# documents = [
#     "Cats are small furry carnivores that are often kept as pets.",
#     "Dogs are domesticated mammals, not natural wild animals.",
#     "Hummingbirds can hover in mid-air by rapidly flapping their wings."
# ]

# Load data
def clean_text(text):
    return re.sub(r'\[\d+\]', '', text)

def load_context(data_path):
    df = pd.read_csv(data_path)
    documents = "Title: " + df["Title"] + " Plot: " + df["Plot"]
    documents = documents.apply(clean_text).tolist()
    return documents

documents = load_context(DATA_PATH)

#------------------Local START--------------------
# 1. Load embedding model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(device)

# Basic Chat LLM
chat_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
#------------------Local END----------------------

# #------------------Cluster START------------------
# # 1. Load embedding model
# LOCAL_MODEL_PATH = "/home/s1808795/.cache/huggingface/hub/models--intfloat--multilingual-e5-large-instruct/snapshots/84344a23ee1820ac951bc365f1e91d094a911763"
# embed_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
# embed_model = AutoModel.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
# # Basic Chat LLM
# LOCAL_CHAT_MODEL_PATH = "/home/s1808795/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6"
# chat_tokenizer = AutoTokenizer.from_pretrained(LOCAL_CHAT_MODEL_PATH, local_files_only=True)
# chat_model = AutoModelForCausalLM.from_pretrained(LOCAL_CHAT_MODEL_PATH, local_files_only=True)
# chat_pipeline = pipeline("text-generation", model=chat_model, tokenizer=chat_tokenizer)
# #------------------Cluster END--------------------

## Hints:

### Step 3.1:
# 1. Initialize a request queue
# 2. Initialize a background thread to process the request (via calling the rag_pipeline function)
# 3. Modify the predict function to put the request in the queue, instead of processing it immediately

### Step 3.2:
# 1. Take up to MAX_BATCH_SIZE requests from the queue or wait until MAX_WAITING_TIME
# 2. Process the batched requests

def get_embedding(text: str) -> np.ndarray:
    """Compute a simple average-pool embedding."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Precompute document embeddings
if os.path.exists(EMBEDDING_PATH):
    print("Existing embeddings found. Loading...")
    doc_embeddings = np.load(EMBEDDING_PATH)
else:
    print("Embeddings not found. Computing...")
    doc_embeddings = []
    for doc in tqdm.tqdm(documents):
        doc_embeddings.append(get_embedding(doc))
    doc_embeddings = np.vstack(doc_embeddings)
    np.save(EMBEDDING_PATH, doc_embeddings)

### You may want to use your own top-k retrieval method (task 1)
def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    """Retrieve top-k docs via dot-product similarity."""
    sims = doc_embeddings @ query_emb.T
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]
    return [documents[i] for i in top_k_indices]

def rag_pipeline(query: str, k: int = 2) -> str:
    query_emb = get_embedding(query)
    retrieved_docs = retrieve_top_k(query_emb, k)

    context = "\n".join(retrieved_docs)
    prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer:\n"

    generated_text = chat_pipeline(prompt, max_new_tokens=50, do_sample=True)[0]["generated_text"]

    answer_start = generated_text.find("Answer:")
    if answer_start != -1:
        generated_text = generated_text[answer_start + len("Answer:"):].strip()

    return generated_text


# Define request model
class QueryRequest(BaseModel):
    query: str
    k: int = 2

# Background processing thread
def process_requests():
    while True:
        batch = []
        start_time = time.time()

        while len(batch) < MAX_BATCH_SIZE and (time.time() - start_time) < MAX_WAITING_TIME:
            try:
                batch.append(request_queue.get(timeout=MAX_WAITING_TIME))  # Correct tuple handling
            except queue.Empty:
                break

        if batch:
            results = [(request['payload'], request['future'], rag_pipeline(request['payload'].query, request['payload'].k)) for request in batch]
            for req, fut, result in results:
                fut.set_result(result)

# Start the background thread
threading.Thread(target=process_requests, daemon=True).start()

async def wait_for_future(future: Future):
    future.result()  # This will complete in the background

@app.post("/rag")
def predict(payload: QueryRequest, background_tasks: BackgroundTasks):
    future = Future()
    request_queue.put({"payload": payload, "future": future})
    background_tasks.add_task(wait_for_future, future)  # Non-blocking response
    return {"query": payload.query, "result": future.result()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
