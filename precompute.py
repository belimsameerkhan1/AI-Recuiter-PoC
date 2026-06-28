import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

print("Loading data...")
df = pd.read_json("data/candidates.jsonl", lines=True)

# Build the context string (same as your engine.py)
df['skills_flat'] = df['skills'].apply(lambda s: " ".join([x.get('name', '') for x in s]) if isinstance(s, list) else "")
df['summary'] = df['profile'].apply(lambda x: x.get('summary', '') if isinstance(x, dict) else '')
df['headline'] = df['profile'].apply(lambda x: x.get('headline', '') if isinstance(x, dict) else '')
df['Context'] = df['headline'] + ". " + df['summary'] + ". Skills: " + df['skills_flat']

print("Embedding 100,000 candidates (Grab a coffee, this will take ~10 mins)...")
# Change this line in precompute.py
model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
candidate_embeddings = model.encode(df['Context'].tolist())

# Save the mathematical vectors to a hyper-fast file
np.save("data/candidate_embeddings.npy", candidate_embeddings)
print("✅ Embeddings saved successfully to data/candidate_embeddings.npy!")