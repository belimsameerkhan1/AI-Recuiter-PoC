import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')

def load_data(file_path):
    """Loads the candidates instantly using Parquet if available."""
    
    # 1. THE SPEED HACK: If the fast parquet file exists, load it in 1 second
    parquet_path = "data/candidates.parquet"
    if os.path.exists(parquet_path) and "sample" not in file_path:
        print("Loading instant Parquet data...")
        return pd.read_parquet(parquet_path)
        
    # 2. THE FALLBACK: For the small sandbox sample JSON on GitHub
    print("Loading JSON data...")
    if file_path.endswith('.jsonl'):
        df = pd.read_json(file_path, lines=True)
    else:
        df = pd.read_json(file_path)
        
    # Flatten the data (only needed if falling back to JSON)
    df['years_of_experience'] = df['profile'].apply(lambda x: x.get('years_of_experience', 0) if isinstance(x, dict) else 0)
    
    if 'signals' in df.columns:
        df['response_rate'] = df['signals'].apply(lambda x: x.get('recruiter_response_rate', 1.0) if isinstance(x, dict) else 1.0)
        df['interview_rate'] = df['signals'].apply(lambda x: x.get('interview_completion_rate', 1.0) if isinstance(x, dict) else 1.0)
    else:
        df['response_rate'] = 1.0
        df['interview_rate'] = 1.0
        
    df['skills_flat'] = df['skills'].apply(lambda s: " ".join([x.get('name', '') for x in s]) if isinstance(s, list) else "")
    df['summary'] = df['profile'].apply(lambda x: x.get('summary', '') if isinstance(x, dict) else '')
    df['headline'] = df['profile'].apply(lambda x: x.get('headline', '') if isinstance(x, dict) else '')
    df['Context'] = df['headline'] + ". " + df['summary'] + ". Skills: " + df['skills_flat']
    
    return df

def generate_offline_reasoning(row):
    """Generates a dynamic explanation for why the AI picked them."""
    return f"Ranked for {row.get('years_of_experience', 0)} years of experience. Strong semantic alignment (AI Score: {row['Semantic_Score']:.2f})."

def rank_candidates(df, job_description, required_experience):
    print("AI is embedding the job description...")
    jd_embedding = model.encode([job_description])

    npy_path = "data/candidate_embeddings.npy"
    
    # --- ☢️ THE NUCLEAR DEBUG BLOCK ---
    if len(df) > 1000:
        try:
            print("Loading pre-computed BGE candidate memory...")
            candidate_embeddings = np.load(npy_path)
        except FileNotFoundError:
            import streamlit as st
            import os
            # This forces the web app to show you exactly where it is looking
            st.error(f"🚨 CRITICAL ERROR: I cannot find the memory file! I am looking exactly here: {os.path.abspath(npy_path)}")
            st.stop()
    else:
        # Fallback for the small GitHub sandbox
        print("Calculating candidate embeddings on the fly...")
        candidate_embeddings = model.encode(df['Context'].tolist())

    print("Calculating semantic similarities...")
    similarities = cosine_similarity(jd_embedding, candidate_embeddings)[0]
    df['Semantic_Score'] = similarities

    def calculate_final_score(row):
        score = row['Semantic_Score']
        
        if row.get('response_rate', 0) < 0.10 or row.get('interview_rate', 0) == 0.0:
            return 0.0 
        
        yoe = row.get('years_of_experience', 0)
        if 6 <= yoe <= 8:
            score += 0.15  
        elif 5 <= yoe <= 9:
            score += 0.10  
            
        if row.get('response_rate', 0) > 0.80:
            score += 0.05
            
        profile_text = str(row).lower()
        product_companies = ['google', 'meta', 'swiggy', 'razorpay', 'amazon', 'microsoft', 'flipkart', 'freshworks']
        service_companies = ['tcs', 'wipro', 'infosys', 'accenture', 'cognizant', 'hcl', 'capgemini']
        
        has_product = any(company in profile_text for company in product_companies)
        has_service = any(company in profile_text for company in service_companies)
        
        if has_product:
            score += 0.10  
        elif has_service and not has_product:
            score -= 0.05 
            
        return score

    df['Final_Score'] = df.apply(calculate_final_score, axis=1)

    ranked_df = df.sort_values(by='Final_Score', ascending=False).head(100)
    ranked_df['Reasoning'] = ranked_df.apply(generate_offline_reasoning, axis=1)

    return ranked_df

def export_submission(ranked_df, output_filename="submission.csv"):
    """Exports the final CSV perfectly matched to the Hackathon validator schema."""
    submission_df = ranked_df.copy()
    submission_df['rank'] = range(1, len(submission_df) + 1)
    submission_df = submission_df.rename(columns={'Final_Score': 'score', 'Reasoning': 'reasoning'})
    final_cols = ['candidate_id', 'rank', 'score', 'reasoning']
    submission_df = submission_df[final_cols]
    submission_df.to_csv(output_filename, index=False)
    print(f"✅ Successfully exported candidates to {output_filename}")