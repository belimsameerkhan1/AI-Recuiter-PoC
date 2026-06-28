import pandas as pd

print("⏳ Reading the massive JSONL file (this will take a couple of minutes)...")
df = pd.read_json("data/candidates.jsonl", lines=True)

print("⚙️ Flattening the nested data...")
# Pre-calculate all the nested JSON so the app doesn't have to do it later
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

print("🚀 Saving to hyper-fast Parquet format...")
df.to_parquet("data/candidates.parquet")
print("✅ Done! Your data is now optimized for instant loading.")