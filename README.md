# 👔 Team Red_Reaper - Intelligent Candidate Discovery & Ranking Engine

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Redrob AI](https://img.shields.io/badge/Talent%20Intelligence-Redrob%20Hackathon%20v4-red?style=for-the-badge)](https://redrob.com/)
[![ML Framework](https://img.shields.io/badge/Engine-Sentence--Transformers%20%7C%20Streamlit-green?style=for-the-badge&logo=scikit-learn)](https://huggingface.co/sentence-transformers)

An advanced, fully offline talent matching and ranking system built specifically for the **Redrob Hackathon**. This engine parses candidate profiles (`.jsonl`),
calculates semantic alignment using the `BAAI/bge-small-en-v1.5` model, applies heuristic scoring based on experience and company backgrounds, and presents the top 100 candidates through a highly interactive Streamlit dashboard.

---

## 📐 Pipeline Architecture

The system processes candidate profiles through a bulletproof, fault-tolerant pipeline that handles missing data gracefully, culminating in a dynamic visual dashboard:

```mermaid

flowchart TD
    A[Judge's Sandbox / .jsonl Upload] --> B[engine.py: Robust Data Loader]
    B --> C[Bulletproof Feature Extraction]
    
    C -->|Headline, Summary, Skills| D[Sentence-Transformers BGE Embedding]
    C -->|Response Rate, Interviews| E[Behavioral Signals Filtering]
    C -->|YoE, Company History| F[Heuristic Scoring Adjustments]
    
    D --> G[Cosine Similarity Calculation]
    E --> H[Final Score Aggregation]
    F --> H
    G --> H
    
    H -->|Sort & Extract Top 100| I[app.py: Streamlit Dashboard]
    I --> J[Score Distribution & Heatmap Visualization]
    I --> K[Hackathon Schema CSV Export]
🚀 Execution & Usage
The entire pipeline is wrapped in a user-friendly Streamlit application that requires zero external API calls and runs entirely on the CPU to meet strict hackathon resource limits.
1. Install dependencies:
pip install -r requirements.txt
2. Launch the AI Dashboard:
streamlit run app.py
This will open the web sandbox where you can view the default 100-rank submission or upload custom .jsonl files for live AI ranking.

📝 Dynamically Generated Reasoning Example
For transparency, the submission output contains a natural-language description (reasoning column) for the top-100 ranked candidates. For example:

"Ranked for 7.8 years of experience. Strong semantic alignment (AI Score: 0.74)."


🔒 Security & Git Protection
To comply with hackathon rules and prevent pushing large files to GitHub, .gitignore has been pre-configured to block data artifacts that exceed GitHub's 100MB limit:

Large JSONL logs (*.jsonl)

Raw pre-computed embedding memories (*.npy)

Parquet caches (*.parquet)

Python environment files, caches, and IDE configs (.venv/, __pycache__/, .vscode/)

🏆 Submission Validation Checklists
Our engine automatically runs formatting constraints to guarantee 100% compliance with the Redrob submission validator before exporting:

1 Count Constraint: System extracts exactly the top 100 candidates.

2 Schema Compliance: Outputs exactly 4 columns: candidate_id, rank, score, and reasoning.

3 Strict Monotonicity: Verifies the candidate list is strictly sorted by Final_Score in descending order, with ties broken deterministically by ID.

4 Robust Extraction: Safely handles missing profile, skills, or signals objects without crashing via explicit df.columns checks.
