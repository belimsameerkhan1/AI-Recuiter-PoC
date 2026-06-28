import streamlit as st
import pandas as pd
import os
from engine import load_data, rank_candidates

# Set up the page with a clean, professional title
st.set_page_config(page_title="AI Recruiter Dashboard", layout="wide")
st.title("🤖 AI Recruiter Dashboard")
st.markdown("An intelligent semantic ranking engine for technical hiring.")
st.markdown("---")

def render_dashboard(df, is_live=False):
    """Creates an amazing visual dashboard with metrics and heatmaps."""
    score_col = 'Final_Score' if is_live else 'score'
    reasoning_col = 'Reasoning' if is_live else 'reasoning'
    
    # --- 📥 NEW EXPORT BUTTON ---
    # Format the data perfectly for the Hackathon before they download it
    export_df = df.copy()
    if is_live:
        export_df['rank'] = range(1, len(export_df) + 1)
        export_df = export_df[['candidate_id', 'rank', 'Final_Score', 'Reasoning']]
        export_df.columns = ['candidate_id', 'rank', 'score', 'reasoning']
    else:
        export_df = export_df[['candidate_id', 'rank', 'score', 'reasoning']]

    csv_data = export_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Final Submission CSV",
        data=csv_data,
        file_name="team_Red_Reaper.csv",
        mime="text/csv",
        type="primary", # Makes the button a bright, clickable color
        use_container_width=True # Makes it span the whole width!
    )
    st.markdown("<br>", unsafe_allow_html=True)
    # ----------------------------
    
    # 1. Top-Level Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Candidates Processed", len(df))
    col2.metric("Highest AI Score", f"{df[score_col].max():.3f}")
    col3.metric("Average Score", f"{df[score_col].mean():.3f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Charts & Heatmaps Layout
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        st.subheader("📊 Score Distribution")
        st.write("Visual spread of candidate alignment:")
        hist_data = pd.cut(df[score_col], bins=10).value_counts().sort_index()
        hist_data.index = hist_data.index.astype(str)
        st.bar_chart(hist_data)
        
    with right_col:
        st.subheader("🔥 Top Candidates Heatmap")
        st.write("Candidates highlighted by AI semantic score:")
        
        display_df = df[['candidate_id', score_col, reasoning_col]].copy()
        display_df.columns = ['Candidate ID', 'AI Score', 'Reasoning']
        
        styled_df = display_df.style.background_gradient(subset=['AI Score'], cmap='viridis')
        st.dataframe(styled_df, use_container_width=True, height=400)

# 1. The Sandbox Upload Feature (Sidebar)
st.sidebar.header("Judge's Sandbox")
st.sidebar.write("Upload a custom .jsonl file to test the AI engine live.")
uploaded_file = st.sidebar.file_uploader("Upload candidate file:", type=["jsonl", "csv"])

if uploaded_file is not None:
    st.info("Running AI ranking engine on uploaded data...")
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        df = load_data(temp_file_path) 
        job_description = "We are looking for a Software Engineer with strong Python and AI skills." 
        results_df = rank_candidates(df, job_description, 5)
        
        st.success("✅ Live Ranking Complete!")
        render_dashboard(results_df, is_live=True)
        
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

else:
    # IF NO UPLOAD: Show your perfect 100-rank submission
    file_found = None
    for filename in ["team_Red_Reaper.csv", "Red_Reaper.csv", "submission.csv", "final_submission.csv"]:
        if os.path.exists(filename):
            file_found = filename
            break
            
    if file_found:
        default_df = pd.read_csv(file_found)
        render_dashboard(default_df, is_live=False)
    else:
        st.error("🚨 Could not find the final CSV file. Please make sure your 100-row CSV is in the same folder as app.py!")