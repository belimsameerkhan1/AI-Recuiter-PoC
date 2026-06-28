import streamlit as st
import pandas as pd
from engine import load_data, rank_candidates, export_submission


@st.cache_data
def get_data(path):
    return load_data(path)




st.set_page_config(page_title="AI Recruiter PoC", layout="wide")
st.title("🚀 Intelligent Candidate Discovery")
st.markdown("Track 1: Data & AI Challenge")


DATA_PATH = "data/sample_candidates.json"


try:
    df = get_data(DATA_PATH)
    st.success(f"✅ Successfully loaded {len(df)} candidates from the database!")
except Exception as e:
    st.error(f"⚠️ Error loading data: {e}")
    st.stop()

st.sidebar.header("Filter Signals")
required_exp = st.sidebar.slider("Minimum Years of Experience", 0, 15, 2)


job_description = st.text_area("Paste the Job Description Here:", height=200)


if 'results' not in st.session_state:
    st.session_state.results = None

if st.button("Rank Candidates"):
    if job_description:
        with st.spinner("AI is analyzing semantics and ranking candidates..."):
          
            st.session_state.results = rank_candidates(df, job_description, required_exp)
    else:
        st.warning("Please enter a job description to search.")


if st.session_state.results is not None:
    results = st.session_state.results
    
    st.subheader("🏆 Top Matches")
    
    display_cols = ['candidate_id', 'years_of_experience', 'Semantic_Score', 'Final_Score', 'Reasoning']
    actual_cols = [col for col in display_cols if col in results.columns]
    
    st.dataframe(results[actual_cols].style.format({
        'Semantic_Score': '{:.2f}',
        'Final_Score': '{:.2f}'
    }))
    
    st.divider()
    st.subheader("📦 Export for Submission")
    if st.button("Generate Final CSV"):
        export_submission(results, "final_submission.csv")
        st.success("✅ CSV Generated! Check your VS Code file list for 'final_submission.csv'.")