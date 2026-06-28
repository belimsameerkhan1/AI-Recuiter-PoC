import argparse
from engine import load_data, rank_candidates, export_submission

if __name__ == "__main__":
    # 1. Set up the command line arguments the judges will pass
    parser = argparse.ArgumentParser(description="Rank candidates for Redrob Hackathon")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates file")
    parser.add_argument("--out", type=str, required=True, help="Path to save the output CSV")
    args = parser.parse_args()

    # 2. Define the target Job Description for the hackathon
    job_description = "Senior AI Engineer — Founding Team. Company: Redrob AI. Location: Pune/Noida, India. Experience Required: 5-9 years."
    required_exp = 5

    print(f"Loading data from {args.candidates}...")
    
    # 3. Run your exact pipeline from engine.py
    df = load_data(args.candidates)
    ranked_df = rank_candidates(df, job_description, required_exp)
    
    # 4. Save using your export function
    export_submission(ranked_df, args.out)