import argparse
import pandas as pd 
from pathlib import Path

current_dir = Path(__file__).resolve().parent

def parse_mean(val):
    try:
        if isinstance(val, str) and "±" in val:
            return float(val.split("±")[0].strip())
        return float(val)
    except:
        return None
    
def parse_relevant_cols(df):
    for col in ["declared_coverage", "empirical_coverage", "compliance"]:
        df[f"{col}_mean"] = df[col].apply(parse_mean)
    return df

def generate_summaries(df, output_dir):
    all_tasks = df['task'].unique()
    all_providers = df['provider'].unique()
    all_models = df['model'].unique()
    
    for task in all_tasks:
        task_df = df[df['task'] == task].copy()
        
        full_grid = pd.MultiIndex.from_product(
            [all_providers, all_models], names=["provider", "model"]
        ).to_frame(index=False)
        
        merged = full_grid.merge(task_df, on=["provider", "model"], how="left")
        
        for col in ["declared_coverage_mean", "empirical_coverage_mean", "compliance_mean"]:
            merged[col] = merged[col].fillna(0.0)

        merged["detail"] = merged["declared_coverage"].apply(
            lambda x: "test not ran for this model" if pd.isna(x) else ""
        )


        summary = merged[[
            'run_id', 'provider', 'model', 'declared_coverage_mean', 'empirical_coverage_mean', 'compliance_mean', 'detail'
        ]].sort_values(by=['provider', 'model'])

        filename = f"{task}_summary.csv".replace(" ", "_")
        summary.to_csv(output_dir / filename, index=False, sep=";")
        print(f"summary of task {task} saved in {output_dir / filename}" )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate per-task summaries from json schema bench evaluation results.")
    parser.add_argument("--run-id", required=True, help="Run ID used to locate eval_results.csv")
    args = parser.parse_args()
    
    run_id = args.run_id
    output_dir = current_dir / "results" / run_id / "openai_compatible"
    general_results_file = (
        current_dir / "outputs" / run_id / "openai_compatible" / "eval_results.csv"
    )

    df = pd.read_csv(general_results_file, sep=";")
    df = parse_relevant_cols(df)
    generate_summaries(df, output_dir)
