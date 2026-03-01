#%%
import pandas as pd

# Load the bigvul data
splits = {'train': 'data/train-00000-of-00001-c6410a8bb202ca06.parquet',
          'validation': 'data/validation-00000-of-00001-d21ad392180d1f79.parquet',
          'test': 'data/test-00000-of-00001-d20b0e7149fa6eeb.parquet'}
df = pd.read_parquet("hf://datasets/bstee615/bigvul/" + splits["train"])


def get_stratified_sample(df, sample_size):
    # Calculate the proportion of each class
    subset_df = df.groupby('vul', group_keys=False).apply(
        lambda x: x.sample(n=max(1, int(sample_size * len(x) / len(df)))),
        include_groups=False
    )

    return subset_df

subset_df = get_stratified_sample(df, 100)

def process_to_json(df, num_rows=50):
    """
    Converts the BigVul DataFrame into a structured format for Mistral fine-tuning.
    """
    output_data = []
    subset_df = df.head(num_rows)

    for _, row in subset_df.iterrows():
        # 'vul' column indicates if the code is vulnerable (1) or a patch/safe (0)
        is_vulnerable = row.get('vul')

        # Construct the JSON object
        entry = {
            "violation_type": str(row.get('CWE ID', 'security_flaw')),
            "severity": "critical" if is_vulnerable == 1 else "low",
            "vulnerable_code": row['func_before'],
            "explanation": f"{row.get('commit_message', 'N/A')}",
            "risk": " ",
            "compliant_patch": row['func_after']
        }
        output_data.append(entry)

    return output_data


data = process_to_json(subset_df)
