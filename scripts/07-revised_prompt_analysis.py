import pandas as pd
from pathlib import Path
import click

from AI_representation_bias_in_farming import utils

@click.command()
def main():
    megadata = utils.read_megadata()
    
    # Step 1: Filter with multiple conditions
    no_revise_df = megadata[
        (megadata['generation_type'].str.contains('no_revise', case=False)) & 
        (megadata['model'] == 'dall-e-3') & 
        (~megadata['generation_type'].str.contains('reality', case=False))
    ]

    # Step 2: Create cleaned prompts column
    no_revise_df['cleaned_revised_prompt'] = no_revise_df['revised_prompt'].apply(clean_prompt)

    # Step 3: Select specific columns (corrected syntax)
    no_revise_df = no_revise_df[[
        'generation_type', 
        'country', 
        'farm_type', 
        'prompt', 
        'cleaned_revised_prompt'
    ]]

    # Step 4: Group by the specified columns, including NaN values
    grouped = no_revise_df.groupby([
        'generation_type', 
        'country', 
        'farm_type', 
        'prompt', 
        'cleaned_revised_prompt'
    ], dropna=False)

    # Step 2: Count rows in each group
    counts = grouped.size()

    # Step 3: Convert to dataframe with counts in a new column
    counts_df = counts.reset_index(name='count')

    # Step 4: Sort for readability (optional)
    counts_df = counts_df.sort_values(['farm_type', 'generation_type', 'country'])

    # save counts_df
    cur_file_path = Path() / "results" / "megadata" / "revised_prompt_count.csv"
    counts_df.to_csv(cur_file_path, index=False)
    

# Preprocess revised prompts: lowercase, trim, remove trailing periods
def clean_prompt(prompt):
    if isinstance(prompt, str):
        return prompt.lower().strip().rstrip('.')
    return prompt

# Only execute this code if the script is run directly
if __name__ == "__main__":
    main()