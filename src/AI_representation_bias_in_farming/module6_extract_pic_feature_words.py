"""Library of functions used to see which words are the key features for describing images of 
   extensive VS intensive farms
"""

from pathlib import Path

import pandas as pd
from PIL import Image


def process_farm_images(
    df, farm_type, condition_type, word_list, source_base, dest_base, model
):
    """
    Filter images based on word occurrences and organize them into folders.

    Parameters:
    df (DataFrame): dataframe with image metadata
    farm_type (string): 'dairy' or 'pig'
    condition_type (string): 'extensive' or 'intensive'
    word_list (list): list of words to filter by
    source_base (path): base path for source images
    dest_base (path): base path for destination folders

    Return:
    None
    """
    for word in word_list:
        # Create destination folder if it doesn't exist
        dest_folder = dest_base / farm_type / condition_type / word
        if not dest_folder.exists():
            dest_folder.mkdir(parents=True)

        # Filter rows where the word count is 1
        filtered_rows = df[
            (df[word] == 1) & (df["farm_type"] == farm_type) & (df["model"] == model)
        ]

        if len(filtered_rows) > 0:
            for _, row in filtered_rows.iterrows():
                # Get source image path
                source_file = (
                    source_base
                    / (model + "-images")
                    / row["generation_type"]
                    / row["file"]
                )

                if source_file.exists():
                    try:
                        # Open and resize image
                        img = Image.open(source_file)
                        # Reduce quality by resizing to 50% of original size
                        new_size = tuple(dim // 2 for dim in img.size)
                        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

                        # Save to destination
                        dest_file = dest_folder / row["file"]
                        img_resized.save(dest_file, quality=85, optimize=True)

                    except Exception as e:
                        print(f"Error processing {source_file}: {e}")
                else:
                    print(f"Source file not found: {source_file}")


def create_feature_column(df, farm_type_filter, word_list, feature_name):
    """
    Create a binary column in a DataFrame based on farm type and word list conditions.

    Parameters
    ----------
    df (pandas.DataFrame): Input DataFrame containing farm data and word count columns
    farm_type_filter (str): The farm type to filter for (e.g., 'dairy' or 'pig')
    word_list (list of str): List of words to check for in the DataFrame columns
    feature_name (str):  Name of the new binary column to create

    Returns
    -------
    pandas.DataFrame
        DataFrame with new binary column ("inten" stands for intensive; and "exten" stands
        for extensive) where 1 indicates rows matching the farm type
        and having a count of 1 for any word in word_list that are features of a intensive
        or extensive farm, 0 otherwise.

    """
    # Create copy of dataframe to avoid warnings
    df = df.copy()

    # Filter for farm type
    mask = df["farm_type"] == farm_type_filter

    # For each word in list, update feature column if word count is 1
    for word in word_list:
        if word in df.columns:
            df.loc[mask & (df[word] == 1), feature_name] = 1

    return df


def calculate_feature_stats(df):
    """
    Calculate summary statistics for intensive and extensive images grouped by specific columns.

    Parameters
    ----------
    df (pandas.DataFrame): Input DataFrame containing farm data with 'inten'
                            and 'exten' columns and grouping columns
                            ('generation_type', 'country', 'farm_type', 'prompt', 'model')

    Returns
    -------
    pandas.DataFrame
    Summary DataFrame containing the following columns:
    - generation_type : str
        Generation type of image
    - country : str
        Country location
    - farm_type : str
        Type of farm: dairy or pig
    - prompt : str
        Prompt used
    - model : str
        Model used
    - total_rows : int
        Total number of rows for each unique combination of generation_type, country, farm_typem, model
    - inten_sum : int
        total number of images including intensive features
    - exten_sum : int
        total number of images including extensive features
    - inten_pct : float
        Percentage of intensive images
    - exten_pct : float
        Percentage of extensive images

    """

    # Define our grouping columns for finding unique instances
    group_columns = ["generation_type", "country", "farm_type", "prompt", "model"]

    # Initialize a list to store our summary data
    summary_data = []

    # Find unique combinations of our grouping columns
    unique_combinations = df[group_columns].drop_duplicates()

    # For each unique combination in our data
    for _, combo in unique_combinations.iterrows():
        # Get all rows matching this combination
        matching_rows = df[
            (df["prompt"] == combo["prompt"]) & (df["model"] == combo["model"])
        ]

        total_rows = len(matching_rows)
        inten_sum = matching_rows["inten"].sum()
        exten_sum = matching_rows["exten"].sum()
        inten_pct = round((inten_sum / total_rows), 2)
        exten_pct = round((exten_sum / total_rows), 2)

        # Create a dictionary with all the information
        row_data = {
            "generation_type": combo["generation_type"],
            "country": combo["country"],
            "farm_type": combo["farm_type"],
            "prompt": combo["prompt"],
            "model": combo["model"],
            "total_rows": total_rows,
            "inten_sum": inten_sum,
            "exten_sum": exten_sum,
            "inten_pct": inten_pct,
            "exten_pct": exten_pct,
        }

        # Add this combination's data to our summary list
        summary_data.append(row_data)

    # Create DataFrame from our summary data
    summary_df = pd.DataFrame(summary_data)

    return summary_df


def intensive_extensive_calculation(
    raw_freq_df, dairy_extensive, dairy_intensive, pig_extensive, pig_intensive
):
    """
    Calculate intensive and extensive feature statistics for dairy and pig farms and save results.

    Parameters
    ----------
    raw_freq_df (pandas.DataFrame): Input DataFrame containing farm data with
                                    word frequency columns and metadata
                                    (must include 'farm_type' and specified word columns)
    dairy_extensive (list): a list of words that are features of an image of extensive dairy farm
    dairy_intensive (list): a list of words that are features of an image of intensive dairy farm
    pig_extensive (list): a list of words that are features of an image of extensive pig farm
    pig_intensive (list): a list of words that are features of an image of intensive pig farm

    Returns
    -------
    pandas.DataFrame
        Summary DataFrame containing statistics for intensive and extensive features,
        including:
        - Grouping columns (generation_type, country, farm_type, prompt, model)
        - Feature sums and percentages for both intensive and extensive characteristics
        - Total rows for each unique combination

    Side Effects
    -----------
    Saves the summary DataFrame to '../results/megadata/intensive_extensive_summary'
    as a CSV file without index
    """
    raw_freq_df = raw_freq_df.copy()
    # Initialize feature column with 0
    raw_freq_df["exten"] = 0
    raw_freq_df["inten"] = 0
    # Create feature columns
    df = create_feature_column(raw_freq_df, "dairy", dairy_extensive, "exten")
    df = create_feature_column(df, "dairy", dairy_intensive, "inten")
    df = create_feature_column(df, "pig", pig_extensive, "exten")
    df = create_feature_column(df, "pig", pig_intensive, "inten")

    # Calculate statistics
    summary_df = calculate_feature_stats(df)

    output_file = (
        Path("..") / "results" / "megadata" / "intensive_extensive_summary.csv"
    )
    summary_df.to_csv(output_file, index=False)

    return summary_df
