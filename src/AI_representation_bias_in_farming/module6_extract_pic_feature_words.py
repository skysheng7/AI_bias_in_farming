"""Library of functions used to see which words are the key features for describing images of 
   extensive VS intensive farms
"""

from pathlib import Path

import pandas as pd
from PIL import Image


def image_copy_and_paste_all_rows(filtered_rows, source_base, dest_folder, model):
    """
    Copy and resize multiple images based on filtered DataFrame rows.

    Parameters
    ----------
    filtered_rows (pandas.DataFrame): DataFrame containing rows with image information,
                                    must have 'generation_type' and 'file' columns
    source_base (Path): Base path where source images are stored
    dest_folder (Path): Destination folder path where processed images will be saved
    model (str): Model name used to construct source image path (e.g., 'dall-e-3')

    Returns
    -----
    None
    """

    if len(filtered_rows) > 0:
        for _, row in filtered_rows.iterrows():
            # Get source image path
            source_file = (
                source_base / (model + "-images") / row["generation_type"] / row["file"]
            )

            image_copy_and_paste(source_file, dest_folder)


def image_copy_and_paste(source_file, dest_folder):
    """
    Copy an image file to a destination folder with resizing and quality reduction.

    Parameters
    ----------
    source_file (Path): Source path of the image file to be processed
    dest_folder (Path): Destination folder path where the processed image will be saved

    Side Effects
    -----------
    - Creates a resized copy of the source image in the destination folder
    - Prints error messages if source file is not found or processing fails

    Retunrs
    -----
    None
    """
    if source_file.exists():
        try:
            # Open and resize image
            img = Image.open(source_file)
            # Reduce quality by resizing to 50% of original size
            new_size = tuple(dim // 2 for dim in img.size)
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

            # Save to destination
            dest_file = dest_folder / source_file.name
            img_resized.save(dest_file, quality=85, optimize=True)

        except Exception as e:
            print(f"Error processing {source_file}: {e}")
    else:
        print(f"Source file not found: {source_file}")


def process_farm_images(
    df,
    farm_type,
    condition_type,
    word_list,
    source_base,
    dest_base,
    model,
    grouping_style="by word",
    negative_word_list=None,
):
    """
    Filter images based on word occurrences and organize them into folders.

    Parameters
    ----------
    df (pandas.DataFrame):
        DataFrame with image metadata, must include 'farm_type' and 'model' columns
    farm_type (str):
        Farm type to filter for ('dairy' or 'pig')
    condition_type (str):
        Condition type for folder organization ('extensive' or 'intensive')
    word_list (list):
        List of words to filter by (must be column names in df)
    source_base (Path):
        Base path for source images
    dest_base (Path):
        Base path for destination folders
    model (str):
        Model name for filtering and path construction (e.g., 'dall-e-3')
    grouping_style (str), optional:
        How to organize output folders:
        - "by word": separate folder for each word
        - "combine": single folder for all matches
        Default is "by word"
    negative_word_list (list), optional:
        List of words that must have count of 0 in filtered results
        Default is None

    Returns
    -------
    None
    """
    for word in word_list:

        if grouping_style == "by word":
            dest_folder = dest_base / farm_type / condition_type / word
        elif grouping_style == "combine":
            dest_folder = dest_base / farm_type / condition_type

        # Create destination folder if it doesn't exist
        if not dest_folder.exists():
            dest_folder.mkdir(parents=True)

        # Filter rows where the word count is 1
        filtered_rows = df[
            (df[word] == 1) & (df["farm_type"] == farm_type) & (df["model"] == model)
        ]

        # More efficient negative word filtering
        if negative_word_list is not None:
            # Filter rows where all columns in negative_word_list are 0
            filtered_rows = filtered_rows[
                filtered_rows[negative_word_list].eq(0).all(axis=1)
            ]

        image_copy_and_paste_all_rows(filtered_rows, source_base, dest_folder, model)


def create_feature_column(
    df, farm_type_filter, word_list, feature_name, negative_word_list=None
):
    """
    Create a binary column in a DataFrame based on farm type and word list conditions.

    Parameters
    ----------
    df (pandas.DataFrame): Input DataFrame containing farm data and word count columns
    farm_type_filter (str): The farm type to filter for (e.g., 'dairy' or 'pig')
    word_list (list of str): List of words to check for in the DataFrame columns
    feature_name (str):  Name of the new binary column to create
    negative_word_list (list), optional:
        List of words that must have count of 0 in filtered results
        Default is None

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

    # For each word in the negative_word_list, if any negative word exist (==1),
    # revert the feature name to be 0
    if negative_word_list is not None:
        for neg_word in negative_word_list:
            if neg_word in df.columns:
                df.loc[mask & (df[neg_word] == 1), feature_name] = 0

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
    raw_freq_df,
    dairy_extensive,
    dairy_intensive,
    pig_extensive,
    pig_intensive,
    dairy_extensive_negative_list,
    dairy_intensive_negative_list,
    pig_extensive_negative_list,
    pig_intensive_negative_list,
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
    df = create_feature_column(
        raw_freq_df,
        "dairy",
        dairy_extensive,
        "exten",
        negative_word_list=dairy_extensive_negative_list,
    )
    df = create_feature_column(
        df,
        "dairy",
        dairy_intensive,
        "inten",
        negative_word_list=dairy_intensive_negative_list,
    )
    df = create_feature_column(
        df,
        "pig",
        pig_extensive,
        "exten",
        negative_word_list=pig_extensive_negative_list,
    )
    df = create_feature_column(
        df,
        "pig",
        pig_intensive,
        "inten",
        negative_word_list=pig_intensive_negative_list,
    )

    # Calculate statistics
    summary_df = calculate_feature_stats(df)

    # sort
    summary_df = summary_df.sort_values(["model", "farm_type", "generation_type"])

    output_file = (
        Path("..") / "results" / "megadata" / "intensive_extensive_summary.csv"
    )
    summary_df.to_csv(output_file, index=False)

    return summary_df
