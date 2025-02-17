"""This includes all functions used to prompt GPT-4 for descriptions of a dairy farm, a pig farm etc.
"""

from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

from AI_representation_bias_in_farming import module7_3d_plot


def load_prompt_df(megadata):
    """
    Load and filter unique prompts used for DALL-E 3 image generation.

    Parameters
    ----------
    megadata : pandas.DataFrame
        Input DataFrame containing image generation metadata including
        generation type, prompts, and farm information

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing unique prompts with columns:
        - farm_type: Type of farm
        - major_group: Major classification group
        - minor_group: Minor classification group
        - revise_status: Revision status (filtered to "revise" only)
        - generation_type: Type of generation used
        - country: Country information
        - prompt: Text prompt used for generation

    Notes
    -----
    This function:
    - Adds grouping columns using module7_3d_plot.add_grouping
    - Filters for revised prompts only
    - Removes duplicate entries
    - Preserves only essential columns for prompt analysis

    Examples
    --------
    >>> prompt_df = load_prompt_df(megadata)
    >>> print(prompt_df.shape)
    (100, 7)  # Example output showing 100 unique prompts with 7 columns
    """

    megadata = module7_3d_plot.add_grouping(megadata)
    sub_df = megadata[megadata["revise_status"] == "revise"]
    sub_df2 = sub_df[
        [
            "farm_type",
            "major_group",
            "minor_group",
            "revise_status",
            "generation_type",
            "country",
            "prompt",
        ]
    ].drop_duplicates()

    return sub_df2


def calculate_prompt_word_counts(prompts_series):
    """
    Calculate word counts for each unique prompt in a series.

    Parameters
    ----------
    prompts_series : pandas.Series
        Series containing text prompts to analyze

    Returns
    -------
    int
        Maximum word count found across all prompts


    Notes
    -----
    The function handles non-string values by skipping them and can process
    empty series. For duplicate prompts, only unique values are counted.
    Word count is calculated by splitting on whitespace.

    Examples
    --------
    >>> max_count = calculate_prompt_word_counts(df["prompts"])
    >>> print(f"Maximum word count: {max_count}")
    Maximum word count: 25
    """
    # Create a dictionary to store word counts
    word_counts = {}

    # Calculate word count for each unique prompt
    for prompt in prompts_series.unique():
        if isinstance(prompt, str):  # Check if prompt is a string
            word_count = len(prompt.split())
            word_counts[prompt] = word_count

    # Find the maximum word count and its corresponding prompt
    if word_counts:
        max_count = max(word_counts.values())
    else:
        max_count = 0

    return max_count


def process_text_completions(
    prompt_df,
    repetition_per_prompt=10,
    model="gpt-4",
    system_prompt=None,
    max_tokens=80,
    temperature=0.2,
    start_index=0,
    end_index=None,
    store_results=True,
):
    """
    Process prompts with multiple repetitions and preserve prompt metadata.

    Parameters
    ----------
    prompt_df : pandas.DataFrame
        DataFrame containing prompts and their metadata with columns:
        farm_type, major_group, minor_group, revise_status,
        generation_type, country, prompt
    repetition_per_prompt : int, optional
        Number of independent API calls to make for each prompt
    model : str, optional
        Model identifier (e.g., "gpt-4")
    system_prompt : str, optional
        System message to set context for all completions
    max_tokens : int, optional
        Maximum tokens allowed in the completion
    temperature : float, optional
        Controls randomness in the output (0.0 to 2.0)
    start_index : int, optional
        Starting index in prompt_df
    end_index : int, optional
        Ending index (exclusive) in prompt_df
    store_results : bool, optional
        Whether to store results in a CSV file

    Returns
    -------
    pandas.DataFrame
        DataFrame containing all prompt metadata plus completion results:
        - Original columns from prompt_df
        - completion: Generated text
        - model: Model used
        - completion_tokens: Number of tokens in completion
        - system_fingerprint: Model system fingerprint

    Notes
    -----
    Results are saved to 'results/megadata/gpt4_describe_farm.csv' after
    each completion if store_results is True.
    """
    # Initialize OpenAI client
    load_dotenv()
    client = OpenAI()

    # Set end_index to DataFrame length if not specified
    if end_index is None:
        end_index = len(prompt_df)

    # Initialize results storage
    results_data = []

    # Process each prompt in the specified range
    for idx in range(start_index, end_index):
        # Get the row containing prompt and metadata
        row = prompt_df.iloc[idx]

        # Perform multiple repetitions for each prompt
        for rep_idx in range(repetition_per_prompt):
            result = process_single_prompt(
                prompt=row["prompt"],
                model=model,
                client=client,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Create result dictionary with all metadata
            result_data = {
                # Original metadata from prompt_df
                "farm_type": row["farm_type"],
                "major_group": row["major_group"],
                "minor_group": row["minor_group"],
                "revise_status": row["revise_status"],
                "generation_type": row["generation_type"],
                "country": row["country"],
                "prompt": row["prompt"],
                # API call results
                "completion": result.choices[0].message.content,
                "model": model,
                "completion_tokens": result.usage.completion_tokens,
            }

            results_data.append(result_data)

            # Optional: Save results after each completion
            if store_results:
                df = pd.DataFrame(results_data)
                df.to_csv(
                    (Path() / "results" / "megadata" / "gpt4_describe_farm.csv"),
                    index=False,
                )

    return pd.DataFrame(results_data)


def process_single_prompt(
    prompt,
    model,
    client,
    system_prompt=None,
    max_tokens=150,
    temperature=0.2,
):
    """
    Processes a single text prompt and returns the completion result.

    This function constructs the messages for the API call, including an optional
    system prompt, and handles the completion request for a single prompt.

    Parameters:
        prompt (str): The text prompt to process
        model (str): Model identifier to use for completion
        client (OpenAI): Initialized OpenAI client
        system_prompt (Optional[str]): System message to set context
        max_tokens (int): Maximum tokens in the completion
        temperature (float): Controls randomness in the output

    Returns:
        dict: API response containing the completion and metadata

    Example:
        client = OpenAI()
        result = process_single_prompt(
            "Explain quantum computing",
            "gpt-4",
            client,
            system_prompt="You are a physics expert"
        )
    """
    # Construct messages
    messages = []

    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add user prompt
    messages.append({"role": "user", "content": prompt})

    # Call API with parameters
    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
    )

    return response
