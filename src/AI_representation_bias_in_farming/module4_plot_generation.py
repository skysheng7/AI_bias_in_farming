"""Library of functions used to generate grid plots based on image metadata
"""

def filter_data(df, generation_types, farm_type=None, model="dall-e-3", country=None):
    """
    Filter the DataFrame based on specified generation types, and optionally by farm type and country.

    Parameters:
    df (pandas.DataFrame): The DataFrame to filter.
    generation_types (list or set): A list or set of generation types to include in the filtered DataFrame.
    farm_type (str, optional): The farm type to filter by. Defaults to None.
    country (str, optional): The country to filter by. Defaults to None.
    model (str): which model did we use, Options: 'dall-e-3'

    Returns:
    pandas.DataFrame: The filtered DataFrame containing only the rows that match the specified criteria.
    """
    filtered_df = df[df['generation_type'].isin(generation_types)]
    filtered_df = filtered_df[filtered_df['model'] == model]
    if farm_type is not None:
        filtered_df = filtered_df[filtered_df['farm_type'].isin(farm_type)]
    if country is not None:
        filtered_df = filtered_df[filtered_df['country'].isin(country)]
    return filtered_df

def generate_stopwords(prompt_text, additional_stop_list = set()):
    """
    Generate a custom set of stopwords by combining standard stopwords, words from a given prompt, 
    and any additional specified stopwords.

    Args:
        prompt_text (str): The original prompt text to extract words from for custom stopwords. 
                           Words in this text will be excluded from the word cloud.
        additional_stop_list (set, optional): Additional words to include in the stopwords. 
                                              Defaults to an empty set.

    Returns:
        set: A set of stopwords combining the standard stopwords, words from the prompt, 
             and any additional specified words.
    """
    # Create a set of stopwords from the prompt
    if prompt_text is not None:
        # Combine original stopwords with words from the prompt
        custom_stopwords = STOPWORDS.union(set(prompt_text.lower().split()))
        custom_stopwords = custom_stopwords.union(additional_stop_list)

    else:
        custom_stopwords = STOPWORDS
    
    return custom_stopwords