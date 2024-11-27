"""Library of functions used to count the occurance of words in each revised prompt/image descriptions
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# import the megadata dataframe, and sort by the length of revised prompts
from pathlib import Path
megadata_file = Path(".") / "results" / "megadata" / "image_megadata.csv"
megadata = pd.read_csv(megadata_file, header=0)


def count_word_freq_in_revised_prompt(megadata, min_freq=20):
    """This function creates bag of word to count if certain words appear in the text
    description in each row.

    Parameters:
    megadata (dataframe): the dataframe recording megadata about generated images
    min_freq (int, optional): sets a minimum document frequency threshold,
                            a filter that says "only keep words that appear in
                            at least min_freq documents." Defaults to 10.
    
    Returns:
    pandas.DataFrame: Original megadata with additional columns for word frequencies
    """

    # Create the vectorizer with specified parameters
    vec = CountVectorizer(min_df=min_freq, binary=True, stop_words="english")
    
    # Extract the revised prompts and convert them to a list
    revised_prompts = megadata['revised_prompt'].fillna('')  # Handle any potential NaN values
    
    # Fit and transform the revised prompts into a binary bag of words 
    prompts_vec = vec.fit_transform(revised_prompts)
    
    # Create a DataFrame from the vectorized prompts
    # Each column will represent a word, and each row will show whether that word appears
    prompts_vec_df = pd.DataFrame(
        data=prompts_vec.toarray(),
        columns=vec.get_feature_names_out(),
        index=megadata.index  # Preserve the original index for proper alignment
    )
    
    # Combine the original megadata with the word frequency data
    # Using axis=1 to concatenate horizontally (add new columns)
    result_df = pd.concat([megadata, prompts_vec_df], axis=1)
    
    return result_df


    # sort megadata by "prompt"
    # iterate through every unique "prompt" in the "prompt" column in megadata
    # create a subset dataframe for each unique prompt, see for all the "revised_prompt", count the unique number of "revised_prompt" using .unique()
    # if the total number of unique "revised_prompt" <= 3 (could be NA, needs to be replaced with ""), don't do anything
    # else take the "revised_prompt" and count the total occurance of each word in the columns=vec.get_feature_names_out() across all rows, summarize it into a table
    
    
    