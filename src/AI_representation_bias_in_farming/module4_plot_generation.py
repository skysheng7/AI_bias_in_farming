"""Library of functions used to generate grid plots based on image metadata
"""

import os
import re
import random
import textwrap
from pathlib import Path
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk import FreqDist
from nltk.util import ngrams


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
    filtered_df = df[df["generation_type"].isin(generation_types)]
    filtered_df = filtered_df[filtered_df["model"] == model]
    if farm_type is not None:
        filtered_df = filtered_df[filtered_df["farm_type"].isin(farm_type)]
    if country is not None:
        filtered_df = filtered_df[filtered_df["country"].isin(country)]
    return filtered_df


def generate_word_cloud(
    text_list, prompt_text=None, additional_stop_list=set(), seed=7
):
    """
    Generate a word cloud image from a list of descriptions or prompts, excluding words found
    in the original prompt and any additional specified stopwords.

    Args:
        text_list (list of str): A list of strings, such as revised prompts or descriptions,
                                 to be combined and visualized in the word cloud.
        prompt_text (str, optional): The original prompt text. Words in this prompt will be excluded
                                     from the word cloud. Defaults to None.
        additional_stop_list (set, optional): Additional words to exclude from the word cloud.
                                              Defaults to an empty set.

    Returns:
        PIL.Image.Image: The generated word cloud image as a PIL image object.
    """
    # Concatenate the list of text into a single string
    if text_list is not None:
        combined_text = " ".join(text_list)
    else:
        combined_text = ""

    # Create a set of stopwords from the prompt
    custom_stopwords = generate_stopwords(prompt_text, additional_stop_list)

    # count uni-grams, bi-grams, and tri-grams frequency
    ngram_frequencies = compute_ngram_frequencies(
        combined_text, custom_stopwords, max_ngram=3
    )

    # Generate the word cloud, excluding words in the prompt
    wordcloud = WordCloud(
        colormap="ocean",
        width=400,
        height=400,
        background_color="white",
        random_state=seed,
    ).generate_from_frequencies(ngram_frequencies)

    return wordcloud.to_image()


def compute_ngram_frequencies(text, custom_stopwords, max_ngram=3):
    """
    Compute the frequencies of uni-grams, bi-grams, tri-grams, etc., up to the specified n-gram length,
    combining them into a single frequency dictionary without duplication. Higher-order n-grams take precedence.

    Args:
        text (str): The input text to analyze.
        custom_stopwords (set): A set of stopwords combining the standard stopwords, words from the prompt,
                                and any additional specified words.
        max_ngram (int): The maximum length of n-grams to include (e.g., 3 includes uni-grams, bi-grams, and tri-grams). Default is 3.

    Returns:
        dict: A dictionary where keys are n-grams (as strings) and values are their frequencies in the text.
    """
    # Step 1: Tokenize text into words
    tokens = [
        word.lower()
        for word in re.findall(r"\b\w+\b", text)
        if word.lower() not in custom_stopwords
    ]

    # Step 2: Create n-grams and count frequencies for each n up to max_ngram
    ngram_dicts = []
    for n in range(1, max_ngram + 1):
        ngrams_freq = FreqDist(ngrams(tokens, n))
        # Convert n-grams to strings and add to list of dictionaries
        ngram_dicts.append({" ".join(k): v for k, v in ngrams_freq.items()})

    # Step 3: Combine frequencies without duplication, prioritizing higher-order n-grams
    combined_freq = {}
    for ngram_dict in reversed(ngram_dicts):  # Start with the highest-order n-grams
        for phrase, freq in ngram_dict.items():
            if phrase not in combined_freq:
                combined_freq[phrase] = freq

    return combined_freq


def generate_stopwords(prompt_text, additional_stop_list=set()):
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


def plot_text(ax, text, farm_type, max_character_per_line=27):
    """
    Display wrapped text in an axis with centered alignment, wrapping by number of words.

    Args:
        ax (matplotlib.axes._axes.Axes): The axis to plot the text.
        text (str): The text to be displayed.
        farm_types (list): List of farm types to display.
        words_per_line (int): Number of words per line for wrapping. Defaults to 7.
        max_character_per_line (int): max number of characters per line in a plot
    """
    # Make "dairy" or "pig" bold by using Matplotlib's mathtext
    bold_text = text.replace(farm_type, rf"$\mathbf{{{farm_type}}}$")

    # Group words into lines based on words_per_line
    wrapped_text = textwrap.fill(bold_text, width=max_character_per_line)

    # Display wrapped text in the center of the axis
    ax.text(0.5, 0.5, wrapped_text, ha="center", va="center", fontsize=18, wrap=True)
    ax.axis("off")
    ax.grid(True)


def plot_wordcloud(ax, text_list, prompt_text, additional_stop_list, seed):
    """
    Display a word cloud image on an axis.

    Args:
        ax (matplotlib.axes._axes.Axes): The axis to plot the word cloud.
        text_list (list of str): A list of strings to be included in the word cloud.
        prompt_text (str): The original prompt text to exclude words from the word cloud.
        additional_stop_list (set): Additional words to exclude from the word cloud.
    """
    wordcloud_image = generate_word_cloud(
        text_list, prompt_text, additional_stop_list, seed
    )
    ax.imshow(wordcloud_image)
    ax.axis("off")
    ax.grid(True)


def plot_revised_prompt(
    ax, revised_prompt_col, prompt_text, additional_stop_list, seed, farm_type
):
    """
    Display either a single revised prompt as text or a word cloud of multiple revised prompts on a given axis.

    Args:
        ax (matplotlib.axes._axes.Axes): The axis to display the revised prompt or word cloud.
        revised_prompt_col (pd.Series): A column of revised prompts. If only one unique prompt is present,
                                        it will be displayed as text; otherwise, a word cloud is generated.
        prompt_text (str): The original prompt text to exclude words from the word cloud.
        additional_stop_list (set): Additional words to exclude from the word cloud.
        seed (int, optional): Random seed for reproducibility when selecting sample images. Defaults to 7.
        farm_types (list): List of farm types to display.

    Returns:
        None
    """
    unique_list = revised_prompt_col.unique()
    if len(unique_list) == 1:  # if there is only one unique revised prompt
        plot_text(ax, unique_list[0], farm_type)
    else:  # if there are multiple, use word cloud
        revised_prompt_text = revised_prompt_col.tolist()
        plot_wordcloud(ax, revised_prompt_text, prompt_text, additional_stop_list, seed)


def plot_image(ax, image_path):
    """
    Display an image on an axis, or show placeholder text if the image is not found.

    Args:
        ax (matplotlib.axes._axes.Axes): The axis to display the image.
        image_path (Path): The path to the image file.
    """
    if image_path.exists():
        image = Image.open(image_path)
        ax.imshow(image)
        ax.axis("off")
    else:
        ax.text(0.5, 0.5, "Image not found", ha="center", va="center")
    ax.grid(True)


def plot_grid(
    megadata,
    generation_types,
    farm_types,
    title,
    additional_stop_list_dir,
    col_num=4,
    model="dall-e-3",
    seed=7,
):
    """
    Plot a grid with rows for each combination of generation type and farm type, and columns for
    various visualizations: prompt text, revised prompt word cloud, sample image, and GPT-4 description word cloud.

    Args:
        megadata (pd.DataFrame): The DataFrame containing the data.
        generation_types (list): List of generation types to display.
        title (str): Title for the entire plot.
        additional_stop_list_dir (dict): Dictionary of additional stopwords for each farm type.
        col_num (int, optional): Number of columns in the grid. Defaults to 4.
        model (str, optional): The model used for generating images. Defaults to 'dall-e-3'.
        seed (int, optional): Random seed for reproducibility when selecting sample images. Defaults to 7.
    """
    row_num = len(generation_types) * len(farm_types)

    # Create a figure with two grids, one for the header and one for content
    fig = plt.figure(figsize=(18, 15))
    fig.suptitle(title, fontsize=25, y=1)

    # Header row for column names
    gs_header = fig.add_gridspec(1, col_num, top=0.96, bottom=0.94)
    header_axes = [fig.add_subplot(gs_header[0, i]) for i in range(col_num)]

    # Content rows
    gs_content = fig.add_gridspec(
        row_num, col_num, top=0.94, bottom=0.03, hspace=0.03, wspace=0.05
    )
    content_axes = [
        [fig.add_subplot(gs_content[i, j]) for j in range(col_num)]
        for i in range(row_num)
    ]

    # Set column headers
    column_titles = ["Prompt", "Revised Prompt", "Example Image", "Description"]
    for idx, col_title in enumerate(column_titles):
        header_axes[idx].text(
            0.5, 0.5, col_title, ha="center", va="center", fontsize=20, weight="bold"
        )
        header_axes[idx].axis("off")

    # Plot content in the main grid
    for i, farm_type in enumerate(farm_types):
        for j, generation_type in enumerate(generation_types):
            row = i * len(generation_types) + j  # Row index for content_axes
            additional_stop_list = additional_stop_list_dir.get(farm_type, set())
            filtered_df = filter_data(
                megadata, [generation_type], [farm_type], model, country=None
            )

            if not filtered_df.empty:
                # Column 1: Prompt Text
                prompt_text = filtered_df["prompt"].values[0]
                plot_text(content_axes[row][0], prompt_text, farm_type)

                # Column 2: Revised Prompt Word Cloud
                revised_prompt_col = filtered_df["revised_prompt"].dropna()
                plot_revised_prompt(
                    content_axes[row][1],
                    revised_prompt_col,
                    prompt_text,
                    additional_stop_list,
                    seed,
                    farm_type,
                )

                # Column 3: Sample Image
                random.seed(seed)
                file_name = random.choice(filtered_df["file"].values)
                image_path = (
                    Path("..")
                    / "results"
                    / f"{model}-images"
                    / generation_type
                    / file_name
                )
                plot_image(content_axes[row][2], image_path)

                # Column 4: GPT-4 Description Word Cloud
                gpt4_description = filtered_df["GPT4o_description"].dropna().tolist()
                plot_wordcloud(
                    content_axes[row][3],
                    gpt4_description,
                    prompt_text,
                    additional_stop_list,
                    seed,
                )

    save_plt(plt, generation_types[0])  # save the plot
    plt.show()


def save_plt(plt, generation_type, farm_type=None):
    """
    Save a matplotlib plot as a PNG file in a specified directory, creating the directory if it does not exist.

    Args:
        plt (matplotlib.pyplot): The matplotlib plot to save.
        generation_type (str): The type of generation, used to create the file name.
        farm_type (str, optional): The type of farm, included in the file name if provided. Defaults to None.

    Saves:
        A PNG file of the plot in the "results/plots" directory, with a file name formatted based on the
        `generation_type` and `farm_type`. The file is saved at 300 dpi for high quality.
    """
    if farm_type is None:
        file_name = generation_type.split("_")[0] + "_plot_grid.png"
    else:
        file_name = (
            generation_type.split("_")[0]
            + "_"
            + farm_type
            + "_by_country_plot_grid.png"
        )

    img_dir = Path("..") / "results" / "plots"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)  # create the directory if the directory does not exist

    output_file = img_dir / file_name
    plt.savefig(output_file, format="png", dpi=300)  # Save plot with high resolution
