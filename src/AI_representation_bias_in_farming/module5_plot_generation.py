"""Library of functions used to generate grid plots based on image metadata
"""

import os
import re
import random
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud


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


def create_darker_ocean_colormap():
    """
    Creates a modified version of the 'ocean' colormap where the minimum blue shade
    is darker for better visibility. This preserves the beautiful ocean color progression
    while ensuring all text remains readable against a white background.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The modified ocean colormap
    """
    # Get the original ocean colormap colors
    ocean_colors = plt.cm.ocean(np.linspace(0.45, 1, 256))

    # Adjust the brightness range to ensure minimum darkness
    # We'll compress the brightness range to stay within darker values
    min_brightness = 0.1  # Minimum darkness level (0 is black, 1 is white)
    max_brightness = 0.7  # Maximum brightness level

    # Create new color array with adjusted brightness
    new_colors = ocean_colors.copy()
    for i in range(len(new_colors)):
        # Calculate the original position in the color range (0 to 1)
        position = i / (len(new_colors) - 1)
        # Adjust the brightness to stay within our desired range
        brightness = min_brightness + position * (max_brightness - min_brightness)
        # Apply the brightness adjustment while preserving the color's hue
        new_colors[i] = np.clip(ocean_colors[i] * brightness, 0, 1)

    # Create new colormap with adjusted colors
    darker_ocean = LinearSegmentedColormap.from_list("darker_ocean", new_colors)

    return darker_ocean


def generate_word_cloud(ngram_frequencies, seed=7):
    """
    Generate a word cloud image from a dictionary of word fre counts

    Args:
    ngram_frequencies (dictionary): a dictionary that summaries for each prompt, the freq of each word occurring
    seed (int): a random seed for reproducibility

    Returns:
        PIL.Image.Image: The generated word cloud image as a PIL image object.
    """

    # Generate the word cloud, excluding words in the prompt
    wordcloud = WordCloud(
        colormap=create_darker_ocean_colormap(),
        width=400,
        height=400,
        background_color="white",
        random_state=seed,
        min_font_size=10,
        max_font_size=70,
        prefer_horizontal=0.8,  # Allow some vertical text for better space usage
        relative_scaling=0.5,  # Adjust size based on frequency, but not too extremely
        collocations=False,  # Important: disable automatic collocation detection
    ).generate_from_frequencies(ngram_frequencies)

    return wordcloud.to_image()


def plot_text(ax, text, farm_type, country=None, max_character_per_line=23):
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

    if country is not None:
        # Split the country name into words, bold each word separately, and join them with spaces
        bold_country = " ".join([rf"$\mathbf{{{word}}}$" for word in country.split()])
        wrapped_text = wrapped_text.replace(country, bold_country)

    # Display wrapped text in the center of the axis
    ax.text(0.5, 0.5, wrapped_text, ha="center", va="center", fontsize=18, wrap=True)
    ax.axis("off")
    ax.grid(True)


def plot_wordcloud(ax, ngram_frequencies, seed):
    """
    Display a word cloud image on an axis.

    Args:
        ax (matplotlib.axes._axes.Axes): The axis to plot the word cloud.
        ngram_frequencies (dictionary): a dictionary recording freq of word occurring
        seed (int): random seed for reproducibility
    """
    wordcloud_image = generate_word_cloud(ngram_frequencies, seed)
    ax.imshow(wordcloud_image)
    ax.axis("off")
    ax.grid(True)


def plot_revised_prompt(
    ax,
    revised_prompt_col,
    ngram_frequencies,
    seed,
    farm_type,
    country=None,
):
    """
    Display either a single revised prompt as text or a word cloud of multiple revised prompts on a given axis.

    Args:
        ax (matplotlib.axes._axes.Axes): The axis to display the revised prompt or word cloud.
        revised_prompt_col (pd.Series): A column of revised prompts. If only one unique prompt is present,
                                        it will be displayed as text; otherwise, a word cloud is generated.
        ngram_frequencies (dictionary): a dictionary recording the freq of each word occurring
        seed (int, optional): Random seed for reproducibility when selecting sample images. Defaults to 7.
        farm_types (list): List of farm types to display.

    Returns:
        None
    """
    unique_list = revised_prompt_col.unique()
    if (len(unique_list)) <= 3:  # if there is only 1-3 unique prompts
        if country is not None:
            plot_text(ax, unique_list[0], farm_type, country)
        else:
            plot_text(ax, unique_list[0], farm_type)
    else:  # if there are multiple, use word cloud
        plot_wordcloud(ax, ngram_frequencies, seed)


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


def add_grey_to_no_revise_col(
    rows_to_highlight, fig, content_top, content_bottom, row_num
):
    """
    Adds light grey background rectangles to highlight specific rows in a figure.

    Args:
        rows_to_highlight (list): a list of index of rows to highlight
        fig (matplotlib.figure.Figure): The figure to add background rectangles to.
        content_top (float): The top boundary of the content area in figure coordinates.
        content_bottom (float): The bottom boundary of the content area in figure coordinates.
        row_num (int): The total number of rows in the content area.

    Returns:
        matplotlib.figure.Figure: The figure with added background rectangles on specified rows.
    """

    row_height = (
        content_top - content_bottom
    ) / row_num  # Calculate row height based on layout

    for row_idx in rows_to_highlight:
        # Calculate bottom position for each row based on row index
        bottom_position = content_top - (row_idx + 1) * row_height
        rect = patches.Rectangle(
            (0, bottom_position),
            1,
            row_height,  # Position and size of the rectangle
            transform=fig.transFigure,
            color="lightgrey",
            alpha=0.5,  # Adjust alpha for lighter grey
            zorder=0,
        )
        fig.add_artist(rect)

    return fig


def extract_word_frequencies(
    word_freq_summary,
    generation_type,
    farm_type,
    model,
    top_word_n_to_show=20,
    country=None,
):
    """
    Extracts word frequencies from a summary DataFrame for a specific combination of
    generation type, farm type, and model, converting the filtered data into a dictionary
    of word frequencies.

    This function performs three main steps:
    1. Filters the summary data for the specified conditions
    2. Removes metadata columns that aren't word frequencies
    3. Converts the remaining frequency data into a dictionary format

    Parameters:
    -----------
    word_freq_summary : pandas.DataFrame
        The input DataFrame containing word frequency summaries across different
        conditions. Should contain columns for metadata (generation_type, country, etc.)
        and word frequency counts.

    generation_type : str
        The specific generation type to filter for (e.g., 'basic').

    farm_type : str
        The type of farm to analyze (e.g., 'dairy', 'pig').

    model : str
        The model used for generation (e.g., 'dall-e-3').

    top_word_n_to_show : int
        The number used in the summary column name (e.g., if 20, will remove
        'top_20_words' column).

    country : str, optional
        The specific country to filter for.
        Default is None.

    Returns:
    --------
    dict
        A dictionary where keys are words/phrases and values are their frequencies.
        For example: {'green pasture': 15, 'sunny day': 10}

    """
    # First, filter the data for our specific conditions
    filtered_data = filter_data(
        word_freq_summary, [generation_type], [farm_type], model, country=country
    )

    # Define columns to remove (metadata and summary columns)
    columns_to_drop = [
        "generation_type",
        "country",
        "farm_type",
        "prompt",
        "model",
        f"top_{top_word_n_to_show}_words",
    ]

    # Remove these columns to leave only frequency data
    frequency_data = filtered_data.drop(columns=columns_to_drop)

    # Convert to dictionary format - we expect only one row
    # to_dict('records') returns a list of dictionaries, one per row
    # We take the first (and should be only) row with [0]
    word_frequencies = frequency_data.to_dict("records")[0]

    return word_frequencies


def plot_grid(
    megadata,
    generation_types,
    farm_types,
    title,
    revised_word_freq_summary,
    description_word_freq_summary,
    top_word_n_to_show=20,
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
        revised_word_freq_summary (dataframe): the dataframe recording the frequency of each word occurring in revised prompts
        description_word_freq_summary (dataframe): the dataframe recording the frequency of each word occurring in GPT4o descriptions
        top_word_n_to_show (int): default=20
            The number of most frequent words/phrases to include in the summary.
            For example, if set to 20, shows the 20 most frequently occurring terms.
        col_num (int, optional): Number of columns in the grid. Defaults to 4.
        model (str, optional): The model used for generating images. Defaults to 'dall-e-3'.
        seed (int, optional): Random seed for reproducibility when selecting sample images. Defaults to 7.
    """
    row_num = len(generation_types) * len(farm_types)
    content_top = 0.94  # top position of the plot grid in this figure
    content_bottom = 0.03  # bottom position of the plot grid in this figure

    # Create a figure with two grids, one for the header and one for content
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(title, fontsize=25, y=1)

    # Add background rectangles for the 2nd and 4th rows
    rows_to_highlight = [1, 3]  # Index of rows to highlight
    fig = add_grey_to_no_revise_col(
        rows_to_highlight, fig, content_top, content_bottom, row_num
    )

    # Header row for column names
    gs_header = fig.add_gridspec(1, col_num, top=0.96, bottom=content_top)
    header_axes = [fig.add_subplot(gs_header[0, i]) for i in range(col_num)]

    # Content rows
    gs_content = fig.add_gridspec(
        row_num,
        col_num,
        top=content_top,
        bottom=content_bottom,
        hspace=0.03,
        wspace=0.05,
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
            filtered_df = filter_data(
                megadata, [generation_type], [farm_type], model, country=None
            )

            # extract the ngram frequency count from revised prompts
            revised_ngram_frequencies = extract_word_frequencies(
                revised_word_freq_summary,
                generation_type,
                farm_type,
                model,
                top_word_n_to_show,
                country=None,
            )

            # extract the ngram frequency count from gpt4o descriptions
            description_ngram_frequencies = extract_word_frequencies(
                description_word_freq_summary,
                generation_type,
                farm_type,
                model,
                top_word_n_to_show,
                country=None,
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
                    revised_ngram_frequencies,
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
                    description_ngram_frequencies,
                    seed,
                )

    save_plt(plt, generation_types[0])  # save the plot
    plt.show()


def plot_grid_country(
    megadata,
    generation_types,
    farm_type,
    countries,
    title,
    revised_word_freq_summary,
    description_word_freq_summary,
    top_word_n_to_show=20,
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
        revised_word_freq_summary (dataframe): the dataframe recording the frequency of each word occurring in revised prompts
        description_word_freq_summary (dataframe): the dataframe recording the frequency of each word occurring in GPT4o descriptions
        top_word_n_to_show (int): default=20
            The number of most frequent words/phrases to include in the summary.
            For example, if set to 20, shows the 20 most frequently occurring terms.
        col_num (int, optional): Number of columns in the grid. Defaults to 4.
        model (str, optional): The model used for generating images. Defaults to 'dall-e-3'.
        seed (int, optional): Random seed for reproducibility when selecting sample images. Defaults to 7.
    """
    row_num = len(generation_types) * len(countries)
    content_top = 0.94  # top position of the plot grid in this figure
    content_bottom = 0.03  # bottom position of the plot grid in this figure

    # Create a figure with two grids, one for the header and one for content
    fig = plt.figure(figsize=(18, 25))
    fig.suptitle(title, fontsize=25, y=1)

    # Add background rectangles for the 2nd and 4th rows
    rows_to_highlight = [1, 3, 5]  # Index of rows to highlight
    fig = add_grey_to_no_revise_col(
        rows_to_highlight, fig, content_top, content_bottom, row_num
    )

    # Header row for column names
    gs_header = fig.add_gridspec(1, col_num, top=0.96, bottom=content_top)
    header_axes = [fig.add_subplot(gs_header[0, i]) for i in range(col_num)]

    # Content rows
    gs_content = fig.add_gridspec(
        row_num,
        col_num,
        top=content_top,
        bottom=content_bottom,
        hspace=0.03,
        wspace=0.05,
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
    for i, country in enumerate(countries):
        for j, generation_type in enumerate(generation_types):
            row = i * len(generation_types) + j  # Row index for content_axes
            filtered_df = filter_data(
                megadata, [generation_type], [farm_type], model, [country]
            )

            # extract the ngram frequency count from revised prompts
            revised_ngram_frequencies = extract_word_frequencies(
                revised_word_freq_summary,
                generation_type,
                farm_type,
                model,
                top_word_n_to_show,
                country=country,
            )

            # extract the ngram frequency count from gpt4o descriptions
            description_ngram_frequencies = extract_word_frequencies(
                description_word_freq_summary,
                generation_type,
                farm_type,
                model,
                top_word_n_to_show,
                country=country,
            )

            if not filtered_df.empty:
                # Column 1: Prompt Text
                prompt_text = filtered_df["prompt"].values[0]
                plot_text(content_axes[row][0], prompt_text, farm_type, country)

                # Column 2: Revised Prompt Word Cloud
                revised_prompt_col = filtered_df["revised_prompt"].dropna()
                plot_revised_prompt(
                    content_axes[row][1],
                    revised_prompt_col,
                    revised_ngram_frequencies,
                    seed,
                    farm_type,
                    country,
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
                plot_wordcloud(
                    content_axes[row][3],
                    description_ngram_frequencies,
                    seed,
                )

    save_plt(plt, generation_types[0], farm_type)  # save the plot
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


# import the megadata dataframe, and sort by the length of revised prompts
import pandas as pd
megadata_file = Path(".") / "results" / "megadata" / "image_megadata.csv"
megadata = pd.read_csv(megadata_file, header=0)
seed = 707
model = "dall-e-3"
words_to_exclude_1gram = [
    "cows",
    "cow",
    "farm",
    "farms",
    "pig",
    "pigs",
    "dairy",
    "scene",
    "background",
    "picture",
    "show",
    "shows",
    "depicts",
    "depiction",
    "depicting",
    "image",
    "typical",
    "representation",
    "representing",
    "showcase",
    "showcasing",
    "include",
    "including",
    "elements",
    "united",
    "states",
    "america",
    "american",
    "germany",
    "german",
    "new",
    "zealand",
    "spain",
    "spanish",
    "australia",
    "australian",
    "seen",
    "nearby",
    "overall",
    "atmosphere",
    "area", 
    "foreground",
    "setting",
    "settings",
    "detailed",
    "suggesting",
    "indicating",
    "country", 
    "size", 
    "model",
    "file",
    "prompt",
    "quality"
]
words_to_exclude_2_3gram = [
    "dairy cows",
    "dairy cow",
    "dairy farm",
    "dairy farms",
    "pig farms",
    "pig farm",
    "typical dairy",
    "typical dairy farm",
    "typical dairy farms",
    "typical pig",
    "typical pig farm",
    "typical pig farms",
    "image typical",
    "image typical dairy",
    "image typical pig",
    "image shows",
    "representation typical",
    "representation typical dairy",
    "representation typical pig",
    "depiction typical",
    "depiction typical dairy",
    "depiction typical pig",
    "depicting typical",
    "depict detailed",
    "depicts detailed",
    "overall atmosphere",
    "farm scene",
    "farm setting",
    "pig farm scene",
    "dairy farm scence",
    "realistic depiction",
    "realistic depiction typical",
    "accurate representation",
    "accurate representation typical",
    "realistic image",
    "realistic representation",
    "realistic representation typical",
    "accurate depiction",
    "image depicts",
    "image features",
    "generate image",
    "create image",
    "scene include",
    "scene depicting",
    "farm scene includes",
    "united states",
    "new zealand",
    "dairy farm united",
    "dairy farms united",
    "pig farm united",
    "pig farms united",
    "farm united",
    "farms united",
    "farm united states",
    "farms united states",
    "states scene",
    "united states image",
    "united states scene",
    "dairy farm germany",
    "dairy farms germany",
    "farm germany",
    "farms germany",
    "germany scene",
    "farm germany scene",
    "farms germany scene",
    "farm new",
    "farm new zealand",
    "dairy farm new",
    "dairy farms new",
    "new zealand scene",
    "zealand scene",
    "farm spain",
    "farms spain",
    "pig farm spain",
    "pig farms spain",
    "spain scene",
    "farm spain scene",
    "farms spain scene",
    "farm australia",
    "farms australia",
    "pig farm australia",
    "pig farms australia",
    "australia scene",
    "farm australia scene",
    "farms australia scene",
    "typical australian",
    "fluffy white",
]
words_to_exclude = words_to_exclude_1gram + words_to_exclude_2_3gram

# set number of columns in the grid of plots, words that we want to remove from wordclowd
num_cols = 4
top_word_n_to_show = 20
countries_by_farm_type = {
        "dairy": ["the United States", "Germany", "New Zealand"],
        "pig": ["the United States", "Spain", "Australia"],
    }  # list of 
# Define generation types and farm types
gen_types = ["reality_country", "reality_country_no_revise"]
farm_type = megadata['farm_type'].unique()[0]
countries = countries_by_farm_type[farm_type]
title = f"Prompt DALL·E 3 for realistic {farm_type} farms at 3 different countries"

plot_grid_country(
    megadata,
    generation_types = gen_types,
    farm_type,
    countries,
    title,
    revised_word_freq_summary = revised_prompt_2gram_summary,
    description_word_freq_summary=gpt4o_description_2gram_summary,
    top_word_n_to_show,
    col_num = num_cols, 
    model = model, 
    seed = seed
)

def plot_grid_country(
    megadata,
    generation_types,
    farm_type,
    countries,
    title,
    revised_word_freq_summary,
    description_word_freq_summary,
    top_word_n_to_show=20,
    col_num=4,
    model="dall-e-3",
    seed=7,
):