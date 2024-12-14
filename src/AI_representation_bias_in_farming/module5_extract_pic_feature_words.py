"""Library of functions used to see which words are the key features for describing images of 
   outdoor VS indoor farms
"""

from pathlib import Path

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random


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


def cluster_farm_images(df, source_base, dest_base):
    """
    Filter images based on the cluster GPT-4o assiigned to them ("outdoor", "indoor" or "other")
    and organize them into folders.

    Parameters
    ----------
    df (pandas.DataFrame):
        DataFrame with image metadata, must include 'farm_type' and 'model' columns
    source_base (Path):
        Base path for source images
    dest_base (Path):
        Base path for destination folders

    Returns
    -------
    None
    """
    condition_types = df["GPT4o_cluster"].unique()
    farm_types = df["farm_type"].unique()
    models = df["model"].unique()

    for model in models:
        for farm_type in farm_types:
            for condition_type in condition_types:
                dest_folder = dest_base / model / farm_type / condition_type

                # Create destination folder if it doesn't exist
                if not dest_folder.exists():
                    dest_folder.mkdir(parents=True)

                # Filter rows based on model, farm_type and the cluster GPT4o assigns
                filtered_rows = df[
                    (df["GPT4o_cluster"] == condition_type)
                    & (df["farm_type"] == farm_type)
                    & (df["model"] == model)
                ]

                # put the images belonging to this condition into the same folder
                image_copy_and_paste_all_rows(
                    filtered_rows, source_base, dest_folder, model
                )


def summarize_clusters(df, output_file="cluster_summary.csv"):
    """
    Calculate summary statistics for indoor and outdoor images grouped by specific columns.

    Parameters
    ----------
    df (pandas.DataFrame): Input DataFrame containing farm data with 'GPT4o_cluster' labeled
                            as either 'indoor', 'outdoor', or 'other'
    output_file (str): the name of the output file

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
    - total sum : int
        Total number of rows for each unique combination of generation_type, country, farm_typem, model
    - indoor_sum : int
        total number of images showing animals kept indoor
    - outdoor_sum : int
        total number of images showing animals kept outdoor
    - other_sum : int
        total number of images not classified as either indoor or outdoor
    - indoor_pct : float
        Percentage of images showing animals kept indoor
    - outdoor_pct : float
        Percentage of images showing animals kept outdoor
    - other_pct : float
        Percentage of images not classified

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
        indoor_sum = len(matching_rows[matching_rows["GPT4o_cluster"] == "indoor"])
        outdoor_sum = len(matching_rows[matching_rows["GPT4o_cluster"] == "outdoor"])
        other_sum = len(matching_rows[matching_rows["GPT4o_cluster"] == "other"])
        indoor_pct = round((indoor_sum / total_rows), 2)
        outdoor_pct = round((outdoor_sum / total_rows), 2)
        other_pct = round((other_sum / total_rows), 2)

        # Create a dictionary with all the information
        row_data = {
            "generation_type": combo["generation_type"],
            "country": combo["country"],
            "farm_type": combo["farm_type"],
            "prompt": combo["prompt"],
            "model": combo["model"],
            "total_rows": total_rows,
            "indoor_sum": indoor_sum,
            "outdoor_sum": outdoor_sum,
            "other_sum": other_sum,
            "indoor_pct": indoor_pct,
            "outdoor_pct": outdoor_pct,
            "other_pct": other_pct,
        }

        # Add this combination's data to our summary list
        summary_data.append(row_data)

    # Create DataFrame from our summary data
    summary_df = pd.DataFrame(summary_data)

    output_file = Path("..") / "results" / "megadata" / output_file
    summary_df.to_csv(output_file, index=False)

    return summary_df


# Function to group generation types
def get_group(gen_type):
    """
    Groups generation types based on their naming pattern.

    Parameters
    ----------
    gen_type (str): The generation type string to be grouped (e.g. 'base_country', 'base')

    Returns
    -------
    tuple
    A tuple containing:
    - group_name_major (str): The primary grouping name, taken from the first word
    - group_name_minor (str): The secondary grouping name. Same as major group
        unless '_country' is present, then includes 'country' suffix

    """
    word_list = gen_type.split("_")
    word1 = word_list[0]
    group_name_major = word1
    group_name_minor = word1
    if len(word_list) > 1:
        word2 = word_list[1]

        if word2 == "country":
            group_name_minor = word1 + " " + word2

    return group_name_major, group_name_minor


def wrap_labels(text, width=20):
    """
    Wraps long text labels into multiple lines for better readability.

    Parameters
    ----------
    text (str): The text string to be wrapped
    width (int), optional: Maximum width of each line in characters (default: 20)

    Returns
    -------
    str: The input text wrapped to the specified width, preserving whole words


    """

    import textwrap

    return textwrap.fill(text, width=width, break_long_words=False)


def save_plot_grid(
    summary_df,
    output_path,
    megadata_df,
    source_base,
    model,
    num_images=10,
    images_per_row=5,
    random_seed=7,
):
    """
    Create and save the plot grid.

    Parameters
    ----------
    summary_df (pandas.DataFrame): DataFrame containing the summary data
    output_path (str or Path): Path where the plot should be saved
    megadata_df (pandas.DataFrame):
        DataFrame containing image metadata and cluster labels
    source_base (Path):
        Base path for image files
    model (str):
        Model name to use for plotting
    num_images (int, optional):
        Number of images to display in each grid (default: 10)
    random_seed (int, optional):
        Seed for random number generation. If None, no seed is set.
    """
    fig = create_plot_grid(
        summary_df,
        megadata_df,
        source_base,
        model,
        num_images,
        images_per_row,
        random_seed,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return fig


def create_bar_plot(ax, plot_data, farm_type, model):
    """
    Create a single bar plot showing indoor and outdoor percentages.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    plot_data : pandas.DataFrame
        Data for this specific plot
    farm_type : str
        Type of farm for the title
    model : str
        Model name for the title
    """
    # Define colors
    indoor_color = "#1f77b4"  # blue
    outdoor_color = "#2ecc71"  # green
    width = 0.6

    # Create x positions with increased spacing
    x = np.arange(len(plot_data)) * 1.2

    # Create twin axis for outdoor percentages (bottom)
    ax_bottom = ax
    ax_bottom.bar(
        x,
        (plot_data["outdoor_pct"] * 100),
        width=width,
        color=outdoor_color,
        alpha=0.7,
        label="On pasture/mud %",
    )
    ax_bottom.set_ylim(0, 100)
    ax_bottom.set_ylabel("")
    ax_bottom.tick_params(axis="y", labelsize=30)

    # Plot outdoor percentages (top)
    ax_top = ax_bottom.twinx()
    ax_top.bar(
        x,
        -(plot_data["indoor_pct"] * 100),
        width=width,
        color=indoor_color,
        alpha=0.7,
        label="Exclusively indoor %",
    )
    ax_top.set_ylim(-100, 0)
    ax_top.set_ylabel("")
    ax_top.tick_params(axis="y", labelsize=30)
    ax_top.yaxis.set_major_formatter(lambda x, pos: str(int(abs(x))))

    # Set title
    ax_bottom.set_title(
        f"{farm_type.capitalize()} farms generated by {model}",
        pad=35,
        fontsize=27,
        fontweight="bold",
    )

    return ax_top, ax_bottom, x


def format_bar_plot_labels(ax_bottom, ax_top, x, plot_data):
    """
    Format the labels and styling of a bar plot.
    """
    # Create tick labels with different colors and weights
    tick_labels = []
    tick_colors = []
    tick_weights = []

    for label in plot_data["x_axis_label"]:
        wrapped = wrap_labels(label, width=25)
        tick_labels.append(wrapped)

        if "no revise" in label.lower():
            tick_colors.append("#FF6F1E")
            tick_weights.append("bold")
        else:
            tick_colors.append("black")
            tick_weights.append("normal")

    # Set and format tick labels
    ax_bottom.set_xticks(x)
    text_objects = ax_bottom.set_xticklabels(
        tick_labels, rotation=90, ha="center", va="top", fontsize=24
    )

    # Apply colors and font weights
    for text_obj, color, weight in zip(text_objects, tick_colors, tick_weights):
        text_obj.set_color(color)
        text_obj.set_fontweight(weight)

    # Add gridlines and group separators
    ax_top.grid(True, alpha=0.3)
    prev_group = None
    for idx, group in enumerate(plot_data["group"]):
        if group != prev_group:
            ax_top.axvline(
                x=x[idx] - 0.75,
                color="gray",
                linestyle="--",
                linewidth=5,
                alpha=0.7,
            )

            ax_top.text(
                x[idx],
                ax_top.get_ylim()[1] * 1.1,
                group.capitalize(),
                horizontalalignment="left",
                verticalalignment="bottom",
                fontsize=24,
                fontweight="bold",
            )
        prev_group = group

    # Add legend with increased font size
    lines_top, labels_top = ax_top.get_legend_handles_labels()
    lines_bottom, labels_bottom = ax_bottom.get_legend_handles_labels()
    ax_top.legend(
        lines_top + lines_bottom,
        labels_top + labels_bottom,
        bbox_to_anchor=(1.05, 1.0),
        loc="lower right",
        fontsize=22,
        frameon=True,
        framealpha=0.4,
        title_fontsize=24,
    )


def plot_image_grid(ax, images, title, num_images=10, images_per_row=5):
    """
    Plot a grid of images with minimal horizontal spacing while maintaining square aspect ratio.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    images : list
        List of image file paths
    title : str
        Title for the plot
    num_images : int
        Number of images to display (default: 10)
    images_per_row : int
        Number of images per row (default: 5)
    """
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])

    # Set title
    ax.set_title(title, fontsize=27, fontweight="bold", pad=15)

    # Calculate grid dimensions
    num_rows = (num_images + images_per_row - 1) // images_per_row

    # Calculate the spacing parameters
    gap = 0.01
    total_gaps = images_per_row - 1
    total_gap_space = gap * total_gaps

    # Calculate image width accounting for gaps
    image_width = (1 - total_gap_space) / images_per_row

    for idx, img_path in enumerate(images[:num_images]):
        row = idx // images_per_row
        col = idx % images_per_row

        # Calculate x position with minimal gaps
        x_position = col * (image_width + gap)

        # Calculate y position (unchanged)
        y_position = 1 - (row + 1) / num_rows

        # Add subplot for each image
        img = plt.imread(img_path)
        ax_sub = ax.inset_axes([x_position, y_position, image_width, 1 / num_rows])
        ax_sub.imshow(img)
        ax_sub.axis("off")


def create_plot_grid(
    summary_df,
    megadata_df,
    source_base,
    model,
    num_images=10,
    images_per_row=5,
    random_seed=7,
):
    """
    Create a grid with bar plots on the left (2 rows) and image grids on the right (3 rows).
    Images are split between dairy and pig farms in each row.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        DataFrame containing summary statistics
    megadata_df : pandas.DataFrame
        DataFrame containing image metadata and cluster labels
    source_base : Path
        Base path for image files
    model : str
        Model name to use for plotting
    num_images : int
        Number of images to display in each grid (default: 8)
    random_seed : int, optional
        Seed for random number generation. If None, no seed is set.

    Returns
    -------
    matplotlib.figure.Figure
        The complete figure containing the plot grid
    """
    # Filter to be only about this one model
    summary_df = summary_df[summary_df["model"] == model].copy()
    megadata_df = megadata_df[megadata_df["model"] == model].copy()

    # Set random seeds if specified
    if random_seed is not None:
        set_random_seeds(random_seed)

    # Create figure with adjusted dimensions
    fig = plt.figure(figsize=(40, 40))  # Increased height for better spacing

    # Create a complex gridspec layout with increased spacing
    main_gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.08)

    # Create sub-gridspecs with increased spacing
    left_gs = main_gs[0].subgridspec(
        2, 1, height_ratios=[1, 1], hspace=0.5 # Increased spacing between plots
    )
    # Modify the right gridspec to fill the available space
    right_gs = main_gs[1].subgridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.1)
    # Create axes for left column (bar plots)
    ax_dairy = fig.add_subplot(left_gs[0])
    ax_pig = fig.add_subplot(left_gs[1])

    # Create axes for right column (image grids)
    ax_outdoor = fig.add_subplot(right_gs[0])
    ax_indoor = fig.add_subplot(right_gs[1])
    ax_other = fig.add_subplot(right_gs[2])
    
    # Add subplot labels (A), (B), (C), (D)
    label_coords = [
        (0.1, 0.9, "(A)"),  # top left
        (0.1, 0.43, "(B)"),  # bottom left
        (0.52, 0.9, "(C)"),  # top right
        (0.52, 0.63, "(D)"),  # bottom right
        (0.52, 0.36, "(E)"),  # bottom right
    ]

    for x, y, label in label_coords:
        fig.text(x, y, label, fontsize=27, fontweight="bold")

    # Remove unnecessary grid
    for i, ax in enumerate([ax_outdoor, ax_indoor, ax_other]):

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Process data for bar plots
    summary_df["model"] = summary_df["model"].apply(
        lambda x: "DALL-E-3" if "dall-e-3" in x else x
    )

    # Create bar plots for dairy and pig farms
    for ax, farm_type in [(ax_dairy, "dairy"), (ax_pig, "pig")]:
        # Filter data
        plot_data = summary_df[(summary_df["farm_type"] == farm_type)].copy()

        # Add grouping
        plot_data["group"], plot_data["group_minor"] = zip(
            *plot_data["generation_type"].apply(get_group)
        )
        plot_data["no_revise"] = plot_data["generation_type"].apply(
            lambda x: " no revise" if "no_revise" in x else ""
        )
        plot_data["x_axis_label"] = (
            plot_data["group"]
            + " "
            + plot_data["country"].fillna("").str.replace("the", "")
            + plot_data["no_revise"]
        )

        # Sort data
        plot_data = plot_data.sort_values(
            ["group", "group_minor", "country", "generation_type"]
        )

        # Create and format bar plot
        ax_top, ax_bottom, x = create_bar_plot(ax, plot_data, farm_type, model)
        format_bar_plot_labels(ax_bottom, ax_top, x, plot_data)

    # Process image data for each cluster, split by farm type
    image_paths = {"outdoor": [], "indoor": [], "other": []}

    # Get image paths for each cluster
    for cluster in ["outdoor", "indoor", "other"]:
        # Process dairy farms (first half)
        dairy_data = megadata_df[
            (megadata_df["GPT4o_cluster"] == cluster)
            & (megadata_df["farm_type"] == "dairy")
        ]
        selected_dairy = dairy_data.sample(
            n=min(num_images // 2, len(dairy_data)),
            random_state=random_seed if random_seed is not None else None,
        )

        # Process pig farms (second half)
        pig_data = megadata_df[
            (megadata_df["GPT4o_cluster"] == cluster)
            & (megadata_df["farm_type"] == "pig")
        ]
        selected_pig = pig_data.sample(
            n=min(num_images // 2, len(pig_data)),
            random_state=random_seed if random_seed is not None else None,
        )

        # Combine paths
        for _, row in pd.concat([selected_dairy, selected_pig]).iterrows():
            source_file = (
                source_base / (model + "-images") / row["generation_type"] / row["file"]
            )
            image_paths[cluster].append(source_file)

    # Create image grids without grid lines
    plot_image_grid(
        ax_outdoor,
        image_paths["outdoor"],
        "Pasture or mud (Top: Dairy, Bottom: Pig)",
        num_images,
        images_per_row,
    )
    plot_image_grid(
        ax_indoor,
        image_paths["indoor"],
        "Exclusively indoor (Top: Dairy, Bottom: Pig)",
        num_images,
        images_per_row,
    )
    plot_image_grid(
        ax_other,
        image_paths["other"],
        "Other (Top: Dairy, Bottom: Pig)",
        num_images,
        images_per_row,
    )

    return fig


def set_random_seeds(seed):
    """
    Set random seeds for both Python's random module and NumPy to ensure reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to use for random number generation
    """
    random.seed(seed)
    np.random.seed(seed)