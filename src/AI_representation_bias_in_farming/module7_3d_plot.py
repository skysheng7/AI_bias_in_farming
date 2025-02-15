"""
Generate 3d plots to summarize image clustering results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from AI_representation_bias_in_farming import module5_extract_pic_feature_words


def add_grouping(df):
    """
    Add grouping columns to the DataFrame based on generation type.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a 'generation_type' column

    Returns
    -------
    pandas.DataFrame
        A copy of the input DataFrame with three new columns:
        - major_group: Major group extracted from generation type
        - minor_group: Minor group extracted from generation type
        - revise_status: 'no revise' or 'revise' based on generation type

    Notes
    -----
    The function creates a copy of the input DataFrame and does not modify
    the original. It uses module5_extract_pic_feature_words.get_group() to
    extract major and minor groups.
    """
    filtered_df = df.copy()
    # Get major and minor groups for each generation type
    filtered_df["major_group"], filtered_df["minor_group"] = zip(
        *filtered_df["generation_type"].apply(
            module5_extract_pic_feature_words.get_group
        )
    )

    # Create revise/no_revise category
    filtered_df["revise_status"] = filtered_df["generation_type"].apply(
        lambda x: "no revise" if "no_revise" in x else "revise"
    )

    return filtered_df


def create_combined_farm_plot(
    cluster_df,
    filtered_mega,
    countries_by_farm_type=None,
    example_img_num=3,
    random_seed=11,
):
    """
    Create a combined visualization with 3D bar plots and image grids for both indoor and outdoor metrics.

    Parameters
    ----------
    cluster_df : pandas.DataFrame
        DataFrame containing the clustering data with columns:
        - generation_type, indoor_pct, outdoor_pct, indoor_ci_lower, indoor_ci_upper,
          outdoor_ci_lower, outdoor_ci_upper
    filtered_mega: pandas.DataFrame
        DataFrame containing the raw image megadata with columns:
        - generation_type, farm_type, GPT4o_cluster
    countries_by_farm_type : dict, default to None
        Dictionary mapping farm types to their top countries by livestock population.
        Contains lists of countries with the highest number of dairy cows and pigs
        in North America, Europe, and Oceania.
        Format:
        {
            "dairy": list of str,  # Top countries for dairy cow population
            "pig": list of str     # Top countries for pig population
        }
        Example:
        {
            "dairy": ["the United States", "Germany", "New Zealand"],
            "pig": ["the United States", "Spain", "Australia"]
        }
    example_img_num: int, default is 3
        How many example images you wish to show for each category
    random_seed: int default is 11
        random seed for reproducibility

    Returns
    -------
    fig : matplotlib.figure.Figure
        The complete figure containing all subplots
    """
    # list all the farm types: dairy and pig
    farm_types = cluster_df["farm_type"].unique()
    metrics = ["outdoor", "indoor"]

    # Create figure with custom layout
    plt.style.use("classic")
    fig = plt.figure(figsize=(34, 30))
    gs = gridspec.GridSpec(
        len(farm_types),
        4,
        width_ratios=[1.5, 0.5, 1.5, 0.5],
        height_ratios=[1] * len(farm_types),
    )

    # plot each row based on farm type
    for row_idx, farm_type in enumerate(farm_types):
        filtered_df = cluster_df[cluster_df["farm_type"] == farm_type]

        for col_idx, metric in enumerate(metrics):
            # Plot outdoor or indoor metrics (top row)
            ax1 = fig.add_subplot(gs[row_idx, (2 * col_idx)], projection="3d")

            # if this is a plot grid by country:
            if countries_by_farm_type is not None:
                countries = countries_by_farm_type[farm_type]
                plot_3d_farm_by_country(
                    filtered_df, countries, metric="indoor", ax=None
                )
            else:
                plot_3d_generation_types(filtered_df, metric=metric, ax=ax1)

            # Add subplot labels (A, B, C, D)
            label = chr(65 + (2 * row_idx + col_idx))  # 65 is ASCII for 'A'
            ax1.text2D(
                0,
                1,
                f"({label})",
                transform=ax1.transAxes,
                fontsize=35,
                fontweight="bold",
            )

            if row_idx == 0:
                if metric == "outdoor":
                    ax1.text2D(
                        0.65,
                        1.15,
                        "Have access to pasture/mud outdoors",
                        transform=ax1.transAxes,
                        fontsize=32,
                        fontweight="bold",
                        horizontalalignment="center",
                    )
                else:
                    ax1.text2D(
                        0.65,
                        1.15,
                        "Exclusively indoors",
                        transform=ax1.transAxes,
                        fontsize=32,
                        fontweight="bold",
                        horizontalalignment="center",
                    )

            # Add farm type labels on the y-axis for the first column of each row
            if col_idx == 0:
                ax1.text2D(
                    -0.1,
                    0.5,
                    f"{farm_type.capitalize()} farm",
                    transform=ax1.transAxes,
                    fontsize=32,
                    fontweight="bold",
                    rotation=90,
                    verticalalignment="center",
                )

            # Add 3 example images
            if countries_by_farm_type is None:  # if this is not a plot by country
                selected_images = image_random_select(
                    filtered_mega, farm_type, metric, example_img_num, random_seed
                )
            else:  # if this is a plot by country
                selected_images = image_random_select_country(
                    filtered_mega,
                    farm_type,
                    metric,
                    countries,
                    example_img_num,
                    random_seed,
                )

            ax2 = fig.add_subplot(gs[row_idx, 2 * col_idx + 1])
            plot_3example_images(ax2, selected_images, "", num_images=example_img_num)

    # Adjust layout
    plt.subplots_adjust(wspace=0.00001, hspace=0.01)

    return fig


def image_random_select(
    filtered_mega, farm_type, metric, example_img_num=3, random_seed=11
):
    """
    Randomly select a specified number of image paths based on farm type and metric criteria.

    Parameters
    ----------
    filtered_mega : pandas.DataFrame
        DataFrame containing image metadata with columns:
        - farm_type: type of farm
        - GPT4o_cluster: metric cluster label
        - model: model name
        - generation_type: type of generation
        - file: image filename
    farm_type : str
        Type of farm to filter by (e.g., 'dairy', 'pig')
    metric : str
        Metric to filter by (e.g., 'indoor', 'outdoor')
    example_img_num : int, defualt 3
        Number of images to randomly select
    random_seed : int or None, default 11
        Random seed for reproducible sampling. If None, sampling is random

    Returns
    -------
    list
        List of Path objects pointing to the selected image files

    Notes
    -----
    Images are selected from the 'results/{model}-images/{generation_type}/' directory
    """
    cur_megadata = filtered_mega[
        (filtered_mega["farm_type"] == farm_type)
        & (filtered_mega["GPT4o_cluster"] == metric)
    ]

    # randomly select a few images
    selected_rows = cur_megadata.sample(
        n=example_img_num, random_state=random_seed if random_seed is not None else None
    )
    selected_images = []
    for _, row in selected_rows.iterrows():
        new_image = (
            Path()
            / "results"
            / (row["model"] + "-images")
            / row["generation_type"]
            / row["file"]
        )
        selected_images.append(new_image)

    return selected_images


def plot_3d_generation_types(df, metric="indoor", ax=None):
    """
    Create a 3D bar plot showing generation types with confidence intervals.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data
    metric : str, optional (default='indoor')
        Which metric to plot. Either 'indoor' or 'outdoor'
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure is created

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot
    """
    if ax is None:
        plt.style.use("classic")
        fig = plt.figure(figsize=(16, 20))
        ax = fig.add_subplot(111, projection="3d")

    # Set colors based on metric
    colors = "lightskyblue" if metric == "indoor" else "yellowgreen"

    # Setup bar positions
    major_groups = ["basic", "typical", "reality"]
    revise_status = ["no revise", "revise"]
    width = depth = 0.5

    # Style settings
    ax.grid(False)  # Remove grid
    ax.xaxis._axinfo["grid"].update({"color": (1, 1, 1, 0)})  # Remove grid lines
    ax.yaxis._axinfo["grid"].update({"color": (1, 1, 1, 0)})
    ax.zaxis._axinfo["grid"].update({"color": (1, 1, 1, 0)})
    ax.set_box_aspect([1, 1, 1])

    # Plot bars and confidence intervals
    for i, major in enumerate(major_groups):
        for j, rev in enumerate(revise_status):
            mask = (df["major_group"] == major) & (df["revise_status"] == rev)
            if any(mask):
                value = df.loc[mask, f"{metric}_pct"].values[0] * 100
                ci_lower = df.loc[mask, f"{metric}_ci_lower"].values[0] * 100
                ci_upper = df.loc[mask, f"{metric}_ci_upper"].values[0] * 100

                if j == 1:
                    j_position = j + 1

                # Add bar
                ax.bar3d(
                    j_position,
                    i,
                    0,
                    width,
                    depth,
                    value,
                    color=colors,
                    shade=True,
                    alpha=0.8,
                    zsort="max",
                    edgecolor="none",
                )

                # Add confidence interval
                x_center = j_position + width / 2
                y_center = i + depth / 2
                ax.plot(
                    [x_center, x_center],
                    [y_center, y_center],
                    [value, ci_upper],
                    color="orange",
                    linewidth=5,
                    zorder=100,
                )

                # Add CI cap
                ci_width = 0.1
                ax.plot(
                    [x_center - ci_width, x_center + ci_width],
                    [y_center, y_center],
                    [ci_upper, ci_upper],
                    color="orange",
                    linewidth=5,
                    zorder=100,
                )

    # Customize axes
    ax.set_xticks([0 + (width / 2), 2 + (width / 2)])
    ax.set_yticks([0 + (width / 2), 1 + (width / 2), 2 + (width / 2)])
    ax.set_xticklabels(revise_status, fontsize=32)
    ax.set_yticklabels(major_groups, fontsize=32)
    # Then adjust the padding
    ax.set_zlabel("Percentage", fontsize=32, labelpad=20)
    ax.tick_params(axis="z", labelsize=24, pad=10)

    # Set view angle and limits
    ax.view_init(elev=20, azim=60)
    ax.set_zlim(0, 100)
    ax.set_xlim(0, 2.8)
    ax.set_ylim(0, 2.6)

    return ax


def plot_3example_images(ax, image_paths, title, num_images=3):
    """
    Plot a vertical grid of 3 randomly sampled images.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    image_paths : list
        List of image file paths to sample from
    title : str
        Title for the plot
    num_images : int, optional (default=3)
        Number of images to display vertically
    """
    # Remove axes and spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Set title
    ax.set_title(title, fontsize=27, fontweight="bold", pad=15)

    # Calculate spacing
    gap = 0.001  # Gap between images
    image_height = (1 - (gap * (num_images - 1))) / num_images

    # Plot images vertically
    for idx, img_path in enumerate(image_paths):
        # Calculate vertical position (starting from top)
        y_position = 1 - (idx + 1) * (image_height + gap)

        # Add subplot
        ax_sub = ax.inset_axes([0, y_position, 1, image_height])
        img = plt.imread(img_path)
        ax_sub.imshow(img)
        ax_sub.axis("off")


def image_random_select_country(
    filtered_mega, farm_type, metric, countries, example_img_num=3, random_seed=11
):
    """
    Randomly select a specified number of image paths based on farm type and metric criteria.

    Parameters
    ----------
    filtered_mega : pandas.DataFrame
        DataFrame containing image metadata with columns:
        - farm_type: type of farm
        - GPT4o_cluster: metric cluster label
        - model: model name
        - generation_type: type of generation
        - file: image filename
    farm_type : str
        Type of farm to filter by (e.g., 'dairy', 'pig')
    metric : str
        Metric to filter by (e.g., 'indoor', 'outdoor')
    countries : list of str
        List of target countries to analyze, representing major livestock producers
        across North America, Europe, and Oceania.
    example_img_num : int, defualt 3
        Number of images to randomly select
    random_seed : int or None, default 11
        Random seed for reproducible sampling. If None, sampling is random

    Returns
    -------
    list
        List of Path objects pointing to the selected image files

    Notes
    -----
    Images are selected from the 'results/{model}-images/{generation_type}/' directory
    """
    selected_images = []

    cur_megadata = filtered_mega[
        (filtered_mega["farm_type"] == farm_type)
        & (filtered_mega["GPT4o_cluster"] == metric)
    ]

    for country in countries:
        cur_megadata_country = cur_megadata[cur_megadata[country] == country]

        # randomly select a few images
        selected_rows = cur_megadata.sample(
            n=1, random_state=random_seed if random_seed is not None else None
        )
        row = selected_rows[0]
        new_image = (
            Path()
            / "results"
            / (row["model"] + "-images")
            / row["generation_type"]
            / row["file"]
        )
        selected_images.append(new_image)

    return selected_images


def plot_3d_farm_by_country(dataframe, countries, metric="indoor", ax=None):
    """
    Create a 3D bar plot showing generation types with confidence intervals.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing the data
    countries : list of str
        List of target countries to analyze, representing major livestock producers
        across North America, Europe, and Oceania.
    metric : str, optional (default='indoor')
        Which metric to plot. Either 'indoor' or 'outdoor'
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure is created

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot
    """
    df = dataframe.copy()

    df["major_group_and_revise_status"] = np.where(
        df["revise_status"] == "no revise",
        df["major_group"] + " " + df["revise_status"],
        df["major_group"],
    )

    if ax is None:
        plt.style.use("classic")
        fig = plt.figure(figsize=(16, 20))
        ax = fig.add_subplot(111, projection="3d")

    # Set colors based on metric
    colors = "lightskyblue" if metric == "indoor" else "yellowgreen"

    # Setup bar positions
    major_group_and_revise_status = [
        "basic",
        "basic no revise",
        "typical",
        "typical no revise",
        "reality",
        "reality no revise",
    ]
    width = depth = 0.5

    # Style settings
    ax.grid(False)  # Remove grid
    ax.xaxis._axinfo["grid"].update({"color": (1, 1, 1, 0)})  # Remove grid lines
    ax.yaxis._axinfo["grid"].update({"color": (1, 1, 1, 0)})
    ax.zaxis._axinfo["grid"].update({"color": (1, 1, 1, 0)})
    ax.set_box_aspect([1, 1, 1])

    # Plot bars and confidence intervals
    for i, country in enumerate(countries):
        for j, status in enumerate(major_group_and_revise_status):
            mask = (df["country"] == country) & (
                df["major_group_and_revise_status"] == status
            )
            if any(mask):
                value = df.loc[mask, f"{metric}_pct"].values[0] * 100
                ci_lower = df.loc[mask, f"{metric}_ci_lower"].values[0] * 100
                ci_upper = df.loc[mask, f"{metric}_ci_upper"].values[0] * 100

                i_position = 2 * i

                # Add bar
                ax.bar3d(
                    j,
                    i_position,
                    0,
                    width,
                    depth,
                    value,
                    color=colors,
                    shade=True,
                    alpha=0.8,
                    zsort="max",
                    edgecolor="none",
                )

                # Add confidence interval
                x_center = j + width / 2
                y_center = i_position + depth / 2
                ax.plot(
                    [x_center, x_center],
                    [y_center, y_center],
                    [value, ci_upper],
                    color="orange",
                    linewidth=5,
                    zorder=100,
                )

                # Add CI cap
                ci_width = 0.1
                ax.plot(
                    [x_center - ci_width, x_center + ci_width],
                    [y_center, y_center],
                    [ci_upper, ci_upper],
                    color="orange",
                    linewidth=5,
                    zorder=100,
                )

    # Customize axes
    ax.set_xticks(
        [
            0 + (width / 2),
            1 + (width / 2),
            2 + (width / 2),
            3 + (width / 2),
            4 + (width / 2),
            5 + (width / 2),
        ]
    )
    ax.set_yticks([0 + (width / 2), 2 + (width / 2), 4 + (width / 2)])
    ax.set_xticklabels(major_group_and_revise_status, fontsize=32)
    ax.set_yticklabels(countries, fontsize=32)
    # Then adjust the padding
    ax.set_zlabel("Percentage", fontsize=32, labelpad=20)
    ax.tick_params(axis="z", labelsize=24, pad=10)

    # Set view angle and limits
    ax.view_init(elev=20, azim=60)
    ax.set_zlim(0, 100)
    ax.set_xlim(0, 5.6)
    ax.set_ylim(0, 5.6)

    return ax
