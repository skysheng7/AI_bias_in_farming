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
        width_ratios=[1.5, 0.7, 1.5, 0.7],
        height_ratios=[1] * len(farm_types),
    )

    # plot each row based on farm type
    for row_idx, farm_type in enumerate(farm_types):
        filtered_df = cluster_df[cluster_df["farm_type"] == farm_type]

        for col_idx, metric in enumerate(metrics):
            # Plot outdoor or indoor metrics (top row)
            ax1 = fig.add_subplot(gs[row_idx, (2 * col_idx)], projection="3d")

            plot_3d_generation_types(filtered_df, metric=metric, ax=ax1)

            # Add subplot labels (A, B, C, D)
            label = chr(65 + (2 * row_idx + col_idx))  # 65 is ASCII for 'A'
            ax1.text2D(
                0,
                1.1,
                f"({label})",
                transform=ax1.transAxes,
                fontsize=35,
                fontweight="bold",
            )

            if row_idx == 0:
                if metric == "outdoor":
                    ax1.text2D(
                        0.65,
                        1.3,
                        "Have access to pasture/mud outdoors",
                        transform=ax1.transAxes,
                        fontsize=45,
                        fontweight="bold",
                        horizontalalignment="center",
                    )
                else:
                    ax1.text2D(
                        0.65,
                        1.3,
                        "Exclusively indoors",
                        transform=ax1.transAxes,
                        fontsize=45,
                        fontweight="bold",
                        horizontalalignment="center",
                    )

            # Add farm type labels on the y-axis for the first column of each row
            if col_idx == 0:
                ax1.text2D(
                    -0.27,
                    0.5,
                    f"{farm_type.capitalize()} farm",
                    transform=ax1.transAxes,
                    fontsize=45,
                    fontweight="bold",
                    rotation=90,
                    verticalalignment="center",
                )

            # Add 3 example images
            selected_images = image_random_select(
                filtered_mega, farm_type, metric, example_img_num, random_seed
            )
            ax2 = fig.add_subplot(gs[row_idx, 2 * col_idx + 1])
            plot_3example_images(ax2, selected_images, "", num_images=example_img_num)

    # Adjust layout
    plt.subplots_adjust(wspace=0.05, hspace=0.06)

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
                else:
                    j_position = j

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
                    alpha=0.6,
                    zsort="max",
                    edgecolor="none",
                )

                # Add confidence interval upper
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

                # Add CI cap upper
                ci_width = 0.1
                ax.plot(
                    [x_center - ci_width, x_center + ci_width],
                    [y_center, y_center],
                    [ci_upper, ci_upper],
                    color="orange",
                    linewidth=5,
                    zorder=100,
                )

                # Add confidence interval lower
                x_center = j_position + width / 2
                y_center = i + depth / 2
                ax.plot(
                    [x_center, x_center],
                    [y_center, y_center],
                    [value, ci_lower],
                    color="orange",
                    linewidth=5,
                    zorder=0,
                )

                # Add CI cap lower
                ci_width = 0.1
                ax.plot(
                    [x_center - ci_width, x_center + ci_width],
                    [y_center, y_center],
                    [ci_lower, ci_lower],
                    color="orange",
                    linewidth=5,
                    zorder=0,
                )

    # Customize axes
    mapping = {"basic": "'basic'", "typical": "'typical'", "reality": "'reality'"}

    # x axis and y axis
    major_groups_update = [mapping[group] for group in major_groups]
    ax.set_xticks([0 + (width / 2), 2 + (width / 2)])
    ax.set_yticks([0 + (width / 2), 1 + (width / 2), 2 + (width / 2)])
    ax.set_xticklabels(revise_status, fontsize=32, rotation_mode="anchor", rotation=15)
    ax.set_yticklabels(
        major_groups_update,
        fontsize=32,
        ha="right",
        rotation_mode="anchor",
        rotation=15,
    )

    # z axis
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("Percentage", fontsize=32, labelpad=28, rotation=90)
    ax.tick_params(axis="z", labelsize=24, pad=12)

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


def create_country_plot(
    cluster_df,
    filtered_mega,
    countries,
    real_world_by_farm,
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
    countries : list of str
        List of target countries to analyze, representing major livestock producers
        across North America, Europe, and Oceania.
    real_world_by_farm : dict
        Nested dictionary containing real-world data for each farm type and location
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

    metrics = ["outdoor", "indoor"]
    major_groups = ["basic", "typical", "reality"]

    # Create figure with custom layout
    plt.style.use("classic")
    fig = plt.figure(figsize=(34, 45))
    gs = gridspec.GridSpec(
        len(countries),
        4,
        width_ratios=[1.5, 0.7, 1.5, 0.7],
        height_ratios=[1] * len(countries),
    )

    # plot each row based on farm type
    for row_idx, major_group in enumerate(major_groups):
        cur_major_group_cluster = cluster_df[cluster_df["major_group"] == major_group]
        cur_major_group__mega = filtered_mega[
            filtered_mega["major_group"] == major_group
        ]

        for col_idx, metric in enumerate(metrics):
            # Plot outdoor or indoor metrics (top row)
            ax1 = fig.add_subplot(gs[row_idx, (2 * col_idx)], projection="3d")

            if real_world_by_farm is not None:
                real_world_by_farm_cur_metric = real_world_by_farm[metric]
            else:
                real_world_by_farm_cur_metric = None
            plot_3d_farm_by_country(
                cur_major_group_cluster,
                countries,
                real_world_by_farm_cur_metric,
                metric=metric,
                ax=ax1,
            )

            # Add subplot labels (A, B, C, D)
            label = chr(65 + (2 * row_idx + col_idx))  # 65 is ASCII for 'A'
            ax1.text2D(
                0,
                1.1,
                f"({label})",
                transform=ax1.transAxes,
                fontsize=35,
                fontweight="bold",
            )

            if row_idx == 0:
                if metric == "outdoor":
                    ax1.text2D(
                        0.65,
                        1.3,
                        "Have access to pasture/mud outdoors",
                        transform=ax1.transAxes,
                        fontsize=45,
                        fontweight="bold",
                        horizontalalignment="center",
                    )
                else:
                    ax1.text2D(
                        0.65,
                        1.3,
                        "Exclusively indoors",
                        transform=ax1.transAxes,
                        fontsize=45,
                        fontweight="bold",
                        horizontalalignment="center",
                    )

            # Add prompt type labels on the y-axis for the first column of each row
            if col_idx == 0:
                ax1.text2D(
                    -0.3,
                    0.5,
                    f"'{major_group.capitalize()}' prompt",
                    transform=ax1.transAxes,
                    fontsize=45,
                    fontweight="bold",
                    rotation=90,
                    verticalalignment="center",
                )

            # Add 3 example images
            selected_images = image_random_select_country(
                cur_major_group__mega,
                metric,
                countries,
                example_img_num,
                random_seed,
            )

            ax2 = fig.add_subplot(gs[row_idx, 2 * col_idx + 1])
            plot_3example_images(ax2, selected_images, "", num_images=example_img_num)

    # Adjust layout
    plt.subplots_adjust(wspace=0.05, hspace=0.06)

    return fig


def image_random_select_country(
    cur_mega, metric, countries, example_img_num=3, random_seed=11
):
    """
    Randomly select a specified number of image paths based on farm type and metric criteria.

    Parameters
    ----------
    cur_mega : pandas.DataFrame
        DataFrame containing image metadata with columns:
        - farm_type: type of farm
        - GPT4o_cluster: metric cluster label
        - model: model name
        - generation_type: type of generation
        - file: image filename
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

    cur_megadata = cur_mega[(cur_mega["GPT4o_cluster"] == metric)]

    for country in countries:
        cur_megadata_country = cur_megadata[cur_megadata["country"] == country]

        # randomly select a few images
        selected_rows = cur_megadata_country.sample(
            n=1, random_state=random_seed if random_seed is not None else None
        )
        row = selected_rows.iloc[0]
        new_image = (
            Path()
            / "results"
            / (row["model"] + "-images")
            / row["generation_type"]
            / row["file"]
        )
        selected_images.append(new_image)

    return selected_images


def plot_3d_farm_by_country(
    df, countries, real_world_by_farm_cur_metric=None, metric="indoor", ax=None
):
    """
    Create a 3D bar plot showing generation types with confidence intervals.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data
    countries : list of str
        List of target countries to analyze, representing major livestock producers
        across North America, Europe, and Oceania.
    real_world_by_farm_cur_metric: dict, (default = None)
       a dictionary containing real-world data for the current farm type, either indoor or outdoor condition
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
    revise_status = ["no revise", "revise"]
    width = depth = 0.5

    # Style settings
    ax.grid(False)  # Remove grid
    ax.xaxis._axinfo["grid"].update({"color": (1, 1, 1, 0)})  # Remove grid lines
    ax.yaxis._axinfo["grid"].update({"color": (1, 1, 1, 0)})
    ax.zaxis._axinfo["grid"].update({"color": (1, 1, 1, 0)})
    ax.set_box_aspect([1, 1, 1])

    x_min = y_min = z_min = 0
    x_max = 2.8
    y_max = 2.6
    z_max = 100

    # Plot bars and confidence intervals
    for i, country in enumerate(countries):
        for j, rev in enumerate(revise_status):
            mask = (df["country"] == country) & (df["revise_status"] == rev)
            if any(mask):
                value = df.loc[mask, f"{metric}_pct"].values[0] * 100
                ci_lower = df.loc[mask, f"{metric}_ci_lower"].values[0] * 100
                ci_upper = df.loc[mask, f"{metric}_ci_upper"].values[0] * 100

                if j == 1:
                    j_position = j + 1
                else:
                    j_position = j

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
                    alpha=0.6,
                    zsort="max",
                    edgecolor="none",
                )

                # Add confidence interval upper
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

                # Add CI cap upper
                ci_width = 0.1
                ax.plot(
                    [x_center - ci_width, x_center + ci_width],
                    [y_center, y_center],
                    [ci_upper, ci_upper],
                    color="orange",
                    linewidth=5,
                    zorder=100,
                )

                # Add confidence interval lower
                x_center = j_position + width / 2
                y_center = i + depth / 2
                ax.plot(
                    [x_center, x_center],
                    [y_center, y_center],
                    [value, ci_lower],
                    color="orange",
                    linewidth=5,
                    zorder=0,
                )

                # Add CI cap lower
                ci_width = 0.1
                ax.plot(
                    [x_center - ci_width, x_center + ci_width],
                    [y_center, y_center],
                    [ci_lower, ci_lower],
                    color="orange",
                    linewidth=5,
                    zorder=0,
                )

                # add real world data as a line
                if real_world_by_farm_cur_metric is not None:
                    real_data_list = real_world_by_farm_cur_metric[country]
                    for real_data in real_data_list:
                        ax.plot(
                            [x_min, x_max],
                            [y_center, y_center],
                            [real_data, real_data],
                            color="lightcoral",
                            linewidth=12,
                            linestyle="solid",
                            alpha=1,
                            zorder=0,
                        )

    # Customize axes
    countries = ["U.S." if x == "the United States" else x for x in countries]
    ax.set_xticks([0 + (width / 2), 2 + (width / 2)])
    ax.set_yticks([0 + (width / 2), 1 + (width / 2), 2 + (width / 2)])
    ax.set_xticklabels(revise_status, fontsize=36, rotation_mode="anchor", rotation=15)
    ax.set_yticklabels(
        countries, fontsize=36, ha="right", rotation_mode="anchor", rotation=15
    )

    # Then adjust the padding
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("Percentage", fontsize=36, labelpad=28, rotation=90)
    ax.tick_params(axis="z", labelsize=26, pad=12)

    # Set view angle and limits
    ax.view_init(elev=20, azim=60)
    ax.set_zlim(z_min, z_max)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    return ax


def create_and_save_farm_plot(
    farm_type,
    real_world,
    country_filtered_cluster,
    country_filtered_mega,
    countries_by_farm_type,
    save_path,
    example_img_num=3,
    random_seed=7,
):
    """
    Creates and saves a 3D plot for a specific farm type showing country-level data.

    Parameters
    ----------
    farm_type : str
        Type of farm to plot (e.g. 'dairy', 'pig')
    real_world : dict
        Nested dictionary containing real-world data for each farm type and location
    country_filtered_cluster : pd.DataFrame
        DataFrame containing the filtered cluster data for all countries
    country_filtered_mega : pd.DataFrame
        DataFrame containing the filtered mega data for all countries
    countries_by_farm_type : dict
        Dictionary mapping farm types to list of countries to include
    save_path : Path
        Base path where plot should be saved
    example_img_num : int, optional
        Number of example images to show per category, by default 3
    random_seed : int, optional
        Random seed for reproducibility, by default 7

    Returns
    -------
    fig
        Saves plot to disk at specified location, and return figure
    """
    if real_world is not None:
        real_world_by_farm = real_world[farm_type]
    else:
        real_world_by_farm = None
    cur_cluster = country_filtered_cluster[
        country_filtered_cluster["farm_type"] == farm_type
    ]
    cur_mega = country_filtered_mega[country_filtered_mega["farm_type"] == farm_type]
    countries = countries_by_farm_type[farm_type]

    fig = create_country_plot(
        cur_cluster,
        cur_mega,
        countries,
        real_world_by_farm,
        example_img_num=example_img_num,
        random_seed=random_seed,
    )

    # Save plot
    plt.savefig(
        save_path / f"3d_country_plot_{farm_type}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    return fig
