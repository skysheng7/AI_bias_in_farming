"""
Generate 3d plots to summarize image clustering results
"""

import matplotlib.pyplot as plt

from AI_representation_bias_in_farming import module5_extract_pic_feature_words


def plot_3d_generation_types(df, metric="indoor"):
    """
    Create a 3D bar plot showing generation types with confidence intervals.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data. Must have columns:
        - generation_type
        - {metric}_pct, {metric}_ci_lower, {metric}_ci_upper
    metric : str, optional (default='indoor')
        Which metric to plot. Either 'indoor' or 'outdoor'

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axis objects
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

    # Set up the coordinates for the bars
    major_groups = [
        "basic",
        "typical",
        "reality",
    ]
    revise_status = ["no revise", "revise"]

    # Create the 3D plot
    plt.style.use("classic")
    fig, ax = plt.subplots(figsize=(10, 12), subplot_kw={"projection": "3d"})
    ax.set_box_aspect([1, 1, 1])

    ax.grid(False)  # Remove grid
    ax.xaxis._axinfo["grid"].update({"color": (1, 1, 1, 0)})  # Remove grid lines
    ax.yaxis._axinfo["grid"].update({"color": (1, 1, 1, 0)})
    ax.zaxis._axinfo["grid"].update({"color": (1, 1, 1, 0)})

    # Define colors and width/depth of bars
    if metric == "indoor":
        colors = "lightskyblue"
    else:
        colors = "yellowgreen"
    width = depth = 0.6

    # Plot bars for each combination
    for i, major in enumerate(major_groups):
        for j, rev in enumerate(revise_status):
            mask = (filtered_df["major_group"] == major) & (
                filtered_df["revise_status"] == rev
            )
            if any(mask):
                # Get values
                value = filtered_df.loc[mask, f"{metric}_pct"].values[0] * 100
                ci_lower = filtered_df.loc[mask, f"{metric}_ci_lower"].values[0] * 100
                ci_upper = filtered_df.loc[mask, f"{metric}_ci_upper"].values[0] * 100

                if j == 1:
                    j = j + 0.5

                # Plot bar
                ax.bar3d(
                    j,
                    i,
                    0,
                    width,
                    depth,
                    value,
                    color=colors,
                    shade=True,  # Add shading for 3D effect
                    alpha=1,  # Solid bars
                    zsort="max",  # Proper sorting of faces
                    edgecolor="none",
                )  # no edge color

                # Add confidence interval
                # if (value > 0) & (value < 100):
                # Center of the bar
                x_center = j + width / 2
                y_center = i + depth / 2

                # Plot CI line
                ax.plot(
                    [x_center, x_center],
                    [y_center, y_center],
                    [value, ci_upper],
                    color="orange",  # Black CI lines
                    linewidth=2,
                    zorder=100,
                )  # Ensure CI lines are visible

                # Add small horizontal lines at ends of CI
                ci_width = 0.1
                # upper CI cap 1
                ax.plot(
                    [x_center - ci_width, x_center + ci_width],
                    [y_center, y_center],
                    [ci_upper, ci_upper],
                    color="orange",
                    linewidth=2,
                    zorder=100,
                )

    # Set axis labels
    ax.set_xticks(
        [0 + (width / 2), 1.5 + (width / 2)]
    )  # Center the labels between bars
    ax.set_yticks([0 + (width / 2), 1 + (width / 2), 2 + (width / 2)])
    ax.set_xticklabels(revise_status, fontsize=20)
    ax.set_yticklabels(major_groups, fontsize=20)

    ax.set_zlabel("Percentage", fontsize=20)
    ax.tick_params(axis="z", labelsize=16)

    # Set view angle for better visualization
    ax.view_init(elev=20, azim=50)

    # Set z-axis limits from 0 to 100 (for percentages)
    ax.set_zlim(0, 100)
    ax.set_xlim(0, 2.3)
    ax.set_ylim(0, 2.6)

    return fig, ax
