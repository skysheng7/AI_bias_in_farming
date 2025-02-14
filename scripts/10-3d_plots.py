"""
Generate 3d plots to summarize image clustering results
"""

import matplotlib.pyplot as plt
from pathlib import Path
import click

from AI_representation_bias_in_farming import utils
from AI_representation_bias_in_farming import module7_3d_plot


@click.command()
def main():
    # import manual cluster check note
    cluster = utils.read_csv_file("cluster_summary.csv")
    megadata = utils.read_csv_file("image_megadata_post_manual_fix.csv")

    # generate a plot for general dairy/pig farm images
    filtered = cluster[(cluster["model"] == "dall-e-3") & (cluster["country"].isna())]
    filtered_cluster = module7_3d_plot.add_grouping(filtered)
    filtered_mega = megadata[
        (megadata["model"] == "dall-e-3") & (megadata["country"].isna())
    ]

    # Create combined plot
    fig = module7_3d_plot.create_combined_farm_plot(
        filtered_cluster, filtered_mega, example_img_num=3, random_seed=11
    )

    # Save or display the plot
    plt.savefig(
        (Path() / "results" / "plots" / "3d_general_plot.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


# Only execute this code if the script is run directly
if __name__ == "__main__":
    main()
