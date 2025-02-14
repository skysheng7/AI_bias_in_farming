"""
Generate 3d plots to summarize image clustering results
"""

import pandas as pd
from pathlib import Path
import click

from AI_representation_bias_in_farming import utils
from AI_representation_bias_in_farming import module7_3d_plot


@click.command()
def main():
    # import manual cluster check note
    cluster = utils.read_csv_file("cluster_summary.csv")
    farm_types = ["dairy", "pig"]
    metrics = ["indoor", "outdoor"]

    # generate a plot for general dairy/pig farm images
    interest = cluster[(cluster["model"] == "dall-e-3") & (cluster["country"].isna())]
    for farm_type in farm_types:
        filtered_df = interest[interest["farm_type"] == farm_type]
        for metric in metrics:
            module7_3d_plot.plot_3d_generation_types(filtered_df, metric="indoor")


# Only execute this code if the script is run directly
if __name__ == "__main__":
    main()
