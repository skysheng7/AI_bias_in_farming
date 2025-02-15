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

    #############################################################################
    ####################### General dairy/pig farms #############################
    #############################################################################
    # generate a plot grid for general dairy/pig farm images
    filtered = cluster[(cluster["model"] == "dall-e-3") & (cluster["country"].isna())]
    filtered_cluster = module7_3d_plot.add_grouping(filtered)
    filtered_mega = megadata[
        (megadata["model"] == "dall-e-3") & (megadata["country"].isna())
    ]
    # Create combined plot
    fig = module7_3d_plot.create_combined_farm_plot(
        filtered_cluster,
        filtered_mega,
        countries_by_farm_type=None,
        example_img_num=3,
        random_seed=11,
    )
    # Save or display the plot
    plt.savefig(
        (Path() / "results" / "plots" / "3d_general_plot.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    #############################################################################
    ###################### Dairy/pig farms by country ###########################
    #############################################################################
    # generate a plot grid for dairy/pig farm in different countries
    country_filtered = cluster[
        (cluster["model"] == "dall-e-3") & (~cluster["country"].isna())
    ]

    # list of countries with the biggest number of dairy cows and pigs in North America, Europe and Oceania
    countries_by_farm_type = {
        "dairy": ["the United States", "Germany", "New Zealand"],
        "pig": ["the United States", "Spain", "Australia"],
    }

    country_filtered_cluster = module7_3d_plot.add_grouping(country_filtered)
    country_filtered_mega = megadata[
        (megadata["model"] == "dall-e-3") & (~megadata["country"].isna())
    ]
    # Create combined plot
    fig = module7_3d_plot.create_combined_farm_plot(
        country_filtered_cluster,
        country_filtered_mega,
        countries_by_farm_type,
        example_img_num=3,
        random_seed=11,
    )
    # Save or display the plot
    plt.savefig(
        (Path() / "results" / "plots" / "3d_country_plot.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


# Only execute this code if the script is run directly
if __name__ == "__main__":
    main()
