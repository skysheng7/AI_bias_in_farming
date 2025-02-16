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
    cluster = module7_3d_plot.add_grouping(cluster)
    megadata = module7_3d_plot.add_grouping(megadata)

    #############################################################################
    ####################### General dairy/pig farms #############################
    #############################################################################
    # generate a plot grid for general dairy/pig farm images
    print("Start generating 3D plot for general dairy and pig farm image summary...")

    filtered_cluster = cluster[
        (cluster["model"] == "dall-e-3") & (cluster["country"].isna())
    ]
    filtered_mega = megadata[
        (megadata["model"] == "dall-e-3") & (megadata["country"].isna())
    ]
    # Create combined plot
    fig = module7_3d_plot.create_combined_farm_plot(
        filtered_cluster,
        filtered_mega,
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

    print("General dairy and pig farm image 3Dplot FINISHED!")

    #############################################################################
    ###################### Dairy/pig farms by country ###########################
    #############################################################################

    # real-world data
    real_world = {
        "dairy": {
            "indoor": None,
            "outdoor": {
                "the United States": [3],
                "Germany": [50, 5],
                "New Zealand": [99],
            },
        },
        "pig": {
            "indoor": {"the United States": [98], "Spain": [94.9], "Australia": [90]},
            "outdoor": None,
        },
    }

    # generate a plot grid for dairy/pig farm in different countries
    country_filtered_cluster = cluster[
        (cluster["model"] == "dall-e-3") & (~cluster["country"].isna())
    ]
    country_filtered_mega = megadata[
        (megadata["model"] == "dall-e-3") & (~megadata["country"].isna())
    ]

    # list of countries with the biggest number of dairy cows and pigs in North America, Europe and Oceania
    countries_by_farm_type = {
        "dairy": ["the United States", "Germany", "New Zealand"],
        "pig": ["the United States", "Spain", "Australia"],
    }

    # Create combined plot 1 per farm type
    farm_types = country_filtered_mega["farm_type"].unique()

    ################### dairy farm by country ######################
    print("Start generating 3D plot for dairy farm image by country summary...")

    farm_type = farm_types[0]  # don't plot real world data for now
    fig = module7_3d_plot.create_and_save_farm_plot(
        farm_type,
        real_world=None,
        country_filtered_cluster=country_filtered_cluster,
        country_filtered_mega=country_filtered_mega,
        countries_by_farm_type=countries_by_farm_type,
        save_path=(Path() / "results" / "plots"),
        example_img_num=3,
        random_seed=7,
    )

    print("Dairy farm image by country 3D plot FINISHED!")

    ################### pig farm by country ######################
    print("Start generating 3D plot for pig farm image by country summary...")
    farm_type = farm_types[1]  # don't plot real world data for now

    fig = module7_3d_plot.create_and_save_farm_plot(
        farm_type,
        real_world=None,
        country_filtered_cluster=country_filtered_cluster,
        country_filtered_mega=country_filtered_mega,
        countries_by_farm_type=countries_by_farm_type,
        save_path=(Path() / "results" / "plots"),
        example_img_num=3,
        random_seed=10,
    )

    print("Pig farm image by country 3D plot FINISHED!")


# Only execute this code if the script is run directly
if __name__ == "__main__":
    main()
