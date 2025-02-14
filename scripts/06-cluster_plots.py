"""
Generate plot grids to summarize image clustering results
"""

import pandas as pd
from pathlib import Path
import click

from AI_representation_bias_in_farming import utils
from AI_representation_bias_in_farming import module5_extract_pic_feature_words
from AI_representation_bias_in_farming import module6_confidence_interval


@click.command()
def main():
    """generate plot grids that summarize image clusters"""

    #######################
    #######################
    ##### Basic stats #####
    #######################
    #######################

    # Print out basic stats about the image clustering
    # import the megadata dataframe
    megadata = utils.read_megadata()
    # import manual cluster check note
    manual_correction = utils.read_csv_file("outlier_image_manual_correction.csv")

    # Create a list of outdoor values to check, combine these and call it all to be "outdoor"
    outdoor_values = ["pasture", "pasture_or_mud"]
    megadata["GPT4o_cluster"] = megadata["GPT4o_cluster"].replace(
        outdoor_values, "outdoor"
    )
    # update the GPT4o_cluster based on my manual check results and cluster label corrections
    megadata2 = pd.merge(
        megadata,
        manual_correction,
        on=("file", "GPT4o_cluster", "GPT4o_cluster_explanation"),
        how="left",
    )
    megadata2["GPT4o_cluster"] = megadata2["GPT4o_cluster"].where(
        pd.isna(megadata2["change_to_cluster"]), megadata2["change_to_cluster"]
    )

    # output manually fixed cluster labels
    megadata2_file = (
        Path() / "results" / "megadata" / "image_megadata_post_manual_fix.csv"
    )
    megadata2.to_csv(megadata2_file, index=False)

    # generate summary
    cluster_summary = module5_extract_pic_feature_words.summarize_clusters(
        megadata2, output_file="cluster_summary.csv"
    )

    # calcualte CI
    updated_cluster = module6_confidence_interval.calculate_ci(
        cluster_summary, megadata2, n_resamples=10000, random_seed=7
    )
    # sort column names
    column_order = [
    'generation_type', 'country', 'farm_type', 'prompt', 'model',
    'total_rows', 'indoor_sum', 'outdoor_sum', 'other_sum',
    'indoor_ci_lower', 'indoor_pct', 'indoor_ci_upper',
    'outdoor_ci_lower', 'outdoor_pct', 'outdoor_ci_upper',
    'other_pct'
    ]
    updated_cluster = updated_cluster[column_order]
    
    # output manually fixed cluster labels
    updated_cluster_file = Path() / "results" / "megadata" / "cluster_summary.csv"
    updated_cluster.to_csv(updated_cluster_file, index=False)
    

    # what's the percentage of images that got manually corrected?
    manual_pct = round(len(manual_correction) / len(megadata2), 5)
    print(f"The percentage of images got manually corrected is: {manual_pct}")

    # what's the percentage of images that are categorized as "outdoor"
    outdoor_total_count = len(megadata2[megadata2["GPT4o_cluster"] == "outdoor"])
    outdoor_total_pct = round(outdoor_total_count / len(megadata2), 5)
    print(f"The percentage of images classified as 'outdoor': {outdoor_total_pct}")

    # what's the percentage of images that are categorized as "indoor"
    indoor_total_count = len(megadata2[megadata2["GPT4o_cluster"] == "indoor"])
    indoor_total_pct = round(indoor_total_count / len(megadata2), 5)
    print(f"The percentage of images classified as 'indoor': {indoor_total_pct}")

    # what's the percentage of images that are categorized as "other"
    other_total_count = len(megadata2[megadata2["GPT4o_cluster"] == "other"])
    other_total_pct = round(other_total_count / len(megadata2), 5)
    print(f"The percentage of images classified as 'other': {other_total_pct}")

    # what's the percentage of images that got manually corrected?
    metal_railing_count = megadata2["manual_note"].str.contains("metal", na=False).sum()
    unclear_count = megadata2["manual_note"].str.contains("unclear", na=False).sum()
    other_image_count = len(megadata2[megadata2["GPT4o_cluster"] == "other"])
    manual_check_total = len(manual_correction)
    metal_railing_pct = round((metal_railing_count / manual_check_total), 5)
    unclear_pct = round((unclear_count / manual_check_total), 5)
    print(
        f"The percentage of images got manually corrected to be in the 'other' category, because they contain animals housed behind metal railings but are outdoors: {metal_railing_pct}"
    )
    print(
        f"The percentage of images got manually corrected to be in the 'other' category, because the background is too unclear to judge: {unclear_pct}"
    )

    #####################################
    #####################################
    ##### Exploratory Data Analysis #####
    ##### Visualize image clusters ######
    #####################################
    #####################################
    # For ease of visualization to see how well the clustering performs.
    # I exported images belonging to each cluster into seperate folders.
    # Images are reduced to 50% quality as this is just an exploratory
    # analysis to see how legit the clustering labels are
    # below are based on GPT 4o's original cluster labels, fully automated
    print(
        "Start pasting images into 3 categories, with the goal of visualizing how well the auto-clustering performs. See images in results/cluster"
    )
    print("WARNING: this might take 15-30 minutes.")
    source_base = Path() / "results"
    dest_base = Path() / "results" / "cluster"
    module5_extract_pic_feature_words.cluster_farm_images(
        megadata, source_base, dest_base
    )
    print("Finished pasting images into 3 categories based on auto-clustering")

    # For ease of visualization to see how well the clustering performs.
    # I exported images belonging to each cluster into seperate folders.
    # Images are reduced to 50% quality as this is just an exploratory
    # analysis to see how legit the clustering labels are
    # below are based on GPT 4o's original cluster labels,
    # and my manual cluster label corrections after checking all images myself.
    print(
        "Start pasting images into 3 categories, with the goal of visualizing image clusters post manual-corrections. See images in results/cluster_post_manual_fix"
    )
    print("WARNING: this might take 15-30 minutes.")
    dest_base = Path() / "results" / "cluster_post_manual_fix"
    module5_extract_pic_feature_words.cluster_farm_images(
        megadata2, source_base, dest_base
    )
    print(
        "Finished pasting images into 3 categories, based on auto-clustering post manual corrections."
    )

    # generate plotgrid summarizing the cluster labels in images generated by DALLE 3
    output_path = Path() / "results" / "plots" / "cluster_summary_dall-e-3.png"
    model = "dall-e-3"
    fig1 = module5_extract_pic_feature_words.save_plot_grid(
        cluster_summary,
        output_path,
        megadata2,
        source_base,
        model,
        num_images=8,
        images_per_row=4,
        random_seed=21,
    )
    print("Generated a plot grid summarizing prompting results using DALL-E 3.")

    # Stable Diffusion 3.5-large
    output_path = Path() / "results" / "plots" / "cluster_summary_sd-3.5.png"
    model = "sd3.5-large"
    fig2 = module5_extract_pic_feature_words.save_plot_grid(
        cluster_summary,
        output_path,
        megadata2,
        source_base,
        model,
        num_images=8,
        images_per_row=4,
        random_seed=24,
    )
    print(
        "Generated a plot grid summarizing prompting results using Stable Diffussion 3.5-large."
    )


# Only execute this code if the script is run directly
if __name__ == "__main__":
    main()
