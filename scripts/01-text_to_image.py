"""This is the first module of the project prompting multiple text-to-image 
    generative AI models to create images related to livestock farming.
    The goal of this project is to evaluate what kind of images gets generated
    based on different prompting techniques, with the ultimate goal of evaluating
    potential representation bias these models have about livestock farming.
    
"""

from pathlib import Path

import pandas as pd
import click

from AI_representation_bias_in_farming import utils


@click.command()
@click.option(
    "--start_index",
    default=1,
    type=int,
    help="The starting index (ID) of images. Default is 1.",
)
@click.option(
    "--total_image_num",
    default=10,
    type=int,
    help="The total number of images you wish to generate in this batch. Default is 10.",
)
@click.option(
    "--model",
    default="dall-e-3",
    type=str,
    help="Which text-to-image generative model to use. Options: 'dall-e-3', 'sd3.5-large'. Default is 'dall-e-3'",
)
def main(start_index, total_image_num, model):
    # Define the prompt types
    #
    # **no_revise**:
    #   original prompt +
    #   "I NEED to test how the tool works with extremely simple prompts.
    #   DO NOT add any detail, just use it AS-IS:"
    #
    # **reality**:
    #   "Please create an image that accurately represents the reality of what most " +
    #   [farm_type] +
    #   " farms look like in " +
    #   [country] + "."
    #
    # **basic**:
    #   "A " +
    #   [farm_type] +
    #   " farm"
    #
    # **typical**:
    #   "A typical " +
    #   [farm_type] +
    #   " farm"
    #
    # **country**:
    #   [country] in each prompt will be replaced by specific country names
    #   [farm_type] == "dairy": [country] = ["the United States", "Germany", "New Zealand"]
    #   [farm_type] == "pig": [country] = ["the United States", "Spain", "Australia"]
    generation_types = [
        "reality",
        "reality_no_revise",
        "reality_country",
        "reality_country_no_revise",
        "basic",
        "basic_no_revise",
        "basic_country",
        "basic_country_no_revise",
        "typical",
        "typical_no_revise",
        "typical_country",
        "typical_country_no_revise",
    ]

    farm_types = ["dairy", "pig"]  # type of livestock farms

    countries_by_farm_type = {
        "dairy": ["the United States", "Germany", "New Zealand"],
        "pig": ["the United States", "Spain", "Australia"],
    }  # list of countries with the biggest number of dairy cows and pigs in North America, Europe and Oceania

    max_retries = 3  # the maximum number of times we will retry prompting the model if the previous prompt failed due to safety reasons.
    retry_delay = 30  # the total number of seconds we wait to let the model reset before trying again

    print("Start generating images based on text descriptions...")
    print(
        "WARNING: This may take 10 minutes to a couple hours to run depending on how many images you wish to generate"
    )
    print(
        "This will create multiple new images based on text prompts using DALL-E 3 or Stable Diffusion 3.5-large (n images per unique prompts, we have 48 unique prompts in total, with n=total_image_num). Generated images will be in results/dall-e-3-images, image metadata will be stored in results/megadata/image_megadata.csv."
    )
    # Loop through each combination of generation_type and farm_type
    for generation_type in generation_types:
        for farm_type in farm_types:
            if "country" in generation_type:
                # If generation_type includes "country", loop through each country
                countries = countries_by_farm_type[farm_type]
            else:
                countries = [pd.NA]

            for country in countries:
                utils.gen_image(
                    country,
                    farm_type,
                    generation_type,
                    start_index,
                    total_image_num,
                    max_retries,
                    retry_delay,
                    model=model,
                )

    print("Finished generating all images!")


# Only execute this code if the script is run directly
if __name__ == "__main__":
    main()
