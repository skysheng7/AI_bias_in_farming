"""This script take each image generated in the script '01-text_to_image.py' as input, 
    and generate a detailed text description for each image using OpenAI's GPT-4o model
"""

import click

from AI_representation_bias_in_farming import module2_GPT4o


@click.command()
@click.option(
    "--start_index",
    default=0,
    type=int,
    help="The starting index in the dataframe which stores all the metadata of generated images. This will be the index of the first image that get processed to generate a text description using GPT4o. Default is 0.",
)
@click.option(
    "--end_index",
    default=None,
    type=int,
    help="The end index in the dataframe which stores all the metadata of generated images. This wil be the index of the last image that get processed to generate a text description using GPT4o. Default is None, so end of the dataframe.",
)
def main(start_index, end_index):
    model = "gpt-4o-2024-08-06"
    prompt = "Describe the image in detail."
    detail_level = "high"
    max_completion_tokens = 1000
    temperature = 0.2

    print(
        "Start generating text descriptions using GPT-4o for images based on images provided..."
    )
    print(
        "WARNING: This may take 10 minutes to a couple hours to run depending on how many images you wish to generate descriptions for."
    )
    print(
        "This will use GPT-4V to create detailed descriptions of each image starting at row=(start_index+2) in results/megadata/image_megadata.csv. results/megadata/image_megadata.csv will be updated to include a text description for each image"
    )

    megadata = module2_GPT4o.describe_all_images(
        model,
        prompt,
        detail_level,
        max_completion_tokens,
        temperature,
        start_index,
        end_index,
    )

    print("Finished generating text descriptions for all images!")


# Only execute this code if the script is run directly
if __name__ == "__main__":
    main()
