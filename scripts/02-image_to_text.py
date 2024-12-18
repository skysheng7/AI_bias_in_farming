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

    megadata = module2_GPT4o.describe_all_images(
        model,
        prompt,
        detail_level,
        max_completion_tokens,
        temperature,
        start_index,
        end_index,
    )


# Only execute this code if the script is run directly
if __name__ == "__main__":
    main()
