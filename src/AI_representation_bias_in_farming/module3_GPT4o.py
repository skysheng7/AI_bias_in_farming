"""Library of general functions used to prompting OpenAI's GPT-4o model
    for generating text descriptions for images
"""

import time
import base64
from pathlib import Path
from io import BytesIO

import pandas as pd
from PIL import Image
import openai
from openai import OpenAI

from AI_representation_bias_in_farming import utils


def find_file_path(model, generation_type, file):
    """find the path for the image file (name as indicated by file)

    Args:
        model (str): which model did we use to generate this image, Options: dall-e-3', 'sd3.5-large', "imagen-3.0-generate-001"
        generation_type (str): what type of text prompt is this. e.g., "basic", "basic_no_revise", "typical", "typical_no_revise"
        file (str): the name of the image file

    Returns:
        path to the image file
    """
    input_file = Path() / "results" / (model + "-images") / generation_type / file

    return input_file


def describe_all_images(
    model,
    prompt,
    detail_level,
    max_completion_tokens,
    temperature,
    start_index=0,
    end_index=None,
):
    """
    Generates and stores descriptions for images in a specified index range within a dataframe.

    This function iterates over a range of rows in the provided dataframe, `megadata`, and generates
    descriptions for images using the `describe_1_image` function. The generated descriptions and their
    token counts are stored in the dataframe under specified columns.

    Parameters:
        model (str): Model name used for generating the description.
        prompt (str): Prompt provided to the model to guide the description generation.
        detail_level (str): resolution level for the input image, options: "high", "low"
        max_completion_tokens (int): Maximum tokens allowed in the model's output.
        temperature (float): Sampling temperature for the model, controlling output randomness.
        start_index (int, optional): Starting index of the dataframe rows to process. Defaults to 0.
        end_index (int, optional): Ending index (exclusive) of the dataframe rows to process. Defaults to None,
                                   which processes up to the last row.

    Returns:
        pd.DataFrame: The modified `megadata` DataFrame with added descriptions and token counts.

    """
    megadata = read_megadata()  # read in megadata

    # get API key
    key = utils.get_key(model)
    client = OpenAI(api_key=key)

    # Set end_index to the last index if not specified
    if end_index is None:
        end_index = len(megadata)

    # Iterate through the specified range of images
    for index in range(start_index, end_index):
        row = megadata.iloc[index]

        result = describe_1_image(
            row, model, client, prompt, detail_level, max_completion_tokens, temperature
        )

        result_content = result.choices[
            0
        ].message.content  # extract content from result
        output_token = result.usage.completion_tokens

        # Store the results back in the dataframe
        megadata.at[index, "description_model"] = model
        megadata.at[index, "GPT4o_description"] = result_content
        megadata.at[index, "GPT4o_description_token_count"] = output_token
        megadata.at[index, "GPT4o_prompt"] = prompt
        megadata.at[index, "GPT4o_image_resolution"] = detail_level
        megadata.at[index, "GPT4o_temperature"] = temperature
        megadata.at[index, "GPT4o_system_fingerprint"] = result.system_fingerprint

        save_megadata_with_description(megadata)

    return megadata


def describe_1_image(
    row, model, client, prompt, detail_level, max_completion_tokens, temperature
):
    input_file = find_file_path(row["model"], row["generation_type"], row["file"])
    base64_image = convert_png_to_base64(input_file)

    # Constructing prompt messages
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpg;base64, {base64_image}",
                        "detail": detail_level,
                    },
                },
            ],
        },
    ]

    # Parameters for the API call
    params = {
        "model": model,
        "messages": prompt_messages,
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
    }

    # Assuming client.chat.completions.create is a function call to an external API
    result = client.chat.completions.create(**params)
    return result


def convert_png_to_base64(png_image_path):
    """
    Converts a PNG image to a base64-encoded string.

    Parameters:
    png_image_path (str): The file path to the PNG image.

    Returns:
    str: The base64-encoded string of the PNG image.
    """
    # Open the PNG image
    with Image.open(png_image_path) as img:
        # Create a BytesIO object to hold the byte stream
        buffer = BytesIO()

        # Save the image to the buffer as PNG
        img.save(buffer, format="PNG")

        # Get the byte stream and encode it to Base64
        png_base64 = base64.b64encode(buffer.getvalue()).decode()
    return png_base64


def read_megadata():
    megadata_file = Path() / "results" / "megadata" / "image_megadata.csv"
    megadata = pd.read_csv(megadata_file, header=0)
    return megadata


def save_megadata_with_description(megadata):
    # define output dir
    megadata_file = Path() / "results" / "megadata" / "image_megadata.csv"
    megadata.to_csv(megadata_file, index=False)
