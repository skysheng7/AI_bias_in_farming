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
from pydantic import BaseModel
from dotenv import load_dotenv

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
    megadata = utils.read_megadata()  # read in megadata

    # Load and set the API key
    load_dotenv()
    client = OpenAI()

    # Set end_index to the last index if not specified
    if end_index is None:
        end_index = len(megadata)

    # Iterate through the specified range of images
    for index in range(start_index, end_index):
        row = megadata.iloc[index]

        result = describe_1_image(
            row,
            model,
            client,
            prompt,
            detail_level,
            max_completion_tokens,
            temperature,
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

        utils.save_megadata_with_description(megadata)

    return megadata


def describe_1_image(
    row, model, client, prompt, detail_level, max_completion_tokens, temperature
):
    """
    Generates a description for a single image using an external API.

    This function retrieves an image file path based on parameters in the provided row,
    converts the image to a Base64 string, constructs a prompt message with the image,
    and sends a request to an API to generate a description.

    Parameters:
        row (pd.Series): A row from a DataFrame containing image metadata, including
                         model, generation type, and file name.
        model (str): The name of the model to use for generating the description.
        client (object): The API client used to make the request.
        prompt (str): The prompt text to guide the description generation.
        detail_level (str): resolution level for the input image, options: "high", "low"
        max_completion_tokens (int): Maximum tokens allowed for the API completion response.
        temperature (float): Sampling temperature to control output randomness.

    Returns:
        object: The result object from the API containing the generated description.
    """
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


def cluster_all_images(
    model,
    prompt_list,
    detail_level,
    max_completion_tokens,
    temperature,
    start_index=0,
    end_index=None,
):
    """
    Cluster images into 3 categories in a specified index range within a dataframe.

    This function iterates over a range of rows in the provided dataframe, `megadata`, and cluster
    images using the `cluster_1_image` function. The determined cluster and their
    token counts are stored in the dataframe under specified columns.

    Parameters:
        model (str): Model name used for generating the description.
        prompt_list (dictionary): a dictionary of prompts provided to the model to guide the cluster generation.
        detail_level (str): resolution level for the input image, options: "high", "low"
        max_completion_tokens (int): Maximum tokens allowed in the model's output.
        temperature (float): Sampling temperature for the model, controlling output randomness.
        start_index (int, optional): Starting index of the dataframe rows to process. Defaults to 0.
        end_index (int, optional): Ending index (exclusive) of the dataframe rows to process. Defaults to None,
                                   which processes up to the last row.

    Returns:
        pd.DataFrame: The modified `megadata` DataFrame with added descriptions and token counts.

    """
    megadata = utils.read_megadata()  # read in megadata

    # get API key
    # Load and set the API key
    load_dotenv()
    client = OpenAI()

    # Set end_index to the last index if not specified
    if end_index is None:
        end_index = len(megadata)

    # Iterate through the specified range of images
    for index in range(start_index, end_index):
        row = megadata.iloc[index]
        farm_type = row["farm_type"]
        prompt = prompt_list[farm_type]

        # if there are existing GPT4o cluster, don't regenerate
        if (
            "GPT4o_cluter" in row.keys()
            and pd.notna(row["GPT4o_cluter"])
            and row["GPT4o_cluter"] != ""
        ):
            megadata = megadata
        else:  # if there is no pre-existing GPT4o cluster
            result = cluster_1_image(
                row,
                model,
                client,
                prompt,
                detail_level,
                max_completion_tokens,
                temperature,
            )

            json_response = result.choices[0].message.parsed
            category = json_response.category
            explanation = json_response.explanation
            output_token = result.usage.completion_tokens

            # Store the results back in the dataframe
            megadata.at[index, "cluter_model"] = model
            megadata.at[index, "GPT4o_cluter"] = category
            megadata.at[index, "GPT4o_cluster_explanation"] = explanation
            megadata.at[index, "GPT4o_cluster_token_count"] = output_token
            megadata.at[index, "GPT4o_cluster_prompt"] = prompt
            megadata.at[index, "GPT4o_cluster_system_fingerprint"] = (
                result.system_fingerprint
            )

        utils.save_megadata_with_description(megadata)

    return megadata


class clusterExtraction(BaseModel):
    category: str
    explanation: str


def cluster_1_image(
    row, model, client, prompt, detail_level, max_completion_tokens, temperature
):
    """
    Generates a description for a single image using an external API.

    This function retrieves an image file path based on parameters in the provided row,
    converts the image to a Base64 string, constructs a prompt message with the image,
    and sends a request to an API to generate a description.

    Parameters:
        row (pd.Series): A row from a DataFrame containing image metadata, including
                         model, generation type, and file name.
        model (str): The name of the model to use for generating the description.
        client (object): The API client used to make the request.
        prompt (str): The prompt text to guide the description generation.
        detail_level (str): resolution level for the input image, options: "high", "low"
        max_completion_tokens (int): Maximum tokens allowed for the API completion response.
        temperature (float): Sampling temperature to control output randomness.

    Returns:
        object: The result object from the API containing the generated description.
    """
    input_file = find_file_path(row["model"], row["generation_type"], row["file"])
    base64_image = convert_png_to_base64(input_file)

    # Constructing prompt messages
    prompt_messages = [
        {
            "role": "system",
            "content": "Categorize an image into one of three categories: 'outdoor', 'indoor', or 'other'. Provide a brief explanation for your classification decision. Structure your response as a JSON format with two fields: 'category' for the classification and 'explanation' for your reasoning.",
        },
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
        "response_format": clusterExtraction,
    }

    # ask for json format
    result = client.beta.chat.completions.parse(**params)
    return result
