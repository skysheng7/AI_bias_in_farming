"""Library of general functions used to generate prompts and save metadata
   generated in the text-to-image generation process
"""

import base64
import os
from pathlib import Path
import time

import pandas as pd
import openai
from openai import OpenAI
import requests


def get_prompt(country, farm_type, generation_type):
    """
    generate a text prompt based on the country, type of farms, and image generation style.

    Parameters:
        country (str): which country should the image content be based on. options: pd.NA, Canada, the United States, Germany
        farm_type (str): which type of livestock farm shoulld the image depict. options: dairy, pig
        generation_type (str): what type of text prompt is this.

    Returns:
        cur_prompt (str): the generated tect prompt

    """

    # read in the type and styles of generation in generation_type
    word_list = generation_type.split("_")
    type1 = word_list[0]
    if len(word_list) > 1:
        revise_note = word_list[-2] + "_" + word_list[-1]
    else:
        revise_note = ""

    # specify the first part of the prompt based on the generation_type
    if type1 == "reality":
        cur_prompt = (
            "Please create an image that accurately represents the reality of what most "
            + farm_type
            + " farms look like"
        )
    elif type1 == "typical":
        cur_prompt = "A typical " + farm_type + " farm"
    elif type1 == "basic":
        cur_prompt = "A " + farm_type + " farm"

    # add the specific country to the prompt if there is "country" in the generation type
    if "country" in generation_type:
        cur_prompt = cur_prompt + " in " + country + "."
    else:
        cur_prompt = cur_prompt + "."

    # openAI automatically revise user prompt. try to stop that
    if revise_note == "no_revise":
        cur_prompt = (
            cur_prompt
            + "I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS:"
        )
        # openAI API note: "While it is not currently possible to disable this feature, you can use prompting to get outputs closer to your requested image by adding the following to your prompt:I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS:"

    return cur_prompt


def save_imag(generation_type, img_count, farm_type, image_bytes, country, model):
    """
    Save the generated images into the local folder

    Parameters:
        generation_type (str): what type of text prompt is this.
        img_count (int): the current index (ID) of the generated image
        farm_type (str): which type of livestock farm shoulld the image depict. options: dairy, pig
        country (str): which country should the image content be based on. options: pd.NA, Canada, the United States, Germany
        model (str): which model did we use, dall-e-3 or imagen3, or stable-diffusion-3.5

    Returns:
        None

    """

    # define output dir
    img_dir = Path() / "results" / (model + "-images") / generation_type
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)  # create the directory if the directory does not exist

    # define filename
    if pd.isna(country):
        file_name = (
            farm_type + "_farm_" + generation_type + "_" + str(img_count) + ".png"
        )
    else:
        file_name = (
            farm_type
            + "_farm_"
            + generation_type
            + "_"
            + country
            + "_"
            + str(img_count)
            + ".png"
        )

    # Save the image locally
    image_name = img_dir / file_name
    with open(image_name, "wb") as image_file:
        image_file.write(image_bytes)


def save_megadata(
    img_count,
    country,
    farm_type,
    generation_type,
    cur_prompt,
    revised_input,
    model,
    response_type,
):
    """
    Save the generated images into the local folder

    Parameters:
        img_count (int): the current index (ID) of the generated image
        country (str): which country should the image content be based on. options: pd.NA, Canada, the United States, Germany
        farm_type (str): which type of livestock farm shoulld the image depict. options: dairy, pig
        generation_type (str): what type of text prompt is this. options: default, default_no_revise, default_country, reality_country
        cur_prompt (str): the generated tect prompt
        revised_input (str): what GPT-4o automatically rephrased the prompt into based on cur_prompt (the prompt we generated)
        model (str): which model did we use, dall-e-3 or imagen3, or stable-diffusion-3.5

    Returns:
        None

    """

    # define output dir
    data_output_dir = Path() / "results" / "megadata"

    # define filename
    if pd.isna(country):
        file_name = (
            farm_type + "_farm_" + generation_type + "_" + str(img_count) + ".png"
        )
    else:
        file_name = (
            farm_type
            + "_farm_"
            + generation_type
            + "_"
            + country
            + "_"
            + str(img_count)
            + ".png"
        )

    # Save the megadata
    result_df = pd.DataFrame(
        {
            "file": [file_name],
            "generation_type": [generation_type],
            "country": [country],
            "farm_type": [farm_type],
            "prompt": [cur_prompt],
            "revised_prompt": [revised_input],
            "model": [model],
            "size": ["1024x1024"],
            "quality": ["standard"],
            "response_format": [response_type],
        }
    )
    megadata_file = data_output_dir / "image_megadata.csv"

    if os.path.exists(megadata_file):
        # Load the existing dataframe
        existing_df = pd.read_csv(megadata_file)
        # Append new rows
        combined_df = pd.concat([existing_df, result_df], ignore_index=True)
    else:
        # If file does not exist, the result_df is the combined dataframe
        combined_df = result_df

    combined_df.to_csv(megadata_file, index=False)
