"""Library of general functions used to generate prompts and save metadata
   generated in the text-to-image generation process
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from AI_representation_bias_in_farming import module0_dalle3
from AI_representation_bias_in_farming import module1_sd


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
        model (str): which model did we use, dall-e-3 or stable-diffusion-3.5

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
    finish_reason,
):
    """
    Save the generated images into the local folder

    Parameters:
        img_count (int): the current index (ID) of the generated image
        country (str): which country should the image content be based on.
        farm_type (str): which type of livestock farm shoulld the image depict. options: dairy, pig
        generation_type (str): what type of text prompt is this. e.g., "basic", "basic_no_revise", "typical", "typical_no_revise"
        cur_prompt (str): the generated tect prompt
        revised_input (str): what GPT-4o automatically rephrased the prompt into based on cur_prompt (the prompt we generated)
        model (str): which model did we use, dall-e-3 or stable-diffusion-3.5

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
            "finish_reason": [finish_reason],
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


def get_key(model):
    """
    Retrieve the API key corresponding to the specified AI model.

    This function loads environment variables and returns the appropriate API key based on the provided model name.

    Parameters:
        model (str): The name of the AI model for which the API key is required.
                     Accepted values are:
                     - "dall-e-3"
                     - "sd3.5-large"

    Returns:
        str or None: The API key as a string if found; otherwise, None.

    Raises:
        ValueError: If the provided model name is not recognized.

    Notes:
        - Ensure that the environment variables 'openai_key', 'stable_diffusion_key' are set in your environment or in a .env file.
        - The function uses the python-dotenv package to load environment variables from a .env file if present.
    """

    # Load and set the API key
    load_dotenv()

    if model == "dall-e-3" or model == "gpt-4o-2024-08-06":
        key = os.getenv("OPENAI_API_KEY")
    elif model == "sd3.5-large":
        key = os.getenv("stable_diffusion_key")
    else:
        key = None

    return key


def gen_image(
    country,
    farm_type,
    generation_type,
    start_index,
    n,
    max_retries,
    retry_delay,
    model="dall-e-3",
):
    """
    Generate images based on specified parameters using the selected AI model.

    This function interfaces with different AI models to generate images according to the provided parameters. It supports models such as DALL·E 3, Stable Diffusion 3.5 Large.

    Parameters:
        country (str): which country should the image content be based on.
        farm_type (str): which type of livestock farm shoulld the image depict. options: dairy, pig
        generation_type (str): what type of text prompt is this. e.g., "basic", "basic_no_revise", "typical", "typical_no_revise"
        start_index (int): the start index (ID) of the generated image in a roll
        n (int): how many images you want to generate starting from the start index
        max_retries (int): the maximum number of times we will retry prompting the model if the previous prompt failed due to safety reasons.
        retry_delay (int): the total number of seconds we wait to let the model reset before trying again
        model (str, optional): The AI model to use for image generation. Defaults to 'dall-e-3'. Options: 'dall-e-3', 'sd3.5-large'

    Returns:
        None

    Notes:
        - Ensure that the appropriate API keys are configured for each model in .env
        - The function delegates the image generation task to model-specific functions based on the 'model' parameter.
    """

    key = get_key(model)

    # prompt DALLE-3
    if model == "dall-e-3":
        module0_dalle3.dalle3_gen_image(
            key=key,
            country=country,
            farm_type=farm_type,
            generation_type=generation_type,
            start_index=start_index,
            n=n,
            max_retries=max_retries,
            retry_delay=retry_delay,
            model=model,
        )
    elif model == "sd3.5-large":  # prompt Stable Diffusion
        module1_sd.sd_gen_image(
            key=key,
            country=country,
            farm_type=farm_type,
            generation_type=generation_type,
            start_index=start_index,
            n=n,
            max_retries=max_retries,
            retry_delay=retry_delay,
            model=model,
        )
    else:
        raise ValueError(
            "Invalid model value provided. Please enter: 'dall-e-3', or 'sd3.5-large'"
        )


def read_megadata():
    """
    Reads the 'image_megadata.csv' file from the 'results/megadata' directory.

    This function loads the 'image_megadata.csv' file located in the specified directory
    into a pandas DataFrame and returns it.

    Returns:
        pd.DataFrame: A DataFrame containing the contents of 'image_megadata.csv'.
    """
    megadata_file = Path() / "results" / "megadata" / "image_megadata.csv"
    megadata = pd.read_csv(megadata_file, header=0)
    return megadata


def save_megadata_with_description(megadata):
    """
    Saves the given DataFrame to 'image_megadata.csv' in the 'results/megadata' directory.

    This function writes the contents of the provided DataFrame to 'image_megadata.csv',
    overwriting any existing file in the specified directory.

    Parameters:
        megadata (pd.DataFrame): The DataFrame to be saved.
    """
    # Define output directory
    megadata_file = Path() / "results" / "megadata" / "image_megadata.csv"
    megadata.to_csv(megadata_file, index=False)
