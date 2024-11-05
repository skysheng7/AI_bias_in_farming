"""Library of general functions used to prompting Google Gemini's Imagen3 model
"""

import os
from pathlib import Path
import time

import google.generativeai as genai
import pandas as pd

from AI_representation_bias_in_farming import utils

def imagen3_prompt_for_img(
    imagen,
    country,
    farm_type,
    generation_type,
    max_retries=3,
    retry_delay=30,
    model="imagen-3.0-generate-001",
):
    """
    Prompt chatGPT API to generate an image response in base 64 encoded jason format

    Parameters:
        client: openAI API client for generating images
        country (str): which country should the image content be based on.
        farm_type (str): which type of livestock farm shoulld the image depict. options: dairy, pig
        generation_type (str): what type of text prompt is this. e.g., "basic", "basic_no_revise", "typical", "typical_no_revise"
        max_retries (int): the maximum number of times we will retry prompting the model if the previous prompt failed due to safety reasons.
        retry_delay (int): the total number of seconds we wait to let the model reset before trying again
        model (str): which model did we use, Options: "imagen-3.0-generate-001"

    Returns:
        response: the response from API call

    """
    cur_prompt = utils.get_prompt(country, farm_type, generation_type)

    for attempt in range(max_retries):
        try:
            result = imagen.generate_images(
                prompt=cur_prompt,
                number_of_images=1,
                safety_filter_level="block_only_high",
                person_generation="allow_adult",
                aspect_ratio="1:1"
            )
            return {"response": result, "prompt": cur_prompt}
        except google.api_core.exceptions.InvalidArgument as e:
            print(f"{generation_type} for {farm_type} in {country}:")
            print(
                f"Attempt {attempt + 1} in reprompting the model after last attempt failed due to {e}. Retrying..."
            )

            if attempt == (max_retries - 1):
                print(
                    "Maximum number of retries reached, terminating the image generation process."
                )
                raise e  # re-raise the error and terminate the program

            time.sleep(retry_delay)  # wait for a while to let the model reset


def imagen3_gen_image(
    key,
    country,
    farm_type,
    generation_type,
    start_index,
    n,
    max_retries=3,
    retry_delay=30,
    model="imagen-3.0-generate-001",
):
    """
    Generate n images in a roll, save the images into local folder, save the megadata related to each image into a csv file.

    Parameters:
        key (str): API key to access imagen3 model
        country (str): which country should the image content be based on.
        farm_type (str): which type of livestock farm shoulld the image depict. options: dairy, pig
        generation_type (str): what type of text prompt is this. e.g., "basic", "basic_no_revise", "typical", "typical_no_revise"
        start_index (int): the start index (ID) of the generated image in a roll
        n (int): how many images you want to generate starting from the start index
        max_retries (int): the maximum number of times we will retry prompting the model if the previous prompt failed due to safety reasons.
        retry_delay (int): the total number of seconds we wait to let the model reset before trying again
        model (str): which model did we use, Options: "imagen-3.0-generate-001"

    Returns:
        None

    """

    genai.configure(api_key=os.environ[key])

    imagen = genai.ImageGenerationModel(model)

    # generate n images
    for img_count in range(start_index, (start_index + n)):
        # prompt API for a image response
        result, cur_prompt = imagen3_prompt_for_img(
            imagen, country, farm_type, generation_type, max_retries, retry_delay, model
        )
        image_bytes = result.images # get the image data

        utils.save_imag(
            generation_type, img_count, farm_type, image_bytes, country, model
        )
        utils.save_megadata(
            img_count,
            country,
            farm_type,
            generation_type,
            cur_prompt,
            revised_input = pd.NA,
            model = model,
            response_type="bytes",
            finish_reason=pd.NA,
        )
