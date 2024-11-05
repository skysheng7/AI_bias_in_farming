"""Library of general functions used to prompting OpenAI's DALLE-3 model
"""

import base64
import os
from pathlib import Path
import time

import pandas as pd
import openai
from openai import OpenAI
import requests

from AI_representation_bias_in_farming import utils


def dalle3_prompt_for_img(
    client, country, farm_type, generation_type, max_retries=3, retry_delay=30
):
    """
    Prompt chatGPT API to generate an image response in base 64 encoded jason format

    Parameters:
        client: openAI API client for generating images
        country (str): which country should the image content be based on. options: pd.NA, Canada, the United States, Germany
        farm_type (str): which type of livestock farm shoulld the image depict. options: dairy, pig
        generation_type (str): what type of text prompt is this.
        max_retries (int): the maximum number of times we will retry prompting the model if the previous prompt failed due to safety reasons.
        retry_delay (int): the total number of seconds we wait to let the model reset before trying again

    Returns:
        response: the response from API call

    """
    cur_prompt = utils.get_prompt(country, farm_type, generation_type)

    for attempt in range(max_retries):
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=cur_prompt,
                size="1024x1024",
                quality="standard",
                response_format="b64_json",
                n=1,
            )
            return {"response": response, "prompt": cur_prompt}
        except (openai.BadRequestError, openai.RateLimitError) as e:
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


def dalle3_gen_image(
    key,
    country,
    farm_type,
    generation_type,
    start_index,
    n,
    max_retries=3,
    retry_delay=30,
    model="dall-e-3",
):
    """
    Generate n images in a roll, save the images into local folder, save the megadata related to each image into a csv file.

    Parameters:
        key (str): API key to access DALLE-3 model
        country (str): which country should the image content be based on. options: pd.NA, Canada, the United States, Germany
        farm_type (str): which type of livestock farm shoulld the image depict. options: dairy, pig
        generation_type (str): what type of text prompt is this.
        start_index (int): the start index (ID) of the generated image in a roll
        n (int): how many images you want to generate starting from the start index
        max_retries (int): the maximum number of times we will retry prompting the model if the previous prompt failed due to safety reasons.
        retry_delay (int): the total number of seconds we wait to let the model reset before trying again
        model (str): which model did we use, dall-e-3 or imagen3, or stable-diffusion-3.5

    Returns:
        None

    """

    client = OpenAI(api_key=key)

    # generate n images
    for img_count in range(start_index, (start_index + n)):
        # prompt API for a image response
        result = dalle3_prompt_for_img(
            client, country, farm_type, generation_type, max_retries, retry_delay
        )
        response = result["response"]
        cur_prompt = result["prompt"]  # generate prompt

        image_data = response.data[0].b64_json
        revised_input = response.data[
            0
        ].revised_prompt  # get what GPT-4o automatically rephrased the prompt into
        image_bytes = base64.b64decode(image_data)  # get the image data

        utils.save_imag(
            generation_type, img_count, farm_type, image_bytes, country, model
        )
        utils.save_megadata(
            img_count,
            country,
            farm_type,
            generation_type,
            cur_prompt,
            revised_input,
            model,
            response_type="b64_json",
        )
