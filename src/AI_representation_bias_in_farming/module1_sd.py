"""Library of general functions used to prompting Stable Diffusion 3.5 model
"""

from pathlib import Path
import time

import requests
import pandas as pd

from AI_representation_bias_in_farming import utils


def sd_prompt_for_img(
    key,
    country,
    farm_type,
    generation_type,
    max_retries=3,
    retry_delay=30,
    model="sd3.5-large",
):
    """
    Prompt chatGPT API to generate an image response in base 64 encoded jason format

    Parameters:
        key (str): API key to access Gemini imagen3 model
        country (str): which country should the image content be based on.
        farm_type (str): which type of livestock farm shoulld the image depict. options: dairy, pig
        generation_type (str): what type of text prompt is this. e.g., "basic", "basic_no_revise", "typical", "typical_no_revise"
        max_retries (int): the maximum number of times we will retry prompting the model if the previous prompt failed due to safety reasons.
        retry_delay (int): the total number of seconds we wait to let the model reset before trying again
        model (str, optional): The AI model to use for image generation. Options: #'sd3.5-large' | 'sd3.5-large-turbo' | 'sd3.5-medium'

    Returns:
        response: the response from API call

    """
    cur_prompt = utils.get_prompt(country, farm_type, generation_type)

    for attempt in range(max_retries):
        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
            headers={"authorization": f"Bearer {key}", "accept": "image/*"},
            files={"none": ""},
            data={
                "model": model,  #'sd3.5-large' | 'sd3.5-large-turbo' | 'sd3.5-medium'
                "prompt": cur_prompt,
                "output_format": "png",
            },
        )
        if response.status_code == 200:  # success
            return (cur_prompt, response)
        elif response.status_code == 429:  # error 429
            print(f"{generation_type} for {farm_type} in {country}:")
            print(
                f"Attempt {attempt + 1} in reprompting the model after last attempt failed due to prompting more than 150 requests in 10 seconds. Retrying..."
            )
        elif response.status_code == 403:  # error 403
            print(f"{generation_type} for {farm_type} in {country}:")
            print(
                f"Attempt {attempt + 1} in reprompting the model after last attempt failed due to request being flagged for content moderation systems. Retrying..."
            )
        else:
            raise Exception(str(response.json()))

        if attempt == (max_retries - 1):
            print(
                "Maximum number of retries reached, terminating the image generation process."
            )
            raise Exception(str(response.json()))

        time.sleep(retry_delay)  # wait for a while to let the model reset


def sd_gen_image(
    key,
    country,
    farm_type,
    generation_type,
    start_index,
    n,
    max_retries=3,
    retry_delay=30,
    model="sd3.5-large",
):
    """
    Generate n images in a roll, save the images into local folder, save the megadata related to each image into a csv file.

    Parameters:
        key (str): API key to access DALLE-3 model
        country (str): which country should the image content be based on.
        farm_type (str): which type of livestock farm shoulld the image depict. options: dairy, pig
        generation_type (str): what type of text prompt is this. e.g., "basic", "basic_no_revise", "typical", "typical_no_revise"
        start_index (int): the start index (ID) of the generated image in a roll
        n (int): how many images you want to generate starting from the start index
        max_retries (int): the maximum number of times we will retry prompting the model if the previous prompt failed due to safety reasons.
        retry_delay (int): the total number of seconds we wait to let the model reset before trying again
        model (str, optional): The AI model to use for image generation. Options: 'sd3.5-large' | 'sd3.5-large-turbo' | 'sd3.5-medium'

    Returns:
        None

    """

    # generate n images
    for img_count in range(start_index, (start_index + n)):
        # prompt API for a image response
        cur_prompt, response = sd_prompt_for_img(
            key, country, farm_type, generation_type, max_retries, retry_delay, model
        )
        image_bytes = response.content  # get the image data
        finish_reason = response.headers.get("Finish-Reason")

        utils.save_imag(
            generation_type, img_count, farm_type, image_bytes, country, model
        )
        utils.save_megadata(
            img_count,
            country,
            farm_type,
            generation_type,
            cur_prompt,
            revised_input=pd.NA,  # stable diffusion model does not automaticaly revise prompt
            model=model,
            response_type="bytes",
            finish_reason=finish_reason,
        )
