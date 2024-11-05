"""Library of general functions used to prompting Google Gemini's Imagen3 model
"""

import os
from pathlib import Path
import time


from AI_representation_bias_in_farming import utils


def imagen3_gen_image(
    key,
    country,
    farm_type,
    generation_type,
    start_index,
    n,
    max_retries=3,
    retry_delay=30,
    model="imagen3",
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
        model (str): which model did we use, Options: 'dall-e-3', 'sd3.5-large', 'imagen3'.

    Returns:
        None

    """
