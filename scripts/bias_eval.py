"""This is the first module of the project prompting multiple text-to-image 
    generative AI models to create images related to livestock farming.
    The goal of this project is to evaluate what kind of images gets generated
    based on different prompting techniques, with the ultimate goal of evaluating
    potential representation bias these models have about livestock farming.
    
"""

import os
import base64
from pathlib import Path
import time

import pandas as pd
import numpy as np
import openai
from openai import OpenAI
from dotenv import load_dotenv

from AI_representation_bias_in_farming import utils
from AI_representation_bias_in_farming import module0_dalle3
from AI_representation_bias_in_farming import module1_sd
from AI_representation_bias_in_farming import module2_imagen3

# Only execute this code if the script is run directly
if __name__ == "__main__":
    # Load and set the API key
    load_dotenv()
    openai_key = os.getenv("openai_key")
    sd_key = os.getenv("stable_diffusion_key")
    imagen_key = os.getenv("imagen3_key")

    # Define the prompt types
    #
    # **no_revise**:
    #   original prompt +
    #   "I NEED to test how the tool works with extremely simple prompts.
    #   DO NOT add any detail, just use it AS-IS:"
    #
    # **reality**:
    #   "Please create an image that accurately represents the reality of what most " +
    #   [farm_type] +
    #   " farms look like in " +
    #   [country] + "."
    #
    # **basic**:
    #   "A " +
    #   [farm_type] +
    #   " farm"
    #
    # **typical**:
    #   "A typical " +
    #   [farm_type] +
    #   " farm"
    #
    # **country**:
    #   [country] in each prompt will be replaced by specific country names
    #   [farm_type] == "dairy": [country] = ["the United States", "Germany", "New Zealand"]
    #   [farm_type] == "pig": [country] = ["the United States", "Spain", "Australia"]
    generation_types = [
        "reality",
        "reality_no_revise",
        "reality_country",
        "reality_country_no_revise",
        "basic",
        "basic_no_revise",
        "basic_country",
        "basic_country_no_revise",
        "typical",
        "typical_no_revise",
        "typical_country",
        "typical_country_no_revise",
    ]

    farm_types = ["dairy", "pig"]  # type of livestock farms

    countries_by_farm_type = {
        "dairy": ["the United States", "Germany", "New Zealand"],
        "pig": ["the United States", "Spain", "Australia"],
    }  # list of countries with the biggest number of dairy cows and pigs in
    # North America, Europe and Oceania

    start_index = 11  # the starting index (ID) of images
    n = 90  # the total number of images you wish to generate in this batch
    max_retries = 3  # the maximum number of times we will retry prompting the model if the previous prompt failed due to safety reasons.
    retry_delay = 30  # the total number of seconds we wait to let the model reset before trying again

    # Loop through each combination of generation_type and farm_type
    for generation_type in generation_types:
        for farm_type in farm_types:
            if "country" in generation_type:
                # If generation_type includes "country", loop through each country
                countries = countries_by_farm_type[farm_type]
                for country in countries:
                    utils.dalle3_gen_image(
                        openai_key,
                        country=country,
                        farm_type=farm_type,
                        generation_type=generation_type,
                        start_index=start_index,
                        n=n,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        model="dall-e-3",
                    )
            else:
                # If no "country" in generation_type, set country to pd.NA
                utils.dalle3_gen_image(
                    openai_key,
                    country=pd.NA,
                    farm_type=farm_type,
                    generation_type=generation_type,
                    start_index=start_index,
                    n=n,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    model="dall-e-3",
                )
