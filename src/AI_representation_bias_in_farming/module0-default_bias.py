import os
import base64
from pathlib import Path

import pandas as pd
import numpy as np
from openai import OpenAI

from AI_representation_bias_in_farming import utils

# Read the API key from the file, Define the path to your API key file
key_file_path = (
    Path.home()
    / "Library"
    / "CloudStorage"
    / "OneDrive-UBC"
    / "R package project and Git"
    / "API_keys"
    / "AI_bias_API_key.txt"
)
with open(key_file_path, "r") as file:
    api_key = file.read().strip()
# Set the API key for OpenAI
client = OpenAI(api_key=api_key)

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
#    
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
    "pig": ["the United States", "Spain", "Australia"]

}

# Loop through each combination of generation_type and farm_type
for generation_type in generation_types:
    for farm_type in farm_types:
        if "country" in generation_type:
            # If generation_type includes "country", loop through each country
            countries = countries_by_farm_type[farm_type]
            for country in countries:
                utils.gen_image_train(
                    client,
                    country=country,
                    farm_type=farm_type,
                    generation_type=generation_type,
                    start_index=1,
                    n=10,
                )
        else:
            # If no "country" in generation_type, set country to pd.NA
            utils.gen_image_train(
                client,
                country=pd.NA,
                farm_type=farm_type,
                generation_type=generation_type,
                start_index=1,
                n=10,
            )
