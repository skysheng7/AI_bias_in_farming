"""
analyze the prompts OpenAI used to evaluate DALL-E 3. The prompts are image captions extracted from MSCOCO 2014 evaluation dataset
"""

import pandas as pd
from pathlib import Path

# set pandas to display complete prompt in a row
pd.set_option('display.max_colwidth', None)  # Show full column content
pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.width', None)         # Don't wrap to multiple lines

# read in the 8k_coco.txt file
coco_path = Path() / "dalle3_eval_data" / "8k_coco.txt"
with open(coco_path, "r") as file:
    lines = file.readlines()

prompts = pd.DataFrame({"prompts": lines})
prompts["prompts"] = prompts["prompts"].str.lower()

# define cow or cattle related terms
cattle_terms = ["cow", "cows", "cattle", "cattles"]
cattle_neg_terms = ["cowboy", "statue of a cow"]
pig_terms = ["pig", "pigs"]
pig_neg_terms = ["pigeon", "guinea pig"]

# Filter for pig and cow related prompts
cattle_prompts = prompts[prompts["prompts"].str.contains("|".join(cattle_terms))]
cattle_prompts = cattle_prompts[
    ~cattle_prompts["prompts"].str.contains("|".join(cattle_neg_terms))
]
pig_prompts = prompts[prompts["prompts"].str.contains("|".join(pig_terms))]
pig_prompts = pig_prompts[~pig_prompts["prompts"].str.contains("|".join(pig_neg_terms))]
n_cow = len(cattle_prompts)
n_pig = len(pig_prompts)

# cattle prompt related to pastre, grass; terms related to indoor housing
outdoor_terms = [
    "pasture",
    "grass",
    "green field",
    "lush green",
    "grazing",
    "graze",
    "field",
    "beach",
    "bushes",
    "meadow",
    "paddock"
]
indoor_terms = ["bars", "metal", "pen", "stall"]

# cattle prompts that are describing outdoor VS indoor, and everything else (i.e., "other")
cattle_outdoor_prompts = cattle_prompts[
    cattle_prompts["prompts"].str.contains("|".join(outdoor_terms))
]
cattle_outdoor_prompts = cattle_outdoor_prompts[~cattle_outdoor_prompts["prompts"].str.contains("|".join(indoor_terms))]
cattle_indoor_prompts = cattle_prompts[
    cattle_prompts["prompts"].str.contains("|".join(indoor_terms))
]
cattle_indoor_prompts = cattle_indoor_prompts[~cattle_indoor_prompts["prompts"].str.contains("|".join(outdoor_terms))]
cattle_other_prompts = cattle_prompts[
    ~cattle_prompts["prompts"].str.contains(
        "|".join(outdoor_terms + indoor_terms)
    )
]

# print the prompt about pig
print(f"{n_pig} prompt about pig found:")
print(pig_prompts["prompts"].iloc[0])

# number of prompts about cows
print(f"{n_cow} prompt about cows found:")
print(f"{len(cattle_outdoor_prompts)} prompts are describing cows outdoor on pasture, beach etc.")
print(f"{len(cattle_indoor_prompts)} prompts are describing cows housed indoors, inside of pens and stalls.")
print(f"{len(cattle_other_prompts)} prompts are describing cows housed indoors, inside of pens and stalls.")
