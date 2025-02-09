"""
analyze the prompts OpenAI used to evaluate DALL-E 3. The prompts are image captions extracted from MSCOCO 2014 evaluation dataset
"""

import pandas as pd
from pathlib import Path


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

# cattle prompt related to pastre, grass
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
]
yard_terms = ["fence", "yard"]
indoor_terms = ["barn", "bars", "metal", "line", "pen"]

cattle_outdoor_prompts = cattle_prompts[
    cattle_prompts["prompts"].str.contains("|".join(outdoor_terms))
]
cattle_indoor_prompts = cattle_prompts[
    cattle_prompts["prompts"].str.contains("|".join(indoor_terms))
]
cattle_yard_prompts = cattle_prompts[
    cattle_prompts["prompts"].str.contains("|".join(yard_terms))
]
cattle_other_prompts = cattle_prompts[
    ~cattle_prompts["prompts"].str.contains(
        "|".join(outdoor_terms + indoor_terms + yard_terms)
    )
]

# print the prompt about pig
print(f"{n_pig} prompt about pig found:")
print(pig_prompts["prompts"].iloc[0])

# number of prompts about cows
print(f"{n_cow} prompt about cows found:")

for index, row in cattle_indoor_prompts.iterrows():
    print(row["prompts"])
