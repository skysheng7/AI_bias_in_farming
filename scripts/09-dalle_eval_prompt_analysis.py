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
cattle_terms = ["cow", "cows", "cattle", "cattles", "bull", "bulls", "calf", "calves"]
pig_terms = [" pig ", " pigs "]

# Filter for pig and cow related prompts
cattle_prompts = prompts[prompts["prompts"].str.contains("|".join(cattle_terms))]
pig_prompts = prompts[prompts["prompts"].str.contains("|".join(pig_terms))]
pig_prompts = pig_prompts[~pig_prompts["prompts"].str.contains("guinea")]
n_cow = len(cattle_prompts)
n_pig = len(pig_prompts)

# cattle prompt related to pastre, grass
pasture_terms = ["pasture", "grass", "green field"]
cattle_pasture_prompts = cattle_prompts[
    cattle_prompts["prompts"].str.contains("|".join(pasture_terms))
]
not_cattle_pasture_prompts = cattle_prompts[
    ~cattle_prompts["prompts"].str.contains("|".join(pasture_terms))
]

# print the prompt about pig
print(f"{n_pig} prompt about pig found:")
print(pig_prompts["prompts"].iloc[0])

#
print(f"{n_cow} prompt about cows found:")

for index, row in not_cattle_pasture_prompts.iterrows():
    print(row["prompts"])
