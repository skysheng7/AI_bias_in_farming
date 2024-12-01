"""Library of functions used to see which words are the key features for describing images of 
   extensive VS intensive farms
"""

import pandas as pd
from pathlib import Path
from PIL import Image

from AI_representation_bias_in_farming import module4_word_freq_count


# import shutil


# import the dataframe that records the 1 gram bag of words in all gpt-4o descriptions
gpt4o_description_file = (
    Path(".") / "results" / "megadata" / "GPT4o_description_1_2gram_bag_of_words.csv"
)
gpt4o_description_1_2gram = pd.read_csv(gpt4o_description_file, header=0)
gpt4o_description_1_2gram
# list of words associated with extensive dairy
dairy_extensive = ["grazing", "graze", "grass", "grasslands", "pasture", "pastures"]
dairy_intensive = [
    "automated",
    "ceiling",
    "interior",
    "machinery",
    "machine",
    "machines",
    "modern",
    "metal",
    "roofed",
    "metal railing",
    "metal railings",
    "large window",
    "large windows",
    "barn spacious",
]
pig_extensive = ["mud", "muddy", "soil", "dirt", "outdoor", "rural", "green", "grazing"]
pig_intensive = [
    "floor",
    "automated",
    "ceiling",
    "interior",
    "machinery",
    "machine",
    "machines",
    "modern",
    "metal",
    "roofed",
    "metal railing",
    "metal railings",
    "large window",
    "large windows",
    "barn spacious",
]


def process_farm_images(
    df, farm_type, condition_type, word_list, source_base, dest_base
):
    """
    Filter images based on word occurrences and organize them into folders.

    Parameters:
    df: DataFrame with image metadata
    farm_type: 'dairy' or 'pig'
    condition_type: 'extensive' or 'intensive'
    word_list: list of words to filter by
    source_base: base path for source images
    dest_base: base path for destination folders
    """
    for word in word_list:
        # Create destination folder if it doesn't exist
        dest_folder = dest_base / farm_type / condition_type / word
        if not dest_folder.exists():
            dest_folder.mkdir(parents=True)

        # Filter rows where the word count is 1
        filtered_rows = df[(df[word] == 1) & (df["farm_type"] == farm_type)]

        for _, row in filtered_rows.iterrows():
            # Get source image path
            source_file = source_base / row["generation_type"] / row["file"]

            if source_file.exists():
                try:
                    # Open and resize image
                    img = Image.open(source_file)
                    # Reduce quality by resizing to 50% of original size
                    new_size = tuple(dim // 2 for dim in img.size)
                    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

                    # Save to destination
                    dest_file = dest_folder / row["file"]
                    img_resized.save(dest_file, quality=85, optimize=True)

                except Exception as e:
                    print(f"Error processing {source_file}: {e}")
            else:
                print(f"Source file not found: {source_file}")


# Set up base paths
source_base = Path(".") / "results" / "dall-e-3-images"
dest_base = Path(".") / "results" / "cluster_eda"

# Process dairy extensive
process_farm_images(
    gpt4o_description_1_2gram,
    "dairy",
    "extensive",
    dairy_extensive,
    source_base,
    dest_base,
)

# Process dairy intensive
process_farm_images(
    gpt4o_description_1_2gram,
    "dairy",
    "intensive",
    dairy_intensive,
    source_base,
    dest_base,
)

# Process pig extensive
process_farm_images(
    gpt4o_description_1_2gram, "pig", "extensive", pig_extensive, source_base, dest_base
)

# Process pig intensive
process_farm_images(
    gpt4o_description_1_2gram, "pig", "intensive", pig_intensive, source_base, dest_base
)
