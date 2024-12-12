"""This script take each image generated in the script '01-text_to_image.py' as input, 
    and cluster images into one of the 4 categories using OpenAI's GPT-4o model. The 3
    categories are: 
    [1] outdoor (in this image, there are cows that are outside on pasture or grassland, and there are pigs outside on pasture, grassland, mud or snow)
    [2] indoor (all cows or pigs in this image are kept indoors)
    [3] other (everything else)
"""

from AI_representation_bias_in_farming import module2_GPT4o


def get_prompt_list():
    prompt_list = {
        "dairy": "Please classify this image into one of these 3 categories. Provide a brief explanation of why you chose this category.\n\n[1] pasture: at least one cow depicted in this image (can be a model or diorama) is clearly standing, lying or grazing on pasture or grassland. It's ok if there are cows both on pasture and kept indoors.\n\n[2] indoor: All visible cows are housed inside buildings or structures\n\n[3] other: Any other images that either:\n   - Does not clearly fit the pasture or indoor categories\n   - background is too ambiguous or unclear to classify",
        "pig": "Please classify this image into one of these 3 categories. Provide a brief explanation of why you chose this category.\n\n[1] pasture_or_mud: at least one pig depicted in this image (can be a model or diorama) is clearly standing, lying or grazing on pasture, mud, dirt, straw, snow, or grassland. It's ok if there are pigs both on pasture and kept indoors.\n\n[2] indoor: All visible pigs are housed inside buildings or structures\n\n[3] other: Any other image that either:\n   - Does not clearly fit the pasture or mud or indoor categories\n   - background is too ambiguous or unclear to classify",
    }

    return prompt_list


def main():

    model = "gpt-4o-2024-08-06"
    prompt_list = get_prompt_list()
    detail_level = "high"
    max_completion_tokens = 1000
    temperature = 0.2
    start_index = 0
    end_index = None

    megadata = module2_GPT4o.cluster_all_images(
        model,
        prompt_list,
        detail_level,
        max_completion_tokens,
        temperature,
        start_index,
        end_index,
    )


# Only execute this code if the script is run directly
if __name__ == "__main__":
    main()
