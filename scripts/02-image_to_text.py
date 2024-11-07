"""This script take each image generated in the script '01-text_to_image.py' as input, 
    and generate a detailed text description for each image using OpenAI's GPT-4o model
"""

from AI_representation_bias_in_farming import module3_GPT4o


def main():
    model = "gpt-4o-2024-08-06"
    prompt = "Describe the image in detail."
    detail_level = "high"
    max_completion_tokens = 1000
    temperature = 0.2
    start_index = 0
    end_index = 479

    megadata = module3_GPT4o.describe_all_images(
        model,
        prompt,
        detail_level,
        max_completion_tokens,
        temperature,
        start_index,
        end_index,
    )


# Only execute this code if the script is run directly
if __name__ == "__main__":
    main()
