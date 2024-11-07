"""This script take each image generated in the script '01-text_to_image.py' as input, 
    and generate a detailed text description for each image using OpenAI's GPT-4o model
"""

from AI_representation_bias_in_farming import module3_GPT4o

model = "gpt-4o-2024-08-06"
prompt = "Describe the image in detail."
detail_level = "high"
max_tokens = 1000
temperature = 0.2

megadata = module3_GPT4o.describe_all_images(
    model, prompt, detail_level, max_tokens, temperature
)
