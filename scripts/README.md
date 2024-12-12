# Scripts

This directory contains analysis scripts for the AI representation bias in farming analysis pipeline.

## Script Descriptions

- `01-text_to_image.py`: Generates dairy/pig farm images using DALL-E 3 and Stable Diffusion based on text prompts.
- `02-image_to_text.py`: Processes generated images through GPT-4o to obtain image descriptions.
- `03-image_cluster.py`: Prompt GPT-4o to cluster images into 3 categories and provide explanation.

## Usage

Scripts should be run sequentially (01 → 02 → 03) to maintain data pipeline consistency. These scripts uses the functions built in `AI_representation_bias_in_farming` python package under `src`.
