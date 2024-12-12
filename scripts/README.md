# Scripts

This directory contains analysis scripts for the AI representation bias in farming analysis pipeline.

## Script Descriptions

- `01-text_to_image.py`: Generates farming-related images using DALL-E 3 and Stable Diffusion based on text prompts.
- `02-image_to_text.py`: Processes generated images through GPT-4 Vision to obtain image descriptions.
- `03-image_cluster.py`: Performs clustering analysis based on image features and descriptions.

## Usage

Scripts should be run sequentially (01 → 02 → 03) to maintain data pipeline consistency. These scripts uses the functions built in `AI_representation_bias_in_farming` python package under `src`.
