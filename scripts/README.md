# Scripts

This directory contains analysis scripts for the AI representation bias in farming analysis pipeline.

## Script Descriptions

- `00-update_enviroment_yml.py`: automatically append version number for each package name installed in conda and pip.
- `01-text_to_image.py`: Generates dairy/pig farm images using DALL-E 3 and Stable Diffusion based on text prompts.
- `02-image_to_text.py`: Processes generated images through GPT-4o to obtain image descriptions.
- `03-image_cluster.py`: Prompt GPT-4o to cluster images into 3 categories and provide explanation.
- `04-bag_of_words.py`: Bag-of-words analysis by counting 1-gram, 2-gram, or 1 & 2 gram words in revised prompts and in text descriptions of images
- `05-generate_plot_grid.py`: Generate plot grids to visualize the megadata from generaetd images
- `06-cluster_plots.py`: Generate plot grids to summarize image clustering results
- `07-revised_prompt_analysis.py`: Generate a dataframe to show success rate of prompt revision inhibition
- `08-reduce_img_size_for_LaTex.py`: shrink down the result plot size so that it can fit into LaTex and avoid long compilation time.
- `09-dalle_eval_prompt_analysis.py`: counts among the MSCOCO prompts OpenAI used to evaluate DALL-E 3, how many are about cows and pigs in different context.
- `10-3d_plots.py`: generate 3d plots to show the percentage of images depicting indoor VS outdoor access

## Usage

Scripts should be run sequentially (00 → 01 → 02 → 03) to maintain data pipeline consistency. These scripts uses the functions built in `AI_representation_bias_in_farming` python package under `src`.
