# Makefile
# Kehan Sky Sheng, 2024-12-18
#
# This GNU Make file allows you to rerun my analysis on images that I generated.
# You also have the option to create new images using text-toimage generative models,
# and create descriptions for images if you would like. 
#
# WARNING: if you want to run 01-text_to_image.py, 02-image_to_text.py and 03-image_cluster.py
#		   yourself, you need to set up the API key as instructed in the README (living in the root
#		   directory), because it costs money to run these models.

# example usage:
# make env

.PHONY: env

# dependencies in the conda environment
env: environment.yml

# Specify version numbers for each dependency used in the current conda environment in environment.yml
environment.yml : scripts/00-update_enviroment_yml.py
	python scripts/00-update_enviroment_yml.py \
		--root_dir="." \
		--env_name="ai_env"
        
# Generate batahces of images based on text descriptions
results/megadata/image_megadata.csv : scripts/01-text_to_image.py
	python scripts/01-text_to_image.py \
		--start_index=1 \
		--total_image_num=10 \
		--model="dall-e-3"
        
# Generate text descriptions for images in batch
results/megadata/image_megadata.csv : scripts/02-image_to_text.py
	python scripts/02-image_to_text.py \
		--start_index=0 \
		--end_index=None

# Automatically cluster images into 3 categories based on what is depicted in the image
results/megadata/image_megadata.csv : scripts/03-image_cluster.py
	python scripts/03-image_cluster.py \
		--start_index=0 \
		--end_index=None
        