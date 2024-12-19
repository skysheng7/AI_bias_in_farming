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

.PHONY: all env temp_metadata word_freq plots cluster clean-all clean-plots clean-cluster clean-temp-megadata clean-word-freq 

# dependencies in the conda environment
env: environment.yml
# image metadata master dataframe
temp_metadata: results/megadata/clustered_image_megadata.csv
# all bag-of-words analysis files
word_freq: results/megadata/GPT4o_description_1_2gram_bag_of_words.csv \
	results/megadata/GPT4o_description_1_2gram_freq_summary.csv \
	results/megadata/GPT4o_description_1gram_bag_of_words.csv \ 
	results/megadata/GPT4o_description_1gram_freq_summary.csv \
	results/megadata/GPT4o_description_2gram_bag_of_words.csv \ 
	results/megadata/GPT4o_description_2gram_freq_summary.csv \
	results/megadata/revised_prompt_1_2gram_bag_of_words.csv \
	results/megadata/revised_prompt_1_2gram_freq_summary.csv \
	results/megadata/revised_prompt_1gram_bag_of_words.csv \
	results/megadata/revised_prompt_1gram_freq_summary.csv \
	results/megadata/revised_prompt_2gram_bag_of_words.csv \
	results/megadata/revised_prompt_2gram_freq_summary.csv
# all plot grids for each prompt type
plots: results/plots/basic_dairy_dall-e-3_by_country_plot_grid.png \
results/plots/basic_dairy_sd3.5-large_by_country_plot_grid.png \
results/plots/basic_dall-e-3_plot_grid.png \ 
results/plots/basic_pig_dall-e-3_by_country_plot_grid.png \ 
results/plots/basic_pig_sd3.5-large_by_country_plot_grid.png \ 
results/plots/basic_sd3.5-large_plot_grid.png \ 
results/plots/reality_dairy_dall-e-3_by_country_plot_grid.png \ 
results/plots/reality_dairy_sd3.5-large_by_country_plot_grid.png \ 
results/plots/reality_dall-e-3_plot_grid.png \ 
results/plots/reality_pig_dall-e-3_by_country_plot_grid.png \ 
results/plots/reality_pig_sd3.5-large_by_country_plot_grid.png \ 
results/plots/reality_sd3.5-large_plot_grid.png \ 
results/plots/typical_dairy_dall-e-3_by_country_plot_grid.png \ 
results/plots/typical_dairy_sd3.5-large_by_country_plot_grid.png \ 
results/plots/typical_dall-e-3_plot_grid.png \ 
results/plots/typical_pig_dall-e-3_by_country_plot_grid.png \ 
results/plots/typical_pig_sd3.5-large_by_country_plot_grid.png \ 
results/plots/typical_sd3.5-large_plot_grid.png

# cluster images into 3 themes
cluster : results/plots/cluster_summary_dall-e-3.png \
results/plots/cluster_summary_sd-3.5.png\
results/megadata/cluster_summary.csv \
results/megadata/image_megadata_post_manual_fix.csv 

# all contains all the plots and cluster summary files
all : results/plots/basic_dairy_dall-e-3_by_country_plot_grid.png \
results/plots/basic_dairy_sd3.5-large_by_country_plot_grid.png \
results/plots/basic_dall-e-3_plot_grid.png \ 
results/plots/basic_pig_dall-e-3_by_country_plot_grid.png \ 
results/plots/basic_pig_sd3.5-large_by_country_plot_grid.png \ 
results/plots/basic_sd3.5-large_plot_grid.png \ 
results/plots/reality_dairy_dall-e-3_by_country_plot_grid.png \ 
results/plots/reality_dairy_sd3.5-large_by_country_plot_grid.png \ 
results/plots/reality_dall-e-3_plot_grid.png \ 
results/plots/reality_pig_dall-e-3_by_country_plot_grid.png \ 
results/plots/reality_pig_sd3.5-large_by_country_plot_grid.png \ 
results/plots/reality_sd3.5-large_plot_grid.png \ 
results/plots/typical_dairy_dall-e-3_by_country_plot_grid.png \ 
results/plots/typical_dairy_sd3.5-large_by_country_plot_grid.png \ 
results/plots/typical_dall-e-3_plot_grid.png \ 
results/plots/typical_pig_dall-e-3_by_country_plot_grid.png \ 
results/plots/typical_pig_sd3.5-large_by_country_plot_grid.png \ 
results/plots/typical_sd3.5-large_plot_grid.png \
results/plots/cluster_summary_dall-e-3.png \
results/plots/cluster_summary_sd-3.5.png\
results/megadata/cluster_summary.csv \
results/megadata/image_megadata_post_manual_fix.csv 

# Specify version numbers for each dependency used in the current conda environment in environment.yml
environment.yml : scripts/00-update_enviroment_yml.py
	python scripts/00-update_enviroment_yml.py \
		--root_dir="." \
		--env_name="ai_env"
        
# Step 1: Generate batahces of images based on text descriptions
results/megadata/raw_image_megadata.csv : scripts/01-text_to_image.py
	python scripts/01-text_to_image.py \
		--start_index=1 \
		--total_image_num=10 \
		--model="dall-e-3"
	cp results/megadata/image_megadata.csv results/megadata/raw_image_megadata.csv
        
# Step 2: Generate text descriptions for images in batch
results/megadata/described_image_megadata.csv : scripts/02-image_to_text.py results/megadata/raw_image_megadata.csv
	python scripts/02-image_to_text.py \
		--start_index=0 \
		--end_index=None
	cp results/megadata/image_megadata.csv results/megadata/described_image_megadata.csv

# Step 3: Prompt GPT4o to utomatically cluster images into 3 categories based on what is depicted in the image
results/megadata/clustered_image_megadata.csv : scripts/03-image_cluster.py results/megadata/described_image_megadata.csv
	python scripts/03-image_cluster.py \
		--start_index=0 \
		--end_index=None
	cp results/megadata/image_megadata.csv results/megadata/clustered_image_megadata.csv

# run bag-of-words analysis
results/megadata/GPT4o_description_1_2gram_bag_of_words.csv \
results/megadata/GPT4o_description_1_2gram_freq_summary.csv \
results/megadata/GPT4o_description_1gram_bag_of_words.csv \ 
results/megadata/GPT4o_description_1gram_freq_summary.csv \
results/megadata/GPT4o_description_2gram_bag_of_words.csv \ 
results/megadata/GPT4o_description_2gram_freq_summary.csv \
results/megadata/revised_prompt_1_2gram_bag_of_words.csv \
results/megadata/revised_prompt_1_2gram_freq_summary.csv \
results/megadata/revised_prompt_1gram_bag_of_words.csv \
results/megadata/revised_prompt_1gram_freq_summary.csv \
results/megadata/revised_prompt_2gram_bag_of_words.csv \
results/megadata/revised_prompt_2gram_freq_summary.csv : scripts/04-bag_of_words.py \
 results/megadata/image_megadata.csv
	python scripts/04-bag_of_words.py

# generate plot grids for each model, each prompt type and farm type
results/plots/basic_dairy_dall-e-3_by_country_plot_grid.png \
results/plots/basic_dairy_sd3.5-large_by_country_plot_grid.png \
results/plots/basic_dall-e-3_plot_grid.png \ 
results/plots/basic_pig_dall-e-3_by_country_plot_grid.png \ 
results/plots/basic_pig_sd3.5-large_by_country_plot_grid.png \ 
results/plots/basic_sd3.5-large_plot_grid.png \ 
results/plots/reality_dairy_dall-e-3_by_country_plot_grid.png \ 
results/plots/reality_dairy_sd3.5-large_by_country_plot_grid.png \ 
results/plots/reality_dall-e-3_plot_grid.png \ 
results/plots/reality_pig_dall-e-3_by_country_plot_grid.png \ 
results/plots/reality_pig_sd3.5-large_by_country_plot_grid.png \ 
results/plots/reality_sd3.5-large_plot_grid.png \ 
results/plots/typical_dairy_dall-e-3_by_country_plot_grid.png \ 
results/plots/typical_dairy_sd3.5-large_by_country_plot_grid.png \ 
results/plots/typical_dall-e-3_plot_grid.png \ 
results/plots/typical_pig_dall-e-3_by_country_plot_grid.png \ 
results/plots/typical_pig_sd3.5-large_by_country_plot_grid.png \ 
results/plots/typical_sd3.5-large_plot_grid.png : scripts/05-generate_plot_grid.py \
results/megadata/revised_prompt_2gram_freq_summary.csv \
results/megadata/GPT4o_description_2gram_freq_summary.csv \
results/megadata/image_megadata.csv
	python scripts/05-generate_plot_grid.py

# visualize the quality of automatic clustering, and the quality of manual correction
results/plots/cluster_summary_dall-e-3.png \
results/plots/cluster_summary_sd-3.5.png \
results/megadata/cluster_summary.csv \
results/megadata/image_megadata_post_manual_fix.csv : scripts/06-cluster_plots.py \
results/megadata/outlier_image_manual_correction.csv \
results/megadata/image_megadata.csv
	python scripts/06-cluster_plots.py

# clean up the analysis files
clean-plots :
    rm -f results/plots/basic_dairy_dall-e-3_by_country_plot_grid.png \
		results/plots/basic_dairy_sd3.5-large_by_country_plot_grid.png \
		results/plots/basic_dall-e-3_plot_grid.png \ 
		results/plots/basic_pig_dall-e-3_by_country_plot_grid.png \ 
		results/plots/basic_pig_sd3.5-large_by_country_plot_grid.png \ 
		results/plots/basic_sd3.5-large_plot_grid.png \ 
		results/plots/reality_dairy_dall-e-3_by_country_plot_grid.png \ 
		results/plots/reality_dairy_sd3.5-large_by_country_plot_grid.png \ 
		results/plots/reality_dall-e-3_plot_grid.png \ 
		results/plots/reality_pig_dall-e-3_by_country_plot_grid.png \ 
		results/plots/reality_pig_sd3.5-large_by_country_plot_grid.png \ 
		results/plots/reality_sd3.5-large_plot_grid.png \ 
		results/plots/typical_dairy_dall-e-3_by_country_plot_grid.png \ 
		results/plots/typical_dairy_sd3.5-large_by_country_plot_grid.png \ 
		results/plots/typical_dall-e-3_plot_grid.png \ 
		results/plots/typical_pig_dall-e-3_by_country_plot_grid.png \ 
		results/plots/typical_pig_sd3.5-large_by_country_plot_grid.png \ 
		results/plots/typical_sd3.5-large_plot_grid.png

clean-cluster :
	rm -f results/plots/cluster_summary_dall-e-3.png \
		results/plots/cluster_summary_sd-3.5.png\
		results/megadata/cluster_summary.csv \
		results/megadata/image_megadata_post_manual_fix.csv 

clean-temp-megadata :
	rm -f results/megadata/clustered_image_megadata.csv

clean-word-freq :
	rm -f results/megadata/GPT4o_description_1_2gram_bag_of_words.csv \
		results/megadata/GPT4o_description_1_2gram_freq_summary.csv \
		results/megadata/GPT4o_description_1gram_bag_of_words.csv \ 
		results/megadata/GPT4o_description_1gram_freq_summary.csv \
		results/megadata/GPT4o_description_2gram_bag_of_words.csv \ 
		results/megadata/GPT4o_description_2gram_freq_summary.csv \
		results/megadata/revised_prompt_1_2gram_bag_of_words.csv \
		results/megadata/revised_prompt_1_2gram_freq_summary.csv \
		results/megadata/revised_prompt_1gram_bag_of_words.csv \
		results/megadata/revised_prompt_1gram_freq_summary.csv \
		results/megadata/revised_prompt_2gram_bag_of_words.csv \
		results/megadata/revised_prompt_2gram_freq_summary.csv

clean-all : clean-plots \
    clean-cluster \
	clean-temp-megadata \
	clean-word-freq 
