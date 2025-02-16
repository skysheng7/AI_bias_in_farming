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
#
# example usage:
# make env

.PHONY: all env bag_of_words plots 3d_plots cluster clean-all clean-plots clean-temp-megadata clean-cluster clean-bag-of-words

# dependencies in the conda environment
env: environment.yml

# all bag-of-words analysis files
bag_of_words: results/megadata/GPT4o_description_1_2gram_bag_of_words.csv\
	results/megadata/GPT4o_description_1_2gram_freq_summary.csv\
	results/megadata/GPT4o_description_1gram_bag_of_words.csv\
	results/megadata/GPT4o_description_1gram_freq_summary.csv\
	results/megadata/GPT4o_description_2gram_bag_of_words.csv\
	results/megadata/GPT4o_description_2gram_freq_summary.csv\
	results/megadata/revised_prompt_1_2gram_bag_of_words.csv\
	results/megadata/revised_prompt_1_2gram_freq_summary.csv\
	results/megadata/revised_prompt_1gram_bag_of_words.csv\
	results/megadata/revised_prompt_1gram_freq_summary.csv\
	results/megadata/revised_prompt_2gram_bag_of_words.csv\
	results/megadata/revised_prompt_2gram_freq_summary.csv

# all plot grids for each prompt type
plots: results/plots/basic_dairy_dall-e-3_by_country_plot_grid.png\
	results/plots/basic_dairy_sd3.5-large_by_country_plot_grid.png\
	results/plots/basic_dall-e-3_plot_grid.png\
	results/plots/basic_pig_dall-e-3_by_country_plot_grid.png\
	results/plots/basic_pig_sd3.5-large_by_country_plot_grid.png\
	results/plots/basic_sd3.5-large_plot_grid.png\
	results/plots/reality_dairy_dall-e-3_by_country_plot_grid.png\
	results/plots/reality_dairy_sd3.5-large_by_country_plot_grid.png\
	results/plots/reality_dall-e-3_plot_grid.png\
	results/plots/reality_pig_dall-e-3_by_country_plot_grid.png\
	results/plots/reality_pig_sd3.5-large_by_country_plot_grid.png\
	results/plots/reality_sd3.5-large_plot_grid.png\
	results/plots/typical_dairy_dall-e-3_by_country_plot_grid.png\
	results/plots/typical_dairy_sd3.5-large_by_country_plot_grid.png\
	results/plots/typical_dall-e-3_plot_grid.png\
	results/plots/typical_pig_dall-e-3_by_country_plot_grid.png\
	results/plots/typical_pig_sd3.5-large_by_country_plot_grid.png\
	results/plots/typical_sd3.5-large_plot_grid.png\

# cluster images into 3 themes
cluster: results/cluster/* \
    results/cluster_post_manual_fix/* \
    results/plots/cluster_summary_dall-e-3.png\
	results/plots/cluster_summary_sd-3.5.png\
	results/megadata/cluster_summary.csv\
	results/megadata/image_megadata_post_manual_fix.csv

revision: results/megadata/revised_prompt_count.csv

3d_plots: results/plots/3d_general_plot.png\
	results/plots/3d_country_plot_dairy.png\
	results/plots/3d_country_plot_pig.png

# all contains all the plots and cluster summary files
all: plots\
	results/megadata/cluster_summary.csv\
	results/megadata/image_megadata_post_manual_fix.csv\
	results/plots/cluster_summary_dall-e-3.png\
	results/plots/cluster_summary_sd-3.5.png\
	results/megadata/revised_prompt_count.csv\
	openai_eval_dalle\
	3d_plots


# Specify version numbers for each dependency used in the current conda environment
environment.yml: scripts/00-update_enviroment_yml.py
	python scripts/00-update_enviroment_yml.py\
		--root_dir="."\
		--env_name="ai_env"

# Run bag-of-words analysis
results/megadata/GPT4o_description_1_2gram_bag_of_words.csv\
results/megadata/GPT4o_description_1_2gram_freq_summary.csv\
results/megadata/GPT4o_description_1gram_bag_of_words.csv\
results/megadata/GPT4o_description_1gram_freq_summary.csv\
results/megadata/GPT4o_description_2gram_bag_of_words.csv\
results/megadata/GPT4o_description_2gram_freq_summary.csv\
results/megadata/revised_prompt_1_2gram_bag_of_words.csv\
results/megadata/revised_prompt_1_2gram_freq_summary.csv\
results/megadata/revised_prompt_1gram_bag_of_words.csv\
results/megadata/revised_prompt_1gram_freq_summary.csv\
results/megadata/revised_prompt_2gram_bag_of_words.csv\
results/megadata/revised_prompt_2gram_freq_summary.csv: scripts/04-bag_of_words.py results/megadata/image_megadata.csv
	python scripts/04-bag_of_words.py

# Generate plot grids for each model, each prompt type and farm type
results/plots/basic_dairy_dall-e-3_by_country_plot_grid.png\
results/plots/basic_dairy_sd3.5-large_by_country_plot_grid.png\
results/plots/basic_dall-e-3_plot_grid.png\
results/plots/basic_pig_dall-e-3_by_country_plot_grid.png\
results/plots/basic_pig_sd3.5-large_by_country_plot_grid.png\
results/plots/basic_sd3.5-large_plot_grid.png\
results/plots/reality_dairy_dall-e-3_by_country_plot_grid.png\
results/plots/reality_dairy_sd3.5-large_by_country_plot_grid.png\
results/plots/reality_dall-e-3_plot_grid.png\
results/plots/reality_pig_dall-e-3_by_country_plot_grid.png\
results/plots/reality_pig_sd3.5-large_by_country_plot_grid.png\
results/plots/reality_sd3.5-large_plot_grid.png\
results/plots/typical_dairy_dall-e-3_by_country_plot_grid.png\
results/plots/typical_dairy_sd3.5-large_by_country_plot_grid.png\
results/plots/typical_dall-e-3_plot_grid.png\
results/plots/typical_pig_dall-e-3_by_country_plot_grid.png\
results/plots/typical_pig_sd3.5-large_by_country_plot_grid.png\
results/plots/typical_sd3.5-large_plot_grid.png: scripts/05-generate_plot_grid.py\
results/megadata/revised_prompt_2gram_freq_summary.csv\
results/megadata/GPT4o_description_2gram_freq_summary.csv\
results/megadata/image_megadata.csv
	python scripts/05-generate_plot_grid.py

# Visualize the quality of automatic clustering and manual correction
results/cluster/* \
results/cluster_post_manual_fix/* \
results/plots/cluster_summary_dall-e-3.png\
results/plots/cluster_summary_sd-3.5.png\
results/megadata/cluster_summary.csv\
results/megadata/image_megadata_post_manual_fix.csv: scripts/06-cluster_plots.py\
results/megadata/outlier_image_manual_correction.csv\
results/megadata/image_megadata.csv
	python scripts/06-cluster_plots.py

# count the success rate of prompt revision inhibition
results/megadata/revised_prompt_count.csv: scripts/07-revised_prompt_analysis.py\
results/megadata/image_megadata.csv
	python scripts/07-revised_prompt_analysis.py\

# analyze the prompts OpenAI used to evaluate the performance of DALL-E 3 for prompt
# following. 
openai_eval_dalle: scripts/09-dalle_eval_prompt_analysis.py\
dalle3_eval_data/8k_coco.txt
	python scripts/09-dalle_eval_prompt_analysis.py\

# generate 3d plots
results/plots/3d_general_plot.png\
results/plots/3d_country_plot_dairy.png\
results/plots/3d_country_plot_pig.png: scripts/10-3d_plots.py\
results/megadata/cluster_summary.csv\
results/megadata/image_megadata_post_manual_fix.csv\
	python scripts/10-3d_plots.py\


# Clean up the analysis files
clean-plots:
	rm -f plots 3d_plots

clean-image-eda:
	rm -rf results/cluster results/cluster_post_manual_fix

clean-cluster:
	rm -f results/plots/cluster_summary_dall-e-3.png\
		results/plots/cluster_summary_sd-3.5.png\
		results/megadata/cluster_summary.csv\
		results/megadata/image_megadata_post_manual_fix.csv

clean-bag-of-words:
	rm -f results/megadata/GPT4o_description_1_2gram_bag_of_words.csv\
		results/megadata/GPT4o_description_1_2gram_freq_summary.csv\
		results/megadata/GPT4o_description_1gram_bag_of_words.csv\
		results/megadata/GPT4o_description_1gram_freq_summary.csv\
		results/megadata/GPT4o_description_2gram_bag_of_words.csv\
		results/megadata/GPT4o_description_2gram_freq_summary.csv\
		results/megadata/revised_prompt_1_2gram_bag_of_words.csv\
		results/megadata/revised_prompt_1_2gram_freq_summary.csv\
		results/megadata/revised_prompt_1gram_bag_of_words.csv\
		results/megadata/revised_prompt_1gram_freq_summary.csv\
		results/megadata/revised_prompt_2gram_bag_of_words.csv\
		results/megadata/revised_prompt_2gram_freq_summary.csv

clean-revision:
	rm -f results/megadata/revised_prompt_count.csv

clean-all: clean-plots clean-image-eda clean-cluster clean-bag-of-words clean-revision