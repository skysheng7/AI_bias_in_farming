# Results

This directory contains generated images, analysis results, and data summaries from AI representation bias in farming study.

## Directory Structure

### dall-e-3-images/

> This folder is hidden from GitHub because images are too big. Please download the images in the databse following the link to `Dataset DOI` in the README file in the root directory.
Contains DALL-E 3 generated images organized by prompt types (100 images per unique prompt):

- `basic/`: Images generated with basic farming prompts: "a [farm type]"
- `basic_country/`: Images with country-specific basic prompts: "a [farm type] in [country]"
- `basic_country_no_revise/`: Country-specific basic prompts without auto-revisions: "a [farm type] in [country].I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS:"
- `basic_no_revise/`: Basic prompts without auto-revisions: : "a [farm type].I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS:"
- `reality/`: Images from reality-focused prompts: "Please create an image that accurately represents the reality of what most [farm type] look like."
- `reality_country/`: Country-specific reality prompts: "Please create an image that accurately represents the reality of what most [farm type] look like in [country]."
- `reality_country_no_revise/`: Country-specific reality prompts with attempts to inhibit auto-revisions: "Please create an image that accurately represents the reality of what most [farm type] look like in [country].I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS:"
- `reality_no_revise/`: Reality prompts with attempts to inhibit autorevisions: "Please create an image that accurately represents the reality of what most [farm type] look like.I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS:"
- `typical/`: Images from typical farming prompts: "a typical [farm type]"
- `typical_country/`: Country-specific typical prompts: "a typical [farm type] in [country]"
- `typical_country_no_revise/`: Country-specific typical prompts without revisions: "a typical [farm type] in [country].I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS:"
- `typical_no_revise/`: Typical prompts without revisions: "a typical [farm type].I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS:"

### sd3.5-large-images/

> This folder is hidden from GitHub because images are too big. Please download the images in the databse following the link to `Dataset DOI` in the README file in the root directory.

Contains Stable Diffusion 3.5 generated images organized by prompt types (10 images per unique prompt):

- `basic/`: Images generated with basic farming prompts: "a [farm type]"
[Additional subdirectories follow same pattern as dall-e-3-images]

### plots/

Contains generated visualizations and plots from the analysis.

### megadata/

Contains analysis results and metadata in CSV format:

- `cluster_summary.csv`: [Summary statistics and key findings from image clustering analysis]

- `GPT4o_description_1_2gram_bag_of_words.csv`: [Bag of words analysis for 1-2 gram combinations from GPT-4 descriptions]

- `GPT4o_description_1_2gram_freq_summary.csv`: [Frequency summary of 1-2 gram combinations from GPT-4 descriptions]

- `GPT4o_description_1gram_bag_of_words.csv`: [Bag of words analysis for 1-gram terms from GPT-4 descriptions]

- `GPT4o_description_1gram_freq_summary.csv`: [Frequency summary of 1-gram terms from GPT-4 descriptions]

- `GPT4o_description_2gram_bag_of_words.csv`: [Bag of words analysis for 2-gram combinations from GPT-4 descriptions]

- `GPT4o_description_2gram_freq_summary.csv`: [Frequency summary of 2-gram combinations from GPT-4 descriptions]

- `image_megadata_v2.csv`: [Comprehensive metadata about generated images - version 2]

- `image_megadata.csv`: [Original comprehensive metadata about generated images]

- `revised_prompt_1_2gram_bag_of_words.csv`: [Bag of words analysis for 1-2 gram combinations from revised prompts]

- `revised_prompt_1_2gram_freq_summary.csv`: [Frequency summary of 1-2 gram combinations from revised prompts]

- `revised_prompt_1gram_bag_of_words.csv`: [Bag of words analysis for 1-gram terms from revised prompts]

- `revised_prompt_1gram_freq_summary.csv`: [Frequency summary of 1-gram terms from revised prompts]

- `revised_prompt_2gram_bag_of_words.csv`: [Bag of words analysis for 2-gram combinations from revised prompts]

- `revised_prompt_2gram_freq_summary.csv`: [Frequency summary of 2-gram combinations from revised prompts]
