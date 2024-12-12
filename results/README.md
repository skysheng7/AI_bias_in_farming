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

#### Common Columns Across Files

Many files share these base columns:

- `file`: Image file name, each image has an unique identifier
- `generation_type`: Type of prompt used
- `country`: Country specified in the prompt
- `farm_type`: Type of farm, dairy or pig
- `prompt`: Original prompt used
- `revised_prompt`: Automatically modified version of the prompt (only for dall-e-3)
- `model`: AI model used (e.g., dall-e-3)
- `size`: Image dimensions
- `quality`: Image quality setting
- `response_format`: Format of the response
- `finish_reason`: Reason for completion
- `description_model`: Model used for description

#### Detailed File Descriptions

- `cluster_summary.csv`:
Provides a statistical overview of what's the percentage of indoor/outdoor depictions we using differernt prompts and in different countries.
  - generation_t: prompt type
  - country: Country specified
  - farm_type: Type of farming, dairy of pig farm
  - prompt_model: Model used
  - total_rows: Total number of images
  - indoor_sum/outdoor_sum/other_sum: Counts of images depicting animals housed indoor, outdoor, or other
  - indoor_pct/outdoor_pct/other_pct: Percentages of images depicting animals housed indoor, outdoor, or other

- `image_megadata.csv`:
Master datasets containing all image metadata, text descriptions, and analysis results for each generated image.
  - All common columns +
  - GPT4o_description: GPT-4o's description of the image
  - GPT4o_description_token_count: Token count of description
  - GPT4o_prompt: Prompt used to generate text description for each image using GPT-4o
  - GPT4o_image_resolution: Image resolution details
  - GPT4o_temperature: Temperature setting
  - GPT4o_system_fingerprint: Unique system identifier
  - cluster_model: Model used for clustering
  - GPT4o_cluster: GPT4o assigned cluster
  - GPT4o_cluster_explanation: GPT4o's explanation of clustering
  - GPT4o_cluster_token_count: Token count for cluster explanation
  - GPT4o_cluster_prompt: Prompt used for clustering
  - GPT4o_cluster_system_fingerprint: Unique system identifier for clustering

#### GPT-4 Description Analysis Files

**Bag of Words Files** :

- `GPT4o_description_1gram_bag_of_words.csv`: Single words that show up in > 20 image descriptions
- `GPT4o_description_2gram_bag_of_words.csv`: Two-word phrases that show up in > 20 image descriptions
- `GPT4o_description_1_2gram_bag_of_words.csv`: Both single words and two-word phrases that show up in > 20 image descriptions

These files contain:

- All common columns
- GPT-4o description metadata (description text, token count, prompt, resolution, temperature, system fingerprint)
- Binary columns (0/1) for each word/phrase indicating presence in the description

**Frequency Summary Files** (aggregated counts):

- `GPT4o_description_1gram_freq_summary.csv`: Single words only
- `GPT4o_description_2gram_freq_summary.csv`: Two-word phrases only
- `GPT4o_description_1_2gram_freq_summary.csv`: Both single words and two-word phrases

These files contain:

- Grouping columns: generation_type, country, farm_type, model
- Count columns: Number of images (out of 100 per prompt type) containing each word/phrase
- top_20_words: List of the 20 most frequently occurring phrases with their counts in parentheses

#### Revised Prompt Analysis Files

Follow the same pattern as GPT-4 Description files, but analyze the revised prompts instead:

**Bag of Words Files**:

- `revised_prompt_1gram_bag_of_words.csv`
- `revised_prompt_2gram_bag_of_words.csv`
- `revised_prompt_1_2gram_bag_of_words.csv`

**Frequency Summary Files**:

- `revised_prompt_1gram_freq_summary.csv`
- `revised_prompt_2gram_freq_summary.csv`
- `revised_prompt_1_2gram_freq_summary.csv`
