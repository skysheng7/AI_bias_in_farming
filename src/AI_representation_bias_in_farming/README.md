# AI Representation Bias in Farming

This python package contains modules for generating images of dairy and pig farms, and analyzing representation bias in these AI models.

## Module Descriptions

- `utils.py`: Contains shared utility functions used across different modules.
- `module0_dalle3.py`: functionalities related to generating images from text using DALL-E 3
- `module1_sd.py`: functionalities related to generating images from text using Stable Diffusion 3.5-large
- `module2_GPT4o.py`: Implements GPT-4o image analysis, generate descriptions for each image and cluster images.
- `module3_word_freq_count.py`: Performs word frequency analysis on image descriptions and revised prompts.
- `module4_plot_generation.py`: Creates visualizations and plots for analysis results.
- `module5_extract_pic_feature_words.py`: Extracts and analyzes key features and words from image descriptions.
- `module6_confidence_interval.py`: Calculates confidence interval for each point estimate.
- `module7-3d_plots.py`: Generates 3d plots to show the percentage of images depicting indoor VS outdoor access

Note: `__init__.py` enables Python package functionality, `__pycache__` stores Python bytecode cache, and `py.typed` indicates type hint support.
