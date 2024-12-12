# AI Representation Bias in Farming

This python package contains modules for analyzing representation bias in AI-generated farm images.

## Module Descriptions

- `utils.py`: Contains shared utility functions used across different modules.
- `update_enviroment_yml.py`: automatically append version number for each package name installed in conda and pip.
- `module0_dalle3.py`: Handles DALL-E 3 image generation and processing functionalities.
- `module1_sd.py`: Contains Stable Diffusion image generation and processing utilities.
- `module2_GPT4o.py`: Implements GPT-4 Vision image analysis, generate descriptions for each image and cluster images.
- `module3_word_freq_count.py`: Performs word frequency analysis on image descriptions and revised prompts.
- `module4_plot_generation.py`: Creates visualizations and plots for analysis results.
- `module5_extract_pic_feature_words.py`: Extracts and analyzes key features and words from image descriptions.

Note: `__init__.py` enables Python package functionality, `__pycache__` stores Python bytecode cache, and `py.typed` indicates type hint support.
