# AI's representation bias about livestock farming

## Abstract

ChatGPT’s text-to-image generative model (DALL-E 3) shows a systematic bias toward depicting cows grazing and pigs rooting in mud when prompted about dairy and pig farms. However, when its automatic prompt revision was inhibited, images shifted to modern reality of livestock farming: animals housed in metal pens on concrete floors. This suggests the base model possesses knowledge of modern farming reality but chose to systematically depict pastoral imagery.

![Figure 1](results/plots/basic_dall-e-3_plot_grid.png)

## Dataset Information

- **Title of Dataset:** Replication Data for: AI's representation bias about livestock farming
- **Paper DOI:**
- **Dataset DOI:** <https://doi.org/10.5683/SP3/EAWR6D>
- **Dataset Created:** 2024-10-01
- **Created by:** Kehan (Sky) Sheng
- **Contact Email:** <skysheng7@gmail.com>

## Contributors

- **Principal Investigator:** Marina von Keyserlingk  
  - ORCID: 0000-0002-1427-3152  
  - Affiliation: University of British Columbia  
  - Email: <nina@mail.ubc.ca>

- **Contributor:** Kehan Sheng  
  - ORCID: 0000-0001-6442-5284  
  - Affiliation: University of British Columbia  
  - Email: <skysheng7@gmail.com>

- **Contributor:** Frank Tuyttens
  - ORCID: 0000-0001-6442-5284
  - Affiliation_1: Fisheries and Food (ILVO)
  - Affiliation_2: Ghent University
  - Email: <frank.tuyttens@ilvo.vlaanderen.be>

## Dependencies

- [Docker](https://docs.docker.com/get-started/)

## Usage Guide

### Starting the Virtual Environment and Run Analysis

> Important: For Windows and Mac users, ensure [Docker Desktop](https://docs.docker.com/get-started/) is actively running before proceeding.

1. First, obtain a copy of this GitHub repository on your local machine.

    - Open your terminal or command prompt and navigate to your preferred project directory using the `cd` command. Then execute these commands:

        ```bash
        git clone https://github.com/skysheng7/AI_bias_in_farming.git
        cd AI_bias_in_farming
        ```

    - New to git and GitHub? Please follow the official [setup guide](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git) to get started.

2. Download all images from this [database](https://doi.org/10.5683/SP3/EAWR6D) as a zip file. Unzip it, copy the `results/dall-e-3-images`, `results/sd3.5-large-images`, `results/cluster`, `results/cluster_post_manual_fix` folders and paste it into the `results` directory of the local copy of this repository.

3. If you only want to analyze the existing image dataset without generating new images, you can create a placeholder .env file. This ensures the Docker container runs properly without real API keys. Create a file named `.env` (e.g., by running`nano .env` in your terminal) in your project's root directory and add these placeholder values:

        ```
        OPENAI_API_KEY=test
        stable_diffusion_key=test
        ```

4. In your terminal, ensure you're in the project's root directory, and have created your own `.env` file, then launch the Docker container:

    ```
    docker compose up
    ```

    Note: To maintain clean and modular code, I've packaged commonly used functions into a local Python package `src/AI_representation_bias_in_farming`. This package has been pre-installed in the Docker environment as a dynamic version, meaning you can modify the source code and see changes without reinstallation (sometimes you may need to close the project and reopen it to see changes).

5. Watch your terminal output for a unique URL beginning with
`http://127.0.0.1:8888/lab?token=`.
You'll see it displayed as highlighted in the example screenshot below.
Copy this URL and open it in your web browser to access the Jupyter Lab interface. In Jupyter Lab, you'll find all the scripts and code folders on the left sidebar.

    <img src="img/docker_demo.png" width=400>

6. Open Terminal in JupyterLab web browser interface:
   - Look for the "+" icon in JupyterLab's launcher
   - Click "Terminal" from the options
  
7. Remove all generated summary files and reset the analysis

    ```
    make clean-all
    ```

8. To execute the entire analysis pipeline, processing all steps from data visualization to analysis. WARNING: this takes about 30 minutes to run on a regular laptop (M1 macbook pro).

    ```
    make all
    ```

    Note: This code only runs the analysis part after all images are generated and image metadata are collected, it does not run the text-to-image, image-to-text, and GPT4o image clustering part because API calls cost money.

### Generating New Images and Run Generative Models

If you want to generate new images, play with text-to-image(T2I) or image-to-text(I2T) models yourself (note: this requires API keys and costs money):

1. You'll need to set up API authentication:

    - Create a file named `.env` in the project's root directory. Add your API keys to this file in the following format:

        ```
        OPENAI_API_KEY=your_openai_key_here
        stable_diffusion_key=your_stability_key_here
        ```

    - To obtain API keys:
        - For OpenAI (DALL-E 3): Follow the [OpenAI API setup guide](https://platform.openai.com/docs/quickstart)
        - For Stable Diffusion: Register and get your key from the [Stability AI platform](https://platform.stability.ai/docs/getting-started)

    ⚠️ **Important**: These steps require API keys for OpenAI services and will incur charges. The `.env` file contains sensitive information and is automatically ignored by git (listed in .gitignore) to protect your API keys.

2. Generate images using text-to-image models (WARNING: This may take 10 minutes to a couple hours to run depending on how many images you wish to generate):

   ```bash
   python scripts/01-text_to_image.py --start_index=300 --total_image_num=2 --model="dall-e-3"
   ```

   This will create multiple new images based on text prompts using DALL-E 3 (n images per unique prompts, we have 48 unique prompts in total, with n=total_image_num). Generated images will be in `results/dall-e-3-images`, image metadata will be stored in `results/megadata/image_megadata.csv`.

   ```bash
   python scripts/01-text_to_image.py --start_index=300 --total_image_num=2 --model="sd3.5-large"
   ```

   This will create new images based on text prompts using Stable Diffusion 3.5-large (n images per unique prompts, we have 48 unique prompts in total, with n=total_image_num). Generated images will be in `results/sd3.5-large-images`, image metadata will be stored in `results/megadata/image_megadata.csv`.

3. Generate text descriptions for the images:

   ```bash
   python scripts/02-image_to_text.py --start_index=5280 --end_index=None
   ```

   This will use GPT-4V to create detailed descriptions of each image starting at row=(start_index+2) in `results/megadata/image_megadata.csv`. `results/megadata/image_megadata.csv` will be updated to include a text description for each image

4. Automatically cluster the images:

   ```bash
   python scripts/03-image_cluster.py --start_index=5280 --end_index=None
   ```

   This will use GPT-4 to categorize images into three thematic clusters, starting the categorization at row=(start_index+2) in `results/megadata/image_megadata.csv`. `results/megadata/image_megadata.csv` will be updated to include a cluster label for each image.

### Project Cleanup

1. When you're finished, properly shut down the container and remove associated resources:
Press `Cntrl` + `C` in your terminal where the container is running, then execute `docker compose rm`

## Collaboration Welcome

If you find this research valuable or interesting, please consider:

- Starring this repository to help others discover this work
- Creating a fork if you'd like to build upon or extend this research
- Opening issues or pull requests if you have suggestions for improvements

## Questions Welcome

I'm committed to making my research fully reproducible and accessible to all. If you encounter any difficulties running the code or need clarification on any part of this project, I welcome you to reach out directly at <skysheng7@gmail.com>.
Open and reproducible data science workflow is my passion. Your ability to understand and build upon this work matters to me, and I'm here to support.

I'll help create a clear repository structure section. Since you want a brief explanation for each item, I'll integrate this into your README.md:

## Repository Structure

This repository is organized as follows:

- `img/`: Contains an images to demo the use of docker, inserted in README
- `results/`: Stores generated images (in database), and image metadata results from running the "scripts" and "src"
- `scripts/`: Houses the main Python scripts that generate image from text, and generate text from images
- `src/`: Contains the source code for our local Python package with helper functions and utilities
- `tests/`: Contains tests written for functions
- `Makefile`: GNU make file to streamline reproducing my analysis from scratch
- `CODE_OF_CONDUCT.md`: Guidelines for maintaining a welcoming and inclusive community atmosphere
- `conda-linux-64.lock`: Conda-lock file that specifies exact versions of Python dependencies for reproducible environments. This conda-lock file is explicitly solved for linux-64 operating systems
- `conda-lock.yml`: Conda-lock file that specifies exact versions of Python dependencies for reproducible environments. This conda-lock file is solved for the following operating systems: ['linux-64', 'osx-64', 'osx-arm64', 'win-64']
- `CONTRIBUTING.md`: Instructions and guidelines for contributing to this project
- `docker-compose.yml`: Docker configuration for setting up the analysis environment
- `Dockerfile`: Instructions for building the project's Docker container
- `environment.yml`: Conda environment specifications for managing Python dependencies
- `LICENSE`: Legal terms under which this project's code can be used and distributed
- `pyproject.toml`: Python project metadata and build system requirements for Python packaging

## Developer notes

### Developer Dependencies

- `conda` (>= 24.11.0)
- `conda-lock` (>= 2.5.7)

### Instructions for Adding New Dependencies

1. Open your terminal locally, direct to the root directory. Make sure you have conda and conda-lock installed on your local computer.
2. Create a conda environment called "ai_env" using the "conda-lock.yml" by running in your terminal:

    ```
    conda-lock install --name ai_env conda-lock.yml
    ```

3. Activate the conda environment

    ```
    conda activate ai_env
    ```

4. Use conda to install new packages (e.g., `conda install {NEW-PACKAGE-NAME}`). If you are installing a new package that is only available on PyPI (e.g., `pip install {NEW-PACKAGE-NAME}`), conda does not track pip-installed packages, you need to append a new "RUN" command to pip install that package (with version number; e.g., `RUN pip install openai==1.57.0`) at the end of the Dockerfile (living at the root of this directory).
5. At root directory, update environment.yml using:

    ```
    conda env export --from-history > environment.yml 
    ```

6. Automatically append dependency version numbers to each of the packages you installed. The conda virtual environment I created is called "ai_env", if you are using another name, please change --env_name:

    ```
    python scripts/00-update_enviroment_yml.py --root_dir="." --env_name="ai_env"
    ```

7. Use Conda-lock to solve and lock the updated environment. I'm using Linux-64 because that's the operating system of my docker image

    ```
    conda-lock lock --file environment.yml
    conda-lock -k explicit --file environment.yml -p linux-64
    ```

8. Re-build the docker image in root directory and use the updated container locally. Please replace {YOUR-IMAGE-NAME} with some meaningful name for your local container. If you think this new dependency should be included in my repository, please make a pull request and I'll push this new image on my docker hub.

    ```
    docker build --tag {YOUR-IMAGE-NAME} .
    ```

    Note: If you are using a M1-M3 MacBook, you may have trouble with docker build. This is a known problem when emulating x86 architectures on ARM-based systems. To solve this, you need to enable QEMU-based emulation and use docker buildx:
    - Confirm That You Have Docker Desktop with Buildx Support (you should see version information after running this command below)

        ```
        docker buildx version
        ```

    - Create a new buildx builder

        ```
        docker buildx create --name mybuilder
        ```

    - Use the new builder

        ```
        docker buildx use mybuilder
        ```

    - Initialize the builder and make sure QEMU is active

        ```
        docker buildx inspect --bootstrap
        ```

    - Building docker buildx with Emulation

        ```
        docker buildx build --platform linux/amd64 -t {YOUR-IMAGE-NAME} --load .
        ```

9. Edit the docker-compose.yml file (living at root of directory), replace the image name with {YOUR-IMAGE-NAME}.
    For example, replace "image: skysheng7/ai_bias:d077bb3" with "image: {YOUR-IMAGE-NAME}"

10. Running the docker image you just built at root directory

    ```
    docker compose up
    ```

## Copyright

- Copyright © 2024 Kehan (Sky) Sheng.
- Free software distributed under the [MIT License](./LICENSE).
- Report and documentation generated in this project are distributed under the [CC BY 4.0](./LICENSE).
