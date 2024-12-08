# AI's representation bias about livestock farming

This project explores the representation bias about livestock farming in text-to-image generative AI models (DALL-E 3 and Stable Diffusion 3.5-large).

## Dataset Information

- **Title of Dataset:** Replication Data for: AI's representation bias about livestock farming
- **Paper DOI:**
- **Dataset DOI:**
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

- [Python 3.11.10](https://www.python.org/downloads/release/python-3110/)
- Python packages listed in [environment.yml](./environment.yml)

## Environment setup instructions

### Prerequisites

- Install [Conda](https://docs.conda.io/en/latest/miniconda.html) to handle dependencies.
- Install conda-lock

    ```bash
    conda install -c conda-forge conda-lock
    ```

- Install git

    ```bash
    conda install -c conda-forge git
    ```

- If you are unfamiliar with git and GitHub, please setup using this [guide](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git)

### Setting up using conda-lock

1. Clone this GitHub repository:
    In your terminal or command line, navigate to the folder (use ``` cd ```) where you want this project to live in

    ```bash
    git clone https://github.com/skysheng7/AI_bias_in_farming.git
    cd AI_bias_in_farming
    ```

2. Alternatively, create and activate the environment using `conda-lock`:

    ```bash
    conda-lock install --name ai_env --file conda-lock.yml
    conda activate ai_env
    ```

3. Install the AI_representation_bias_In_farming python package

    ```bash
    python -m pip install -e .
    ```

4. To terminate the environment

    ```bash
    conda deactivate
    ```

### Setting up using Docker

## Copyright

- Copyright © 2024 Kehan (Sky) Sheng.
- Free software distributed under the [MIT License](./LICENSE).
- Report and documentation generated in this project are distributed under the [CC BY 4.0](./LICENSE).
