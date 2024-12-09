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

- [Docker](https://docs.docker.com/get-started/)

## Usage

### Setup

> If you are using Windows or Mac, make sure [Docker Desktop](https://docs.docker.com/get-started/) is running.

1. Clone this GitHub repository.

- In your terminal or command line, navigate to the folder (use ``` cd ```) where you want this project to live in

    ```bash
    git clone https://github.com/skysheng7/AI_bias_in_farming.git
    cd AI_bias_in_farming
    ```

- If you are unfamiliar with git and GitHub, please setup using this [guide](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git)

### Running the analysis

1. Navigate to the root of this project on your computer using the
   command line and enter the following command:

```
docker compose up
```

2. In the terminal, look for a URL that starts with
`http://127.0.0.1:8888/lab?token=`
(for an example, see the highlighted text in the terminal below).
Copy and paste that URL into your browser. You should see jupyter lab running in your browser

<img src="img/docker_demo.png" width=400>

3. To run the analysis,
navigate to "notebooks" folder on your left hand side, and double click to open any jupyter notebook that you wish to run.

### Clean up

1. To shut down the container and clean up the resources,
type `Cntrl` + `C` in the terminal
where you launched the container, and then type `docker compose rm`

## Developer notes

### Developer dependencies

- `conda` (version 24.11.0 or higher)
- `conda-lock` (version 2.5.7 or higher)

## Copyright

- Copyright © 2024 Kehan (Sky) Sheng.
- Free software distributed under the [MIT License](./LICENSE).
- Report and documentation generated in this project are distributed under the [CC BY 4.0](./LICENSE).
