# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Example Contributions

You can contribute in many ways, for example:

* [Report bugs](#report-bugs)
* [Fix Bugs](#fix-bugs)
* [Implement Features](#implement-features)
* [Write Documentation](#write-documentation)
* [Submit Feedback](#submit-feedback)

## Developer notes

### Developer Dependencies

* `conda` (>= 24.11.0)
* `conda-lock` (>= 2.5.7)

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
    * Confirm That You Have Docker Desktop with Buildx Support (you should see version information after running this command below)

        ```
        docker buildx version
        ```

    * Create a new buildx builder

        ```
        docker buildx create --name mybuilder
        ```

    * Use the new builder

        ```
        docker buildx use mybuilder
        ```

    * Initialize the builder and make sure QEMU is active

        ```
        docker buildx inspect --bootstrap
        ```

    * Building docker buildx with Emulation

        ```
        docker buildx build --platform linux/amd64 -t {YOUR-IMAGE-NAME} --load .
        ```

9. Edit the docker-compose.yml file (living at root of directory), replace the image name with {YOUR-IMAGE-NAME}.
    For example, replace "image: skysheng7/ai_bias:d077bb3" with "image: {YOUR-IMAGE-NAME}"

10. Running the docker image you just built at root directory

    ```
    docker compose up
    ```

## Report Bugs

Report bugs at <https://github.com/skysheng7/AI_representation_bias_in_farming/issues>.

**If you are reporting a bug, please follow the template guidelines. The more
detailed your report, the easier and thus faster we can help you.**

## Fix Bugs

Look through the GitHub issues for bugs. Anything labelled with `bug` and
`help wanted` is open to whoever wants to implement it. When you decide to work on such
an issue, please assign yourself to it and add a comment that you'll be working on that,
too. If you see another issue without the `help wanted` label, just post a comment, the
maintainers are usually happy for any support that they can get.

## Implement Features

Look through the GitHub issues for features. Anything labelled with
`enhancement` and `help wanted` is open to whoever wants to implement it. As
for [fixing bugs](#fix-bugs), please assign yourself to the issue and add a comment that
you'll be working on that, too. If another enhancement catches your fancy, but it
doesn't have the `help wanted` label, just post a comment, the maintainers are usually
happy for any support that they can get.

## Write Documentation

AI's representation bias about livestock farming could always use more documentation, whether as
part of the official documentation, in docstrings, or even on the web in blog
posts, articles, and such. Just [open an issue](<https://github.com/skysheng7/>
AI_representation_bias_in_farming/issues) to let us know what you will be working on
so that we can provide you with guidance.

## Submit Feedback

The best way to send feedback is to file an issue at <https://github.com/>
skysheng7/AI_representation_bias_in_farming/issues. If your feedback fits the format of one of
the issue templates, please use that. Remember that this is a volunteer-driven
project and everybody has limited time.

## Get Started

Ready to contribute? Here's how to set up AI's representation bias about livestock farming for
local development.

1. Fork the <https://github.com/skysheng7/AI_representation_bias_in_farming>
   repository on GitHub.
2. Clone your fork locally

    ```shell
    git clone git@github.com:your_name_here/AI_representation_bias_in_farming.git
    ```

3. Create a branch for local development using the default branch (typically `main`)
   as a starting
   point. Use `fix` or `feat` as a prefix for your branch name.

    ```shell
    git checkout main
    git checkout -b fix-name-of-your-bugfix
    ```

    Now you can make your changes locally.

4. Commit your changes and push your branch to GitHub. Please use [semantic
   commit messages](https://www.conventionalcommits.org/).

    ```shell
    git add .
    git commit -m "fix: summarize your changes"
    git push -u origin fix-name-of-your-bugfix
    ```

5. Open the link displayed in the message when pushing your new branch in order
   to submit a pull request.

### Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put your
   new functionality into a function with a docstring.
3. Your pull request will automatically be checked by the full test suite.
   It needs to pass all of them before it can be considered for merging.
