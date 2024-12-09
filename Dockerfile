FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

# copy conda environment dependencies
COPY conda-linux-64.lock /tmp/conda-linux-64.lock

# copy my local python package files to pip install in docker
COPY pyproject.toml /tmp/pyproject.toml
COPY src /tmp/src
COPY README.md /tmp/README.md

# conda install all the other packages
RUN mamba update --quiet --file /tmp/conda-linux-64.lock \
    && mamba clean --all -y -f \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

# install openai using pip because the openai package insatlled from conda has bug
# also install my local AI_representation_bias_in_farming as a python package
RUN pip install openai==1.57.0 \
    && python -m pip install -e /tmp 