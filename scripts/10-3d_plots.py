"""
Generate 3d plots to summarize image clustering results
"""

import pandas as pd
from pathlib import Path
import click

from AI_representation_bias_in_farming import utils
from AI_representation_bias_in_farming import module7_3d_plot

@click.command()
def main():
    # import manual cluster check note
    cluster = utils.read_csv_file("cluster_summary.csv")
