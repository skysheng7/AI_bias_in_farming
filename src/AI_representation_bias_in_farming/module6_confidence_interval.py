"""Library of functions used to calculate confidence interval for binary classifications across different prompts and models.
"""

import numpy as np
from scipy.stats import bootstrap


def calculate_ci(cluster, megadata, n_resamples=10000, random_seed=7):
    """
    Calculate bootstrap confidence intervals for binary classifications across different prompts and models.

    Parameters
    ----------
    cluster : pandas.DataFrame
        DataFrame containing prompt and model information. Must have columns 'prompt' and 'model'.
        New columns will be added to this DataFrame for CI results.

    megadata : pandas.DataFrame
        Source data containing the classifications. Must have columns 'prompt', 'model',
        and 'GPT4o_cluster' where 'GPT4o_cluster' contains the categorical labels.

    n_resamples : int, optional (default=10000)
        Number of bootstrap resamples to perform.

    random_seed : int, optional (default=7)
        Random seed for reproducibility of bootstrap sampling.

    Returns
    -------
    pandas.DataFrame
        Modified cluster DataFrame with added columns for confidence intervals:
        - '{category}_ci_lower': Lower bound of CI for each category
        - '{category}_ci_upper': Upper bound of CI for each category
        where category is either 'indoor' or 'outdoor'.
        For cases where all values are 1 or all are 0, both CI bounds will be set to that value.

    Notes
    -----
    - Only calculates CIs for 'indoor' and 'outdoor' categories, ignoring 'other'
    - Uses 95% confidence level for bootstrap intervals
    - Modifies the input cluster DataFrame in-place

    Example
    -------
    >>> result = calculate_ci(cluster_df, mega_df, random_seed=7)
    >>> print(result[['prompt', 'model', 'indoor_ci_lower', 'indoor_ci_upper']])
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # only calculate 2 categories of interest, ignore "other"
    category = ["outdoor", "indoor"]

    # iterate through each unique prompt for each model
    for index, row in cluster.iterrows():
        prompt = row["prompt"]
        model = row["model"]
        cur_raw_megadata = megadata[
            (megadata["prompt"] == prompt) & (megadata["model"] == model)
        ]
        col = cur_raw_megadata["GPT4o_cluster"]

        # calculate CI for each category's point estimate using this unique prompt
        for cat in category:
            lower_col_name = cat + "_ci_lower"
            upper_col_name = cat + "_ci_upper"
            converted_col = col.copy()
            converted_col = np.where(converted_col == cat, 1, 0)

            if np.all(converted_col == 0):
                cluster.at[index, lower_col_name] = 0
                cluster.at[index, upper_col_name] = 0
            elif np.all(converted_col == 1):
                cluster.at[index, lower_col_name] = 1
                cluster.at[index, upper_col_name] = 1
            else:
                boot_results = bootstrap(
                    (converted_col,),
                    np.mean,
                    confidence_level=0.95,
                    n_resamples=n_resamples,
                )
                cluster.at[index, lower_col_name] = boot_results.confidence_interval[0]
                cluster.at[index, upper_col_name] = boot_results.confidence_interval[1]

    return cluster
