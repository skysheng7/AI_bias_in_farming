# Tests Folder

This folder contains the test cases for the functions in `AI_representation_bias_in_farming`. The tests are written to validate the functionality of the module and ensure the reliability of the data pipeline.

## Purpose of the Tests

These tests are included as part of a demonstration of good practices in creating a complete, reproducible, and trustworthy data science pipeline. Even though the tests are limited in number and scope, they aim to illustrate the importance of testing functions in any data science workflow.

## Available Tests

Below are tests for `module3_word_freq_count` module in the `AI_representation_bias_in_farming` python package, which implements the function `exclude_words_in_bag`:

1. **Standard Case**:
   - Tests the function when the input bag of words and exclude list are non-empty.
   - Ensures the correct exclusion of specified words from the bag.

2. **Edge Case**:
   - Tests the function when the exclude list is empty.
   - Ensures the bag of words remains unchanged.

3. **Error Handling**:
   - Tests the function's behavior when the input `bag_of_words` is not a list.
   - Confirms that a `TypeError` is raised.

## Running the Tests

These tests are written using the `pytest` framework. To run the tests, run the following command at the root of the repo directory:

```bash
pytest
```
