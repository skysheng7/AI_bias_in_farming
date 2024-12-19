"""
Bag-of-words analysis by counting 1-gram, 2-gram, or 1 & 2 gram words in revised prompts and in text descriptions of images
"""

import click
import pandas as pd
from pathlib import Path

from AI_representation_bias_in_farming import utils
from AI_representation_bias_in_farming import module3_word_freq_count


@click.command()
def main():
    """Process the megadata dataframe for word frequency analysis."""

    # import the megadata dataframe
    megadata = utils.read_megadata()

    # Define a list of words to exclude from analysis because they are not setting the difference between images.
    # **MAYBE**: remove descriptions like "south asian", "middle eastern", "caucasian hispanic", "" because it's describing human's culture background to ensure EDI, irelevant to our study
    # my megadata has a column called size to set the size of image, would be in column name conflict so i exclude from bag of wordds
    # the same reason applies to "model", "country","prompt", "file", "quality"
    words_to_exclude_1gram = [
        "cows",
        "cow",
        "farm",
        "farms",
        "pig",
        "pigs",
        "dairy",
        "scene",
        "background",
        "picture",
        "show",
        "shows",
        "depicts",
        "depiction",
        "depicting",
        "image",
        "typical",
        "representation",
        "representing",
        "showcase",
        "showcasing",
        "include",
        "including",
        "elements",
        "united",
        "states",
        "america",
        "american",
        "germany",
        "german",
        "new",
        "zealand",
        "spain",
        "spanish",
        "australia",
        "australian",
        "seen",
        "nearby",
        "overall",
        "atmosphere",
        "area",
        "foreground",
        "setting",
        "settings",
        "detailed",
        "suggesting",
        "indicating",
        "country",
        "size",
        "model",
        "file",
        "prompt",
        "quality",
    ]
    words_to_exclude_2_3gram = [
        "dairy cows",
        "dairy cow",
        "dairy farm",
        "dairy farms",
        "pig farms",
        "pig farm",
        "typical dairy",
        "typical pig",
        "image typical",
        "image shows",
        "representation typical",
        "depiction typical",
        "depicting typical",
        "depict detailed",
        "depicts detailed",
        "overall atmosphere",
        "setting overall",
        "farm scene",
        "farm setting",
        "realistic depiction",
        "accurate representation",
        "realistic image",
        "realistic representation",
        "accurate depiction",
        "image depicts",
        "image features",
        "generate image",
        "create image",
        "scene include",
        "scene depicting",
        "overall scene",
        "united states",
        "new zealand",
        "farm united",
        "farms united",
        "states scene",
        "farm germany",
        "farms germany",
        "germany scene",
        "farm new",
        "zealand scene",
        "farm spain",
        "farms spain",
        "spain scene",
        "farm australia",
        "farms australia",
        "australia scene",
        "typical australian",
        "capturing essence",
        "likely used",
    ]
    words_to_exclude = words_to_exclude_1gram + words_to_exclude_2_3gram

    # set number of columns in the grid of plots, words that we want to remove from wordclowd
    top_word_n_to_show = 20
    countries_by_farm_type = {
        "dairy": ["the United States", "Germany", "New Zealand"],
        "pig": ["the United States", "Spain", "Australia"],
    }  # list of countries with the biggest number of dairy cows and pigs in North America, Europe and Oceania

    # In revised prompts: create bag of words to count frequency of single word (1-gram), and phrases made of 2 words (2 grams) occurance
    revised_prompt_1gram, revised_prompt_words_1gram, revised_prompt_1gram_summary = (
        module3_word_freq_count.count_word_freq(
            megadata,
            words_to_exclude=words_to_exclude_1gram,
            col_of_interest="revised_prompt",
            ngram_range=(1, 1),
            min_freq_to_include=20,
            top_word_n_to_show=top_word_n_to_show,
        )
    )
    revised_prompt_2gram, revised_prompt_words_2gram, revised_prompt_2gram_summary = (
        module3_word_freq_count.count_word_freq(
            megadata,
            words_to_exclude_2_3gram,
            col_of_interest="revised_prompt",
            ngram_range=(2, 2),
            min_freq_to_include=20,
            top_word_n_to_show=top_word_n_to_show,
        )
    )
    (
        revised_prompt_1_2gram,
        revised_prompt_words_1_2gram,
        revised_prompt_1_2gram_summary,
    ) = module3_word_freq_count.count_word_freq(
        megadata,
        words_to_exclude,
        col_of_interest="revised_prompt",
        ngram_range=(1, 2),
        min_freq_to_include=20,
        top_word_n_to_show=top_word_n_to_show,
    )

    # In GPT4o_description: create bag of words to count frequency of single word (1-gram), and phrases made of 2 words (2 grams) occurance
    gpt4o_description_1gram, gpt4o_words_1gram, gpt4o_description_1gram_summary = (
        module3_word_freq_count.count_word_freq(
            megadata,
            words_to_exclude_1gram,
            col_of_interest="GPT4o_description",
            ngram_range=(1, 1),
            min_freq_to_include=20,
            top_word_n_to_show=top_word_n_to_show,
        )
    )

    gpt4o_description_2gram, gpt4o_words_2gram, gpt4o_description_2gram_summary = (
        module3_word_freq_count.count_word_freq(
            megadata,
            words_to_exclude_2_3gram,
            col_of_interest="GPT4o_description",
            ngram_range=(2, 2),
            min_freq_to_include=20,
            top_word_n_to_show=top_word_n_to_show,
        )
    )

    (
        gpt4o_description_1_2gram,
        gpt4o_words_1_2gram,
        gpt4o_description_1_2gram_summary,
    ) = module3_word_freq_count.count_word_freq(
        megadata,
        words_to_exclude,
        col_of_interest="GPT4o_description",
        ngram_range=(1, 2),
        min_freq_to_include=20,
        top_word_n_to_show=top_word_n_to_show,
    )


# Only execute this code if the script is run directly
if __name__ == "__main__":
    main()
