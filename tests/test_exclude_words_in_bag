import pytest

from AI_representation_bias_in_farming import module3_word_freq_count

# standard case: the list of words to exclude is not empty and contains 2 words
standard_bag = ["this", "IS", "a", "TeSt"]
standard_exclude_list = ["a", "tesT"]
standard_expected = ["this", "IS"]

# edge case: the list of words to exclude is empty
edge_exclude_list = []
edge_expected = ["THIS", "is", "a", "TeSt"]

# error case: the bag of words is not a list
error_bag = 1


# Test standard case
def test_exclude_words_in_bag_standard():
    output = module3_word_freq_count.exclude_words_in_bag(
        standard_bag, standard_exclude_list
    )
    assert output.equals(standard_expected)


# Test edge case
def test_exclude_words_in_bag_edge():
    output = module3_word_freq_count.exclude_words_in_bag(
        standard_bag, edge_exclude_list
    )
    assert output.equals(edge_expected)


# Test error handling
def test_exclude_words_in_bag_error():
    with pytest.raises(TypeError):
        output = module3_word_freq_count.exclude_words_in_bag(
            error_bag, standard_exclude_list
        )
