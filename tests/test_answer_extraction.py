from src.answer_extraction import (
    extract_choice_answer,
    extract_numeric_answer,
    normalize_answer,
)


def test_extract_choice_prefers_final_answer_over_earlier_option_mentions():
    text = "Option A is tempting because of the first clause. Final answer: B."
    assert extract_choice_answer(text) == "B"
    assert normalize_answer(text) == "B"


def test_extract_choice_falls_back_to_last_standalone_letter():
    text = "A is initially plausible, but the evidence supports B"
    assert extract_choice_answer(text) == "B"


def test_extract_numeric_prefers_final_marker_over_intermediate_numbers():
    text = "We first compute 12, then subtract 5. The answer is 7."
    assert extract_numeric_answer(text) == "7"
    assert normalize_answer(text) == "7"


def test_extract_numeric_keeps_hash_answer_inside_think_block():
    text = "<think>2 + 2 = 4\n#### 4</think>"
    assert extract_numeric_answer(text) == "4"
