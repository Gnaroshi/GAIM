import pytest

from gaim.full_train import epoch_groups, evaluation_summary, letter_token_ids, padded_inputs, validate_rows


def row(qid="train:0", note="", gold="B"):
    return {"id": qid, "question": "Original question", "options": dict(zip("ABCD", ("one", "two", "three", "four"))),
            "answer_idx": gold, "note": note}


def test_left_padding_keeps_actual_final_position_and_positions():
    packed = padded_inputs([[10, 11], [20, 21, 22, 23]], pad_id=0)
    assert packed["input_ids"] == [[0, 0, 10, 11], [20, 21, 22, 23]]
    assert packed["attention_mask"] == [[0, 0, 1, 1], [1, 1, 1, 1]]
    assert packed["position_ids"] == [[0, 0, 0, 1], [0, 1, 2, 3]]
    assert all(mask[-1] == 1 for mask in packed["attention_mask"])


def test_epoch_uses_every_row_exactly_once_including_last_partial_group():
    first = epoch_groups(35, 16, seed=20260922, epoch=0)
    second = epoch_groups(35, 16, seed=20260922, epoch=1)
    assert [len(group) for group in first] == [16, 16, 3]
    assert sorted(index for group in first for index in group) == list(range(35))
    assert sorted(index for group in second for index in group) == list(range(35))
    assert first == epoch_groups(35, 16, seed=20260922, epoch=0)
    assert first != second


def test_same_question_and_gold_required_for_both_training_rows():
    assert validate_rows([row(), row(note="A forum reader prefers A.")], "random") == 1
    bad = row(note="A forum reader prefers A.", gold="C")
    with pytest.raises(ValueError, match="원문 또는 정답"):
        validate_rows([row(), bad], "random")
    with pytest.raises(ValueError, match="입력 두 개"):
        validate_rows([row()], "clean")
    with pytest.raises(ValueError, match="원문 한 개"):
        validate_rows([row(), row()], "random")
    with pytest.raises(ValueError, match="추가 문장"):
        validate_rows([row(), row(note="Extra opinion")], "clean")
    assert validate_rows([row(), row()], "clean") == 1


def test_single_letter_token_constraint():
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert not add_special_tokens
            return [ord(text)]
    assert letter_token_ids(Tokenizer()) == {letter: ord(letter) for letter in "ABCD"}

    class BadTokenizer:
        def encode(self, text, add_special_tokens):
            return [1, 2]
    with pytest.raises(ValueError, match="단일 토큰"):
        letter_token_ids(BadTokenizer())


def test_metrics_separate_original_errors_from_new_errors_and_seen_test():
    predictions = [
        {"question_id": "test:0", "correct": True, "note": "", "defended": False, "historical_test_seen": True},
        {"question_id": "test:0", "correct": False, "note": "opinion", "defended": False, "historical_test_seen": True},
        {"question_id": "test:1", "correct": False, "note": "", "defended": False, "historical_test_seen": False},
        {"question_id": "test:1", "correct": False, "note": "opinion", "defended": False, "historical_test_seen": False},
        {"question_id": "test:0", "correct": False, "note": "", "defended": True, "historical_test_seen": True},
        {"question_id": "test:0", "correct": False, "note": "opinion", "defended": True, "historical_test_seen": True},
    ]
    summary = evaluation_summary(predictions)
    assert summary["all/neutral/clean"]["accuracy"] == 0.5
    assert summary["all/neutral/perturbed"]["clean_correct_perturbed_cases"] == 1
    assert summary["all/neutral/perturbed"]["correct_to_wrong"] == 1
    assert summary["all/neutral/perturbed"]["correct_to_wrong_rate"] == 1.0
    assert summary["all/defended/perturbed"]["clean_correct_perturbed_cases"] == 0
    assert summary["all/defended/perturbed"]["correct_to_wrong_rate"] is None
    assert summary["previously_unseen/neutral/clean"]["n"] == 1
    assert summary["previously_unseen/neutral/perturbed"]["correct_to_wrong_rate"] is None


def test_opinions_supporting_wrong_and_correct_answers_have_separate_metrics():
    common = {"question_id": "test:0", "defended": False, "historical_test_seen": True}
    predictions = [
        {**common, "correct": True, "note": "", "kind": "clean"},
        {**common, "correct": False, "note": "I prefer the wrong choice", "kind": "supports_wrong"},
        {**common, "correct": True, "note": "I prefer the correct choice", "kind": "supports_correct"},
    ]
    summary = evaluation_summary(predictions)
    assert summary["all/neutral/perturbed"]["accuracy"] == 0.5
    assert summary["all/neutral/supports_wrong"]["accuracy"] == 0.0
    assert summary["all/neutral/supports_wrong"]["correct_to_wrong_rate"] == 1.0
    assert summary["all/neutral/supports_correct"]["accuracy"] == 1.0
    assert summary["all/neutral/supports_correct"]["correct_to_wrong_rate"] == 0.0
    assert summary["previously_observed/neutral/supports_wrong"]["n"] == 1
    assert "previously_unseen/neutral/supports_wrong" not in summary
