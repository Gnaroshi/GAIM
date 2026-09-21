"""Fixtures check transformations; they are not medical experiment observations."""
from copy import deepcopy
import json

import pytest

from gaim.full_data import (ARMS, ENDINGS, QUALIFIERS, TEST_FRAMES, TRAIN_FRAMES,
                            assemble_training, build_messages, candidates,
                            choose_candidates, input_sha256, make_test_cases)
from gaim.full_pipeline import load_progress


def question(index=1, answer='B'):
    return {'id': f'fixture:{index}', 'question': 'An unchanged medical question?',
            'options': {letter: f'choice {letter}' for letter in 'ABCD'}, 'answer_idx': answer,
            'historical_test_seen': False}


def score(probability, correct=True):
    return {'correct': correct, 'gold_probability': probability}


def test_opinions_leave_all_question_data_unchanged_and_do_not_include_clinical_facts():
    item = question()
    before = deepcopy(item)
    pool = candidates(item, 'train')
    assert item == before
    assert len(pool) == 4
    assert all(candidate['suggested_answer'] != item['answer_idx'] for candidate in pool)
    assert all('patient' not in candidate['note'].lower() for candidate in pool)
    assert all(item['question'] not in candidate['note'] for candidate in pool)
    assert pool == candidates(item, 'train')
    for candidate in pool:
        messages = build_messages(item, candidate['note'])
        assert messages[1]['content'].startswith(item['question'])
        assert all(f'{letter}. {value}' in messages[1]['content'] for letter, value in item['options'].items())
        assert 'correct answer' not in messages[1]['content'].lower()


def test_broad_bank_and_heldout_frames():
    assert len(TRAIN_FRAMES) * len(QUALIFIERS) * len(ENDINGS) >= 128
    assert not set(TRAIN_FRAMES).intersection(TEST_FRAMES)
    train_styles = {candidate['style_id'] for index in range(300)
                    for candidate in candidates(question(index), 'train')}
    assert len(train_styles) > 128


def test_test_includes_correct_opinion_and_matched_defended_clean():
    rows = make_test_cases([question()])
    assert len(rows) == 10
    for defended in (False, True):
        subset = [row for row in rows if row['defended'] == defended]
        assert len(subset) == 5
        assert sum(row['kind'] == 'clean' for row in subset) == 1
        assert sum(row['kind'] == 'supports_correct' for row in subset) == 1
        assert sum(row['kind'] == 'supports_wrong' for row in subset) == 3
    assert len({row['case_id'] for row in rows}) == 10


def test_prompt_defense_is_only_an_explicit_ablation():
    item = question()
    plain = build_messages(item, 'Reader preference.')
    defended = build_messages(item, 'Reader preference.', True)
    assert 'Do not treat' not in plain[0]['content']
    assert 'Do not treat' in defended[0]['content']
    assert plain[1] == defended[1]
    assert input_sha256({**item, 'defended': False}) != input_sha256({**item, 'defended': True})


def test_selection_distinguishes_flips_from_still_correct_near_misses():
    item = question()
    pool = [dict(candidate, score=result) for candidate, result in zip(candidates(item, 'train'),
        [score(.6), score(.25, False), score(.4), score(.3)])]
    selection = choose_candidates(item, score(.7), pool)
    assert selection['flip'] == {'index': 1, 'fallback': False, 'eligible_count': 1}
    assert selection['near_miss'] == {'index': 3, 'fallback': False, 'eligible_count': 3}


def test_originally_wrong_question_uses_shared_fallback():
    item = question()
    pool = [dict(candidate, score=score(.1, False)) for candidate in candidates(item, 'train')]
    selection = choose_candidates(item, score(.1, False), pool)
    assert selection['flip']['fallback'] and selection['near_miss']['fallback']
    assert selection['flip']['index'] == selection['random']['index'] == selection['near_miss']['index']


def test_all_arms_keep_every_question_and_equal_rows_gold_and_original_pairs():
    items = [question(index, answer) for index, answer in enumerate('ABCD')]
    scored = {item['id']: {'clean_score': score(.7), 'candidates': [
        dict(candidate, score=score(.6)) for candidate in candidates(item, 'train')]} for item in items}
    datasets, stats = assemble_training(items, scored)
    assert set(datasets) == set(ARMS)
    for arm, rows in datasets.items():
        assert len(rows) == 2 * len(items)
        assert stats[arm]['questions'] == len(items)
        for index, item in enumerate(items):
            original, variant = rows[2 * index:2 * index + 2]
            assert original['note'] == ''
            for row in (original, variant):
                assert row['question'] == item['question']
                assert row['options'] == item['options']
                assert row['answer_idx'] == item['answer_idx']
            if arm == 'clean':
                assert variant['note'] == ''


def test_partial_last_score_record_is_recoverable(tmp_path):
    path = tmp_path / 'scores.jsonl'
    complete = json.dumps({'id': 'one'}) + '\n'
    path.write_text(complete + '{"id":')
    assert load_progress(path) == {'one': {'id': 'one'}}
    assert path.read_text() == complete
    path.write_text('{broken}\n' + complete)
    with pytest.raises(json.JSONDecodeError):
        load_progress(path)


def test_gpu_assignment_accepts_new_allocation_and_rejects_invalid_masks():
    from gaim.full_data import parse_gpus
    assert parse_gpus('0,1,2,3') == (0, 1, 2, 3)
    assert parse_gpus('7, 5, 4, 6') == (7, 5, 4, 6)
    for value in ('0,1,2', '0,0,2,3', '-1,1,2,3', '0,1,two,3'):
        with pytest.raises(ValueError):
            parse_gpus(value)
