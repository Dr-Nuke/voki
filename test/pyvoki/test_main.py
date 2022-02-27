import datetime
import os
import random
import shutil
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from loguru import logger

import pyvoki as vo

np.random.seed(123)

N_LEVELS = 6
MAX_BOX_CARDS = 20


@pytest.fixture
def caplog(caplog):
    """
    a fixture purely for loguru
    https://github.com/Delgan/loguru/issues/59#issuecomment-1016516449
    """
    handler_id = logger.add(caplog.handler, format="{message}")
    yield caplog
    logger.remove(handler_id)


@pytest.fixture
def trainer():
    root = Path(os.getcwd())
    db_raw_fpath = root / 'data/voki_export_full_db.csv'
    db_raw = pd.read_csv(db_raw_fpath, sep=';')
    db_raw['book'] = db_raw['file'].str.slice(start=31, stop=32).astype(int)
    db_raw.head(1)
    trainer_dir = root / 'tests_1/'
    if trainer_dir.is_dir():
        shutil.rmtree(trainer_dir)
    return vo.VokiTrainer(db_raw,
                          trainer_dir,
                          n_levels=N_LEVELS,
                          max_box_cards=MAX_BOX_CARDS)


@pytest.fixture
def trainer_empty_box():
    root = Path(os.getcwd())
    db_raw_fpath = root / 'data/voki_export_full_db.csv'
    db_raw = pd.read_csv(db_raw_fpath, sep=';')
    db_raw['book'] = db_raw['file'].str.slice(start=31, stop=32).astype(int)
    db_raw.head(1)
    trainer_dir = root / 'tests_2/'
    if trainer_dir.is_dir():
        shutil.rmtree(trainer_dir)
    return vo.VokiTrainer(db_raw,
                          trainer_dir,
                          n_levels=N_LEVELS,
                          max_box_cards=MAX_BOX_CARDS,
                          voki_box=vo.VokiBox(n=N_LEVELS,
                                              max_cards=MAX_BOX_CARDS))


def test_drop_by_hash(trainer_empty_box,caplog):
    hashes = list(trainer_empty_box.db.sample(MAX_BOX_CARDS).index)
    levels = [random.randint(1, N_LEVELS) for _ in range(MAX_BOX_CARDS)]
    for h, lvl in zip(hashes, levels):
        trainer_empty_box.box.onboard_card(h, level=lvl)
    assert trainer_empty_box.box.n_cards() == MAX_BOX_CARDS

    # test positive:
    for h in hashes:
        result = trainer_empty_box.box.drop_by_hash(h)
        assert result

    # test negative:
    assert 'is not present in voibox' not in caplog.text
    result = trainer_empty_box.box.drop_by_hash(hashes[0])
    assert not result
    assert 'is not present in voibox' in caplog.text


    print('')


def test_save_and_load(trainer_empty_box):
    level = 2
    hash_ = trainer_empty_box.db.iloc[0].name
    trainer_empty_box.box.onboard_card(hash_, level=level)
    root = trainer_empty_box.ROOT

    trainer_empty_box.save()
    trainer_empty_box.save()
    trainer_loaded = vo.VokiTrainer.load(root)
    assert all(trainer_empty_box.db.columns == trainer_loaded.db.columns)


def test_fromvokibox_to_freefloat(trainer_empty_box, monkeypatch):
    level = N_LEVELS
    hash_ = trainer_empty_box.db.iloc[0].name
    solution = trainer_empty_box.db.loc[hash_]['text']
    input_output = StringIO(f'{solution}\n')
    monkeypatch.setattr('sys.stdin', input_output)

    assert len(trainer_empty_box.box.levels[level]) == 0
    assert not trainer_empty_box.db.loc[hash_, 'passed_vokibox']
    trainer_empty_box.box.onboard_card(hash_, level=level)
    trainer_empty_box.attempt_card_in_level(lvl=level)
    assert hash_ not in trainer_empty_box.box.levels[level]
    assert trainer_empty_box.db.loc[hash_, 'passed_vokibox']


def test_trainer_class(trainer):
    assert isinstance(trainer, vo.VokiTrainer)
    assert isinstance(trainer.box, vo.VokiBox)
    assert trainer.box.n_cards() == MAX_BOX_CARDS


def test_successful_attempt(trainer, monkeypatch):
    level = 1
    hash_ = trainer.box.levels[level][0]
    solution = trainer.db.loc[hash_]['text']
    input_output = StringIO(f'{solution}\n')
    monkeypatch.setattr('sys.stdin', input_output)

    np.testing.assert_equal(trainer.db.loc[hash_]['last_attempt_time'], np.nan)
    np.testing.assert_equal(trainer.db.loc[hash_]['last_successful_attempt_time'], np.nan)
    assert len(trainer.db.loc[hash_]['attempts']) == 0

    trainer.attempt_card_in_level(lvl=level)

    assert len(trainer.db.loc[hash_]['attempts']) == 1
    assert trainer.db.loc[hash_]['attempts'][0].success
    assert isinstance(trainer.db.iloc[0]['last_attempt_time'], datetime.datetime)
    assert isinstance(trainer.db.iloc[0]['last_successful_attempt_time'], datetime.datetime)
    assert hash_ not in trainer.box.levels[level]
    assert hash_ in trainer.box.levels[level + 1]


def test_unsuccessful_attempt(trainer, monkeypatch):
    level = 1
    hash_ = trainer.box.levels[level][0]
    input_output = StringIO('wrong answer\n')
    monkeypatch.setattr('sys.stdin', input_output)

    np.testing.assert_equal(trainer.db.loc[hash_]['last_attempt_time'], np.nan)
    np.testing.assert_equal(trainer.db.loc[hash_]['last_successful_attempt_time'], np.nan)
    assert len(trainer.db.loc[hash_]['attempts']) == 0

    trainer.attempt_card_in_level(lvl=level)

    assert len(trainer.db.loc[hash_]['attempts']) == 1
    assert not trainer.db.loc[hash_]['attempts'][0].success
    assert isinstance(trainer.db.iloc[0]['last_attempt_time'], datetime.datetime)
    np.testing.assert_equal(trainer.db.loc[hash_]['last_successful_attempt_time'], np.nan)
    assert hash_ in trainer.box.levels[level]


@pytest.mark.parametrize('level', range(1, N_LEVELS + 1))
def test_onboard_card(trainer_empty_box, level):
    hash_ = trainer_empty_box.db.iloc[0].name
    assert len(trainer_empty_box.box.levels[level]) == 0
    result = trainer_empty_box.box.onboard_card(hash_, level=level)
    assert result
    assert trainer_empty_box.box.levels[level][0] == hash_


@pytest.mark.parametrize('level', [-1, 0, N_LEVELS + 1])
def test_onboard_card_bad_level(trainer_empty_box, level, caplog):
    hash_ = trainer_empty_box.db.iloc[0].name
    assert 'this vokibox has only levels 1-' not in caplog.text
    result = trainer_empty_box.box.onboard_card(hash_, level=level)
    assert not result
    assert 'this vokibox has only levels 1-' in caplog.text


def test_onboard_card_already_onboarded(trainer_empty_box, caplog):
    # tests that a card cannot be onboarded twice into the same vokibox
    level = 1
    hash_ = trainer_empty_box.db.iloc[0].name
    assert len(trainer_empty_box.box.levels[level]) == 0
    result = trainer_empty_box.box.onboard_card(hash_, level=level)
    assert result
    assert trainer_empty_box.box.levels[level][0] == hash_

    result = trainer_empty_box.box.onboard_card(hash_, level=level)
    assert not result
    assert 'already in this box' in caplog.text
