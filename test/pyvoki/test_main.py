import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pyvoki as vo
from loguru import logger

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
    trainer_dir = root / 'tests/'
    if trainer_dir.is_dir():
        shutil.rmtree(trainer_dir)
    return vo.VokiTrainer(db_raw,
                          trainer_dir,
                          n_levels=N_LEVELS,
                          max_box_cards=MAX_BOX_CARDS)


def test_trainer_class(trainer):
    assert isinstance(trainer, vo.VokiTrainer)
    assert isinstance(trainer.box, vo.VokiBox)
    assert trainer.box.n_cards() == MAX_BOX_CARDS

def test_attempt(trainer):
    trainer.attempt


# @pytest.fixture
# def trainer_loaded(trainer,cards):
#     for card in cards[:MAX_BOX_CARDS]:
#         trainer.box.onboard_card(card, level=1)


# @pytest.fixture
# def cards(trainer):
#     return [vo.Card(row) for i, row in trainer.db.sample(30).iterrows()]
#
#
# def test_card_onboarding(trainer, cards, caplog):
#     for card in cards[:MAX_BOX_CARDS - 1]:
#         level = random.randint(1, 6)
#         trainer.box.onboard_card(card, level=level)
#
#     assert caplog.text == ''
#
#     trainer.box.onboard_card(cards[MAX_BOX_CARDS - 2], level=level)
#     assert 'already in this box' in caplog.text
#
#     assert 'cannot onboard' not in caplog.text
#     trainer.box.onboard_card(cards[MAX_BOX_CARDS - 1], level=level)
#     trainer.box.onboard_card(cards[MAX_BOX_CARDS], level=level)
#     assert 'cannot onboard' in caplog.text
#
#     assert 'this vokibox has only levels' not in caplog.text
#     trainer.box.onboard_card(cards[MAX_BOX_CARDS], level=N_LEVELS+1)
#     assert 'this vokibox has only levels' in caplog.text
#
#     assert 'but not level 0' not in caplog.text
#     trainer.box.onboard_card(cards[MAX_BOX_CARDS], level=0)
#     assert 'but not level 0' in caplog.text

