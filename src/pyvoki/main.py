import datetime
import pickle
import sys
from hashlib import md5
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

logger.add(sys.stdout,
           colorize=True,
           format="<green>{time}</green> <level>{message}</level>",
           filter="my_module",
           level="INFO")


class VokiTrainer:
    def __init__(self, df: pd.DataFrame, db_fpath, load=False, n_levels=6, max_box_cards=20, voki_box=None):
        """
        initializes a new database off a Dataframe, as well as a associated vokibox
        :param df: data for the database as pd.DataFrame
        :param db_fpath: filepath for db file on disk
        :param load: if True, immediate save is skipped
        """
        self.ROOT = db_fpath
        self.DB_PATH = db_fpath / 'db.csv'

        # brush up the incoming dataframe, and save it as the db variable
        df['hash'] = df['text'].astype(str).apply(lambda x: md5(x.encode('utf-8')).hexdigest())
        df['attempts'] = [[] for _ in range(len(df))]
        df.set_index('hash', drop=True, inplace=True)
        df = df.sort_values(by=['book', 'page', 'y_max'])
        self.db = df
        self.fpath = db_fpath

        if voki_box is None:
            self.box = VokiBox(n=n_levels,
                               max_cards=max_box_cards,
                               fpath=db_fpath / 'box.pk')
        else:
            self.box = voki_box
            self.update(voki_box)

        if not load:
            assert isinstance(db_fpath, Path), f'db_fpath must be pathlib Path, not {type(db_fpath)}'
            assert not db_fpath.is_file(), f'cannot write to existing file {db_fpath}. Specify a non-existing file'
            self.save()

    def __repr__(self):
        return f'Voki Trainer with {len(self.db)} entries in db'

    @staticmethod
    def load(path: Path):
        """
        loads a saved database
        """
        assert isinstance(path, Path)
        df = pd.read_csv(path)
        return VokiTrainer(df, path, load=True)

    def save(self):
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=False)
        self.db.to_csv(self.DB_PATH)

    def update_entry_by_card(self, card):
        """
        updates a card with incoming card
        :param card:
        :return:
        """
        pass

    def update(self, voki_box):
        """
        incorporaes changes of a voki box into the db
        :param voki_box:
        :return:
        """
        pass

    def drop_by_hash(self,hash_):
        """
        given a hash, removes the according card if it is within the box
        :param hash_:
        :return:
        """
        for i, lvl in self.levels.items():
            for card in lvl:
                if card.hash == hahs_:
                    self.levels[i]



class Card:
    def __init__(self, row: pd.core.series.Series):
        hash_ = row.name
        assert hash_ is not None
        kwargs = row.to_dict()
        kwargs.update({'hash': hash_})
        required_attributes = ['text', 'text_g', 'text_s']
        for arg in required_attributes:
            assert arg in kwargs

        self.__dict__.update(kwargs)

    def __repr__(self):
        return f'Card "{self.text}" with ID {self.hash}'


class Attempt:
    def __init__(self, date, answer, success, solution):
        my_assert(date, datetime.datetime, 'date')
        my_assert(answer, str, 'answer')
        my_assert(success, bool, 'success')
        my_assert(solution, str, 'solution')
        self.solution = solution
        self.date = date.replace(microsecond=0)
        self.answer = answer
        self.success = success

    def __repr__(self):
        if self.success:
            success_str = 'Successful'
        else:
            success_str = 'Failed'
        return f'{success_str} attempt at {self.date} on "{self.solution}": "{self.answer}"'


def my_assert(obj, type_, obj_name):
    assert isinstance(obj, type_), f'{obj_name} must be {type_}, but is {type(obj)}: {obj}'


def card_id(card):
    return card['id'][0]


class VokiBox:
    def __init__(self, n=6, max_cards=20, fpath='box.pk'):  # box_db,voki_db

        self.levels = {i: [] for i in range(1, n + 1)}
        self.max_cards = max_cards
        self.n_levels = n
        self.fpath = fpath

    def onboard_card(self, card, level=1):
        my_assert(card, Card, 'card')
        my_assert(level, int, 'level')

        if level > self.n_levels:
            logger.warning(f'this vokibox has only levels 1-{self.n_levels} but not level {level}')
            return False

        if self.n_cards() >= self.max_cards:
            logger.warning(f'cannot onboard card "{card.text}" there are already {self.n_cards()} cards in this box.')
            return False

        if self.check_card_already_in_box(card):
            return False

        self.levels[level].append(card)
        return True

    def get_all_cards(self):
        return list(chain.from_iterable(self.levels.values()))

    def attempt_card_in_level(self, lvl, df):
        # attempt the next card in vokibox level i
        if len(self.levels) <= lvl:
            logger.warning(f'You cannot attempt level {lvl} as there are only levels 0-{len(self.levels) - 1}')
            return False
        if len(self.levels[lvl]) == 0:
            logger.warning(f'vokibox lvl {lvl} is empty. you need to fill it first')
            return False

        top_card_id = self.levels[lvl].pop(0)

        card, result = attempt(top_card_id, df)

        if result:
            self.advance(card, lvl)
        else:
            self.onboard_card(top_card_id)

    def advance(self, card, lvl):
        if lvl < len(self.levels) - 1:
            self.levels[lvl + 1].append(card['id'])
        else:
            print(f'congrats! card \"{card["french_phrase"]}\" made it through the vokibox!')

    def n_cards(self):
        return len(self.get_all_cards())

    def __repr__(self):
        strlist = []
        for i, lvl in self.levels.items():
            strlist.append(f'level {i}')
            for card in lvl:
                strlist.append(f'    {card.text}')

        return '\n'.join(strlist)

    def save(self):

        with open(self.fpath, 'wb') as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(fpath):
        with open(fpath, 'rb') as fh:
            return pickle.load(fh)

    def check_card_already_in_box(self, card):
        for i, lvl in self.levels.items():
            if card.hash in [c.hash for c in lvl]:
                logger.warning(f'Card {card} is already in this box at level {i}')
                return True
        return False


def box_db_load(box_db_file):
    with open(box_db_file, 'rb') as fh:
        return pickle.load(fh)


def box_db_save(box_db_file):
    with open(box_db_file, 'wb') as fh:
        pickle.dump(fh)


def card_from_id(ids, df):
    if not isinstance(ids, list):
        ids = [ids]
    return_cards = df[df['id'].isin(ids)]
    if len(return_cards) == 0:
        print(f'ids {ids} not existing in df')

    return return_cards.iloc[0].to_dict()


def attempt(id_, df):
    card = card_from_id(id_, df)
    now = datetime.datetime.now().replace(microsecond=0)
    guess = input(f'{card["german_phrase"]}')
    result = check_attempt(card, guess)  # true/False

    card['attempt_history'] = card['attempt_history'].append(pd.DataFrame({'time': [now],
                                                                           'result': [result],
                                                                           'answer': [guess]}))
    if result:
        card['streak'] += 1
    else:
        card['streak'] = np.floor(card['streak'] / 2)
    return card, result


def check_attempt(card, guess):
    if guess == card['french_phrase']:
        print('correct!')
        return True
    text = input(f'type "yes" if you think they match:\n {guess} \n {card["french_phrase"]}\n')
    if text == 'yes':
        return True
    return False


def main():
    """
    mycomment
    :return:
    """
    pass


if __name__ == "__main__":
    main()
