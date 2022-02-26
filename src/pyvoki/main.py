import datetime
import pickle
import sys
from hashlib import md5
from itertools import chain
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

logger.add(sys.stdout,
           colorize=True,
           format="<green>{time}</green> <level>{message}</level>",
           filter="my_module",
           level="INFO")


class VokiTrainer:
    def __init__(self,
                 df: pd.DataFrame,
                 db_fpath,
                 load=False,
                 n_levels=6,
                 max_box_cards=20,
                 voki_box=None):
        """
        initializes a new database off a Dataframe, as well as a associated vokibox
        :param df: data for the database as pd.DataFrame
        :param db_fpath: filepath for db file on disk
        :param load: if True, immediate save is skipped
        """
        self.ROOT = db_fpath
        self.DB_PATH = db_fpath / 'db.csv'

        # brush up the incoming dataframe, and save it as the db variable
        hash_base = df[['text', 'text_g', 'text_s']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        df['hash'] = hash_base.astype(str).apply(lambda x: md5(x.encode('utf-8')).hexdigest())
        df.drop_duplicates(subset=['hash'])
        df['attempts'] = [[] for _ in range(len(df))]
        df.set_index('hash', drop=True, inplace=True)
        df = df.sort_values(by=['book', 'page', 'y_max'])
        strcols = ['text', 'text_g', 'text_s']
        for col in strcols:
            df[col] = df[col].astype(str)

        df['streak'] = 0  # how often did you consecutively guess corectly?
        df['last_attempt_time'] = np.nan
        df['last_successful_attempt_time'] = np.nan
        df['passed_vokibox'] = False  # indicates that a card passed the vokibox and is now free-float

        self.db = df
        self.fpath = db_fpath

        if voki_box is None:
            self.box = VokiBox(n=n_levels,
                               max_cards=max_box_cards,
                               fpath=db_fpath / 'box.pk')
            self.box.fill(self.get_next_box_cards())
        else:
            my_assert(voki_box, VokiBox, 'voki_box')
            self.box = voki_box

        if not load:
            assert isinstance(db_fpath, Path), f'db_fpath must be pathlib Path, not {type(db_fpath)}'
            assert not db_fpath.is_file(), f'cannot write to existing file {db_fpath}. Specify a non-existing file'
            self.save()

    def __repr__(self):
        return f'Voki Trainer with {len(self.db)} entries in db'

    def get_next_box_cards(self):
        """
        Given the current voki box fill status, and the db, identify those cards that go noxt into the box
        """
        df = self.db[self.db['attempts'].apply(len) < self.box.n_levels]
        n_missing_cards = self.box.max_cards - self.box.n_cards()
        return list(df.index[:n_missing_cards])

    def attempt_card_in_level(self, lvl: int = 1):
        # attempt the next card in vokibox level lvl
        if lvl not in self.box.levels:
            logger.warning(f'You cannot attempt level {lvl} as there are only levels {list(self.box.keys())}')
            return False
        if len(self.box.levels[lvl]) == 0:
            logger.warning(f'vokibox lvl {lvl} is empty. you need to fill it first')
            return False

        top_card = self.box.levels[lvl][0]
        attempt_result = self.attempt_card(top_card)
        _, passed_box = self.box.update_box(top_card, attempt_result.success)
        if passed_box:
            # todo: fill this
            pass

    def attempt_card(self, card):
        """
        attempt a individual card
        :param card:
        :return:
        """
        row = self.db.loc[card]
        now = datetime.datetime.now().replace(microsecond=0)
        guess = input(f"{row['text_g']}")

        attempt_result = Attempt(now, guess, self.check_attempt(row, guess))
        self.update_db(attempt_result, card)

        return attempt_result

    def check_attempt(self, row, guess):
        if guess == row['text']:
            print('correct!')
            return True
        # Todo: add more logic like levenstein comparison, manual override, etc
        return False

    def update_db(self, attempt_result: "Attempt", card: str):
        """
        incorporaes changes of an attempt into the db
        """
        my_assert(attempt_result, Attempt, 'attempt_result')
        self.db.at[card, 'attempts'].append(attempt_result)
        self.db.at[card, 'last_attempt_time'] = attempt_result.date
        if attempt_result.success:
            self.db.at[card, 'last_successful_attempt_time'] = attempt_result.date
            self.db.at[card, 'streak'] += 1
        else:
            self.db.at[card, 'streak'] = 0
        return True

    def show_box(self):
        strlist = []
        strlist.append(f'Voki Box with {self.box.n_cards()}/{self.box.max_cards} cards')

        for i, lvl in self.box.levels.items():
            strlist.append(f'level {i}')
            for hash_ in lvl:
                strlist.append(f"    {self.db.loc[hash_, 'text_g']}")
        print('\n'.join(strlist))

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

    def drop_by_hash(self, hash_):
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
    def __init__(self, date, answer, success):
        my_assert(date, datetime.datetime, 'date')
        my_assert(answer, str, 'answer')
        my_assert(success, bool, 'success')
        self.date = date.replace(microsecond=0)
        self.answer = answer
        self.success = success

    def __repr__(self):
        if self.success:
            success_str = 'Successful'
        else:
            success_str = 'Failed'
        return f'{success_str} attempt at {self.date}: "{self.answer}"'


def my_assert(obj, type_, obj_name):
    assert isinstance(obj, type_), f'{obj_name} must be {type_}, but is {type(obj)}: {obj}'


class VokiBox:
    def __init__(self, n=6, max_cards=20, fpath='box.pk'):  # box_db,voki_db

        self.n_levels = n
        self.initiate()
        self.max_cards = max_cards
        self.fpath = fpath

    def update_box(self, card: str, success: bool):
        """
        updates the voki box accordingly
        returns (bool,bool) = (True, "card passed last level")
        """
        my_assert(card, str, 'card')
        my_assert(success, bool, 'attempt')

        old_lvl = self.get_key_of_card(card)
        self.levels[old_lvl].remove(card)
        if success:  # move card up or release
            if old_lvl + 1 in self.levels:
                self.levels[old_lvl + 1].append(card)
                return True, False
            else:
                return True, True
        else:  # back to start
            self.levels[1].append(card)
            return True, False

    def get_key_of_card(self, card):
        reverse_lookup = {k: v for v, l in self.levels.items() for k in l}
        return reverse_lookup.get(card)

    def onboard_card(self, hash_, level=1):
        my_assert(hash_, str, 'hash')
        my_assert(level, int, 'level')

        if (level > self.n_levels) or (level < 1):
            logger.warning(f'this vokibox has only levels 1-{self.n_levels} but not level {level}')
            return False

        if self.n_cards() >= self.max_cards:
            logger.warning(f'cannot onboard card "{card.text}" there are already {self.n_cards()} cards in this box.')
            return False

        if self.check_card_already_in_box(hash_):
            return False

        self.levels[level].append(hash_)
        return True

    def fill(self, hashes: List[str], levels: List = None):
        """
        fills the box with the povided hashes
        """
        my_assert(hashes, list, 'hashes')
        assert all([isinstance(x, str) for x in
                    hashes]), f'only strings allowed in list "hashes", got {[type(h) for h in hashes]}'

        for hash_ in hashes:
            self.onboard_card(hash_)


    def get_all_cards(self):
        return list(chain.from_iterable(self.levels.values()))

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
            for hash_ in lvl:
                strlist.append(f'    {hash_}')

        return '\n'.join(strlist)

    def save(self):

        with open(self.fpath, 'wb') as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(fpath):
        with open(fpath, 'rb') as fh:
            return pickle.load(fh)

    def check_card_already_in_box(self, hash_):
        for i, lvl in self.levels.items():
            if hash_ in lvl:
                logger.warning(f'Card {hash_} is already in this box at level {i}')
                return True
        return False

    def initiate(self):
        self.levels = {i: [] for i in range(1, self.n_levels + 1)}


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


def main():
    """
    mycomment
    :return:
    """
    pass


if __name__ == "__main__":
    main()
