import datetime
import numpy as np
import pandas as pd
import os
from pathlib import Path
import pickle


class voki_db:
    def __init__(self, df=None):
        pass


def card_id(card):
    return card['id'][0]


class vokibox():
    def __init__(self, n, max_cards=50):  # box_db,voki_db

        self.levels = {i: [] for i in range(n)}
        self.max_cards = max_cards
        self.n_levels = n

    def onboard_card(self, _id, level=0):
        if level >= self.n_levels:
            print(
                f'this vokibox has only {self.n_levels} levels, so you may specify only up to level index {self.n_levels - 1}. you wanted to onboard a card to level index {level}, ')
            return

        if self.n_ids() >= self.max_cards:
            print(
                f'cannot onboard card {_id} as the maximum number of cards ({self.n_ids}) is already at the maximum allowed cards.')
            return

        if isinstance(_id, pd.DataFrame):
            _id = card_id(_id)

        if not _id in [c_id for lvl in self.levels.values() for c_id in lvl]:
            self.levels[level].append(_id)
        else:
            print(f'card id {_id} already in the box ')

    def attempt_card_in_level(self, lvl, df):
        # attempt the next card in vokibox level i
        if len(self.levels) <= lvl:
            print(f'You cannot attempt level {lvl} as there are only levels 0-{len(self.levels) - 1}')
            return False
        if len(self.levels[lvl]) == 0:
            print(f'vokibox lvl {lvl} is empty. you need to fill it first')
            return

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

    def n_ids(self):
        return len([_id for level in self.levels.values() for _id in level])

    def __repr__(self):
        strlist = []
        for (i, lvl) in self.levels.items():
            strlist.append(f'level {i}')
            for id_ in lvl:
                strlist.append(f'    {id_}')

        return '\n'.join(strlist)


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
    root = Path.cwd()
    data_dir = root / 'data'
    voki_db_file = data_dir / 'voki_db.pkl'
    box_db_file = data_dir / 'box_db.pkl'

    # make a voki db
    df_voki = pd.read_pickle(voki_db_file)
    voki_box = box_db_load(box_db_file)

    card_1 = pd.DataFrame({'french_phrase': ['partout'],
                           'french_sentences': [['Dans la rue du Sentier, la mode est partout.']],
                           'french_comments': [[]],
                           'german_phrase': ['überall'],
                           'german_sentences': [[]],
                           'german_comments': [[]],
                           'created': [datetime.datetime.now().replace(microsecond=0)],
                           'attempt_history': [pd.DataFrame({'time': pd.Series(dtype='datetime64[ns]'),
                                                             'result': pd.Series(dtype='bool'),
                                                             'answer': pd.Series(dtype='str')})],
                           'streak': [0],
                           'id': [hash('partout')]
                           })

    card_2 = pd.DataFrame({'french_phrase': ['la rue'],
                           'french_sentences': [['Dans la rue du Sentier, la mode est partout.']],
                           'french_comments': [[]],
                           'german_phrase': ['Strasse'],
                           'german_sentences': [[]],
                           'german_comments': [[]],
                           'created': [datetime.datetime.now().replace(microsecond=0)],
                           'attempt_history': [pd.DataFrame({'time': pd.Series(dtype='datetime64[ns]'),
                                                             'result': pd.Series(dtype='bool'),
                                                             'answer': pd.Series(dtype='str')})],
                           'streak': [0],
                           'id': [hash('la rue')]
                           })
    df = pd.concat([card_1, card_2])

    myvokibox = vokibox(5)
    myvokibox.onboard_card(card_1)
    myvokibox.onboard_card(card_2)
    print('finish')


if __name__ == "__main__":
    main()
