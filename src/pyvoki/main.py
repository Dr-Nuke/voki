import datetime
import numpy as np
import pandas as pd


class card:
    # a class for vocabulary cards

    def __init__(self, props):
        # the basic card that has typical properties of a vocabulary card
        self.french_phrase = props['french']['phrase']
        self.french_sentences =      props['french']['sentences']
        self.french_comments = props['french']['comments']
        self.german_phrase = props['german']['phrase']
        self.german_sentences = props['german']['sentences']
        self.german_comments = props['german']['comments']

        self.created = datetime.datetime.now()
        self.attempt_history = pd.DataFrame({'time': [], 'result': [], 'answer': []})
        self.id = hash(self.french_phrase)

    def attempt(self, answer):
        now = datetime.datetime.now()
        result = self.check_attempt(answer)

        self.attempt_history.append(pd.DataFrame({'time': [now], 'result': [result], 'answer': [answer]}))

    def check_attempt(self,answer):
        if answer == self.french_phrase:
            return True
        text = input(f'type "yes" if you think they match:\n {answer} \n {self.french_phrase}')
        if text == 'yes':
            return True
        return False


