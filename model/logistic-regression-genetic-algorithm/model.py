import random
import logging
from train import train_and_score
random.seed(a=13361)

class Model():

    def __init__(self, params=None):

        self.accuracy = 0.
        self.params = params
        self.model = {}  # (dic): represents Model parameters

    def create_random(self):

        for key in self.params:
            self.model[key] = random.choice(self.params[key])

    def create_set(self, model):

        self.model = model

    def train(self, dataset):

        if self.accuracy == 0.:
            self.accuracy = train_and_score(self.model, dataset)

    def print_model(self):

        logging.info(self.model)
        logging.info("Model accuracy: %.2f%%" % (self.accuracy * 100))
