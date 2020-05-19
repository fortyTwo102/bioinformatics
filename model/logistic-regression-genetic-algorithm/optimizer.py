from functools import reduce
from operator import add
import random
from model import Model
random.seed(a=13361)

class Optimizer():

    def __init__(self, params, retain=0.4,
                 random_select=0.1, mutate_chance=0.2):

        # create an optimizer.

        # args:
        #     params (dict): Possible model paremters
        #     retain (float): Percentage of population to retain after
        #         each generation
        #     random_select (float): Probability of a rejected model
        #         remaining in the population
        #     mutate_chance (float): Probability a model will be
        #         randomly mutated


        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.params = params

    def create_population(self, count):

        # create a population of random models.

        # args:
        #     count (int): Number of models to generate, aka the
        #         size of the population

        # returns:
        #     (list): Population of model objects

        pop = []
        for _ in range(0, count):

            # Create a random model.
            model = Model(self.params)
            model.create_random()

            # Add the model to our population.
            pop.append(model)

        return pop

    @staticmethod
    def fitness(model):

        # Return the accuracy, which is our fitness function.
        return model.accuracy

    def grade(self, pop):

        # find average fitness for a population.

        # args:
        #     pop (list): The population of models

        # returns:
        #     (float): The average accuracy of the population

        summed = reduce(add, (self.fitness(model) for model in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):

        # make two children as parts of their parents.

        # args:
        #     mother (dict): Model parameters
        #     father (dict): Model parameters

        # returns:
        #     (list): Two model objects

        children = []
        for _ in range(2):

            child = {}

            # Loop through the parameters and pick params for the kid.
            for param in self.params:
                child[param] = random.choice(
                    [mother.model[param], father.model[param]]
                )

            # Now create a model object.
            model = Model(self.params)
            model.create_set(child)

            # Randomly mutate some of the children.
            if self.mutate_chance > random.random():
                model = self.mutate(model)

            children.append(model)

        return children

    def mutate(self, model):
        # randomly mutate one part of the model.

        # rrgs:
        #     model (dict): The model parameters to mutate

        # returns:
        #     (Model): A randomly mutated model object

        
        # Choose a random key.
        mutation = random.choice(list(self.params.keys()))

        # Mutate one of the params.
        model.model[mutation] = random.choice(self.params[mutation])

        return model

    def evolve(self, pop):
    	
        # evolve a population of models.

        # args:
        #     pop (list): A list of model parameters

        # returns:
        #     (list): The evolved population of models

        # Get scores for each model.
        graded = [(self.fitness(model), model) for model in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded)*self.retain)

        # The parents are every model we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining models.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

            # Assuming they aren't the same model...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents
