import logging
from optimizer import Optimizer
from tqdm import tqdm
import random
random.seed(a=13361)

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log4.txt'
)

def train_models(models, dataset):

    pbar = tqdm(total=len(models))
    for model in models:
        model.train(dataset)
        pbar.update(1)
    pbar.close()

def get_average_accuracy(models):

    total_accuracy = 0
    for model in models:
        total_accuracy += model.accuracy

    return total_accuracy / len(models)

def generate(generations, population, params, dataset):

    optimizer = Optimizer(params)
    models = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # Train and get accuracy for networks.
        train_models(models, dataset)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(models)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            models = optimizer.evolve(models)

    # Sort our final population.
    models = sorted(models, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_models(models[:5])

    return models[0]

def print_models(models):

    logging.info('-'*80)
    for model in models:
        model.print_model()

def main():

    generations = 100  # Number of times to evole the population.
    population = 100  # Number of networks in each generation.
    dataset = 'ILPD'

    params = {
    	'C' : [0.1, 1, 10, 100, 1000, 10000, 100000],
    	'tol' : [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'lbfgs','newton-cg']
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    optimum_model = generate(generations, population, params, dataset)

    optimum_model.print_model()

if __name__ == '__main__':
    main()
