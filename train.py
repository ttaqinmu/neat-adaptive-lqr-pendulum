from model import *

fit = []
def eval(genomes, config):
    for genome_id, genome in genomes:
        net = NeuralNetwork(genome, config)
        pendulum = Pendulum(
            DT,
            TIME,
            COEFF,
            INIT,
            REFERENCE
        )
        obj = pendulum.train(net)
        fit.append(obj)
        genome.fitness = obj


def train():
    p = Population(config, eval)
    winner = p.selection(5)
    winner.save("model/"+str(p.generation)+"_%.1f"%winner.fitness)

if __name__ == '__main__':
    config = Config('config')
    train()