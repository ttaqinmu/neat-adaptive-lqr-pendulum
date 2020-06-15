from model import *

def run(name):
    pendulum = Pendulum(
        DT,
        TIME,
        COEFF,
        INIT,
        REFERENCE
    )
    with open('model/'+name, 'rb') as g:
        genome = pickle.load(g)
    net = NeuralNetwork(genome, config)
    data = pendulum.run_adaptive(net)
    #genome.draw(config)
    thread = Thread(target=pendulum.simulation, args=([data]))
    thread.start()
    pendulum.plot(data)
    #print(pendulum.A)
    #print(pendulum.B)
    print(pendulum.obj_func())

if __name__ == '__main__':
    config = Config('config')
    run('5_31.1') # input 7 best