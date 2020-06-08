import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import matplotlib.pyplot as plt
import pygame
import pickle
from numpy import matrix, linspace, array, size, cos, sin, mean, absolute, where
from math import pi, sqrt
from lqr import lqr
from time import sleep
from neat import *

# Adaptive LQR with Neuroevolution Augmented of Topology in Inverted Double Pendulum Control

'''
Parameter of mechanical and motion system
w/ assumtion.   -no friction at joints
                -no motor dc
                -moment of inertia is calculated automatically w/ lagrange
b -> friction
M -> cart mass
m1 -> pendulum buttom mass
m2 -> pendulum up mass
l1 -> length of link 1 (cart to pendulum 1)
l2 -> length of link 2 (pendulum 1 to pendulum 2)
g -> gravity KEEP THIS VARIABLE POSITIVE
'''
b = 0.6
M = 10
m1 = 2
m2 = 2
l1 = 2
l2 = 2
g = 9.8

COEFF = {
    'b':b,
    'M':M,
    'm1':m1,
    'm2':m2,
    'l1':l1,
    'l2':l2,
    'g':g
}

'''
Parameter for time response
'''
TIME = 10
DT = 0.1
FULL = int(TIME/DT)
TIMES = linspace(0,TIME,FULL)

X_INIT = -1.5
T1_INIT = 0.1
T2_INIT = 0.1
INIT = [X_INIT, 0, T1_INIT, 0, T2_INIT, 0]

X_REF = 1.5
T1_REF = 0
T2_REF = 0
REFERENCE = [X_REF, 0, T1_REF, 0, T2_REF, 0]


class Pendulum:
    def __init__(self, dt, end, coeff, init_conds, reference):
        self.dt = dt
        self.t = 0.0
        self.x = init_conds[:]
        self.reference = matrix(reference).T
        self.end = end
        self.f = 0.0
        self.force = []
        self.times = linspace(0, end, int(end/dt))
        self.J = 0.0
        self.coeff = coeff
        self.A = []
        self.B = []
        self.C = []
        self.D = []
        self.Q = []
        self.R = []
        self._set_matrix()
        self.is_train = False
        self.net = None
        
    def get_coeff(self):
        b = self.coeff['b']
        M = self.coeff['M']
        m1 = self.coeff['m1']
        m2 = self.coeff['m2']
        l1 = self.coeff['l1']
        l2 = self.coeff['l2']
        g = self.coeff['g']

        mbar = m1+m2
        L = l1*l2

        D1 = (M-m1*M+m1-m2**2)*L
        D2 = l1*(m1+m2)*(m1**2-0.5*M+0.5*m1*M-m1+m1*m2)
        D3 = l2*(m1*2*m2)*(m1**2-0.5*M+0.5*m1*M-m1+m1*m2+(0.76*m2-0.5*M-0.75*m1+0.25*m1*m2))
        return b,M,m1,m2,l1,l2,g, mbar, L, D1, D2, D3
    
    def _set_matrix(self):
        b, M, m1, m2, l1, l2, g, mbar, L, D1, D2, D3 = self.get_coeff()
        self.A = matrix([
            [0, 1, 0, 0, 0, 0],
            [0, -b*(m1*L)/D1, 0, 0, (-m2*g)/D1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, (-b*(.5*m1**2-.5*m1+m1*m2))/D2, (0.5*m1**3*g + 0.5*m1**2*M*g + m1*m2*M*g)/D2, 0, (m2**2*g + 1.5*m1*m2*g + 0.5*m1**2*g - 0.5*m1**2*m2*g + 0.5*m1*M*g + m2*M*g)/D2, 0],
            [0, 0, 0, 0, 0, 1],
            [0, (-b*(0.25*m1+0.5*m2)*m1*.00001)/D3, ((m1**2+0.5*m1*M+m1*m2-2*m2**2-m2*M)*m1*g)/D3, 0, ((-m1**2-0.5*m1*M-3*m1*m2-2*m2**2-m2*M)*m1*g)/D3, 0]
        ])
        self.B = matrix([
            [0],
            [m1*L/D1],
            [0],
            [(0.5*m1**2 - 0.5*m1 + m1*m2)/D2],
            [0],
            [((0.25*m1+0.5*m2)*m1*.00001)/D3]
        ])
        self.C = matrix([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ])
        self.D = 0

    def cost(self):
        '''
        Calculate J Cost Function from Q and R
        '''
        x_t = []
        i = 0
        for x_, r_ in zip(self.x, self.reference):
            if i % 2 == 0:
                if i == 2 or i == 4:
                    t1 = abs(x_)*10
                    t2 = abs(r_[0,0])*10
                else:
                    t1 = abs(x_)
                    t2 = abs(r_[0,0])
                x_t.append(abs(t1-t2))
            i+=1
        
        #x_t = self.x
        x = matrix(x_t).T
        u = abs(self.f)
        u_t = matrix(u).T
        #print(self.x, self.f)
        tmp = (x_t*x)[0,0]**2 + sqrt(u)
        '''
        if tmp > 10:
            tmp = 10
        '''
        #print((x_t*x)[0,0]**2, sqrt(u))
        self.J += round(tmp, 2)
        
    def reset(self):
        self.J = 0

    def stepinfo(self, T, yout, target):
        RiseTimeLimits=(0.1, 0.9)
        #InfValue = yout[-1]
        InfValue = target
        SettlingTimeThreshold = .02
        tr_lower_index = (where(yout >= RiseTimeLimits[0] * InfValue)[0])[0]
        tr_upper_index = (where(yout >= RiseTimeLimits[1] * InfValue)[0])[0]
        RiseTime = T[tr_upper_index] - T[tr_lower_index]
        inf_margin = (1. - SettlingTimeThreshold) * InfValue
        sup_margin = (1. + SettlingTimeThreshold) * InfValue
        inf_margin = (1. - SettlingTimeThreshold) * InfValue
        for i in reversed(range(T.size)):
            if((yout[i] <= inf_margin) | (yout[i] >= sup_margin)):
                SettlingTime = T[i]
                break
        OverShoot = 100. * (yout.max() - InfValue) / (InfValue - yout[0])
        return RiseTime,SettlingTime,OverShoot

    def average(self, x):
        '''
        Part of rk4_step function
        '''
        x_i, k1, k2, k3, k4 = x
        return x_i + (k1 + 2.0*(k3 + k4) +  k2) / 6.0

    def control(self, u):
        #print(self.x)
        inp = array(self.x)
        out = self.net.feed(inp)
        out = array(out)
        
        for i, item in enumerate(out):
            if item <= 0:
                out[i] = 0.001
            if item > 1000:
                out[i] = 1000
        #print(out)
        
        Q = matrix([
            [out[0], 0, 0, 0, 0, 0],
            [0, out[1], 0, 0, 0, 0],
            [0, 0, out[2], 0, 0, 0],
            [0, 0, 0, out[3], 0, 0],
            [0, 0, 0, 0, out[4], 0],
            [0, 0, 0, 0, 0, out[5]]
        ])
        R = out[6]
        '''
        Q = matrix([
            [100, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 100, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 100, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        R = 0.001
        '''
        
        K,x,e = lqr(self.A, self.B, Q, R)
        
        #K = matrix([out])
        #K = matrix([[316.22776602, 382.78716819, 4228.3768849, 1585.29137973, 554.99105366, 128.4621007]])
        #print(K)
        F = float(-K*(matrix(u).T-self.reference))
        return F

    def derivative(self, u):
        '''
        Differential of Inverted Double Pendulum
        This function is for Runge-Kutte 4th order step
        x1 -> position
        x2 -> x1 d/dy
        x3 -> angle pendulum 1
        x4 = x3 d/dy
        x5 -> angle pendulum 2
        x6 = x5 d/dy
        '''
        F = self.control(u)
        self.f = F

        x1, x2, x3, x4, x5, x6 = u
        x1_dt, x3_dt, x5_dt =  x2, x4, x6

        '''
        Linearization of differential equation from Euler-Lagrange
        approx. -theta^2 = 0
                -sin(theta) = theta
                -cos(theta) = 1
                -theta1*theta2 = 0, except for x6 
        '''
        b, M, m1, m2, l1, l2, g, mbar, L, D1, D2, D3 = self.get_coeff()

        x2_dt = ((-b*x2*(m1*L)) + (-m2*g*x5) + (m1*L*F)) / D1
        x4_dt = ((-b*x2*(.5*m1**2-.5*m1+m1*m2)) + (0.5*m1**3*g + 0.5*m1**2*M*g + m1*m2*M*g)*x3 + (m2**2*g + 1.5*m1*m2*g + 0.5*m1**2*g - 0.5*m1**2*m2*g + 0.5*m1*M*g + m2*M*g)*x5 + (0.5*m1**2 - 0.5*m1 + m1*m2)*F ) / D2
        x6_dt = ((-b*(0.25*m1+0.5*m2)*m1*.00001)*x2 + ((m1**2+0.5*m1*M+m1*m2-2*m2**2-m2*M)*m1*g)*x3 + ((-m1**2-0.5*m1*M-3*m1*m2-2*m2**2-m2*M)*m1*g)*x5 + ((0.25*m1+0.5*m2)*m1*.00001)*F) / D3

        x = [x1_dt, x2_dt, x3_dt, x4_dt, x5_dt, x6_dt]
        return x
    
    def rk4_step(self, dt):
        '''
        Runge-Kutte 4th-order Equation
        '''
        dx = self.derivative(self.x)
        
        '''
        if mean(absolute(dx)) > 100:
            # check if x is too large
            if self.is_train == True:
                self.J = round(mean(absolute(dx))*10, 0)
                #self.cost()
                return False
        '''
        
        k2 = [ dx_i*dt for dx_i in dx ]

        xv = [x_i + delx0_i/2.0 for x_i, delx0_i in zip(self.x, k2)]
        k3 = [ dx_i*dt for dx_i in self.derivative(xv)]

        xv = [x_i + delx1_i/2.0 for x_i,delx1_i in zip(self.x, k3)]
        k4 = [ dx_i*dt for dx_i in self.derivative(xv) ]

        xv = [x_i + delx1_2 for x_i,delx1_2 in zip(self.x, k4)]
        k1 = [self.dt*i for i in self.derivative(xv)]

        self.force.append(self.f)
        self.t += dt
        self.x = list(map(self.average, zip(self.x, k1, k2, k3, k4)))
        
        if self.is_train == True:
            self.cost()
        
        return True

    def integrate(self):
        x = []
        while self.t <= self.end:
            r = self.rk4_step(self.dt)
            if self.is_train == True:
                if r == False:
                    break
            x.append([self.t] + self.x)
        return array(x)

    def plot(self, y=None):
        if y is None:
            y = self.integrate()
        
        times = y[:,0]
        x1 = y[:,1]
        x2 = y[:,3]
        x3 = y[:,5]
        x1_dot = y[:,2]
        x2_dot = y[:,4]
        x3_dot = y[:,6]

        plt.figure()
        plt.plot(times, self.force)
        plt.title('Force on Cart')
        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")

        plt.figure()
        plt.plot(times, x1)
        plt.title('Position')
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")

        plt.figure()
        plt.plot(times, x2)
        plt.title('Angle of Pendulum 1')
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (rad)")

        plt.figure()
        plt.plot(times, x3)
        plt.title('Angle of Pendulum 2')
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (rad)")

        plt.show()

    def simulation(self, y=None):
        if y is None:
            y = self.integrate()

        """
        Parameter fot graphical simulation
        """
        SCREEN_HEIGHT = 400
        SCREEN_WIDTH = 600

        BASE_X = SCREEN_WIDTH/2
        BASE_Y = SCREEN_HEIGHT-50

        WHITE = (255, 255, 255)
        BLACK = (20, 20, 20)
        GREY = (140, 130, 121)
        GROUND = (200, 200, 200)
        GREEN = (143, 153, 62)
        RED = (101, 29, 50)
        BLUE = (0, 151, 169)

        CART_WIDTH = 50
        CART_HEIGHT = 25

        L1 = 100
        L2 = 100

        SCALE = 100

        '''
        Initialize of graphical library
        don't bother!
        '''
        print("Simulation time is",TIME,"second")
        running = True
        pygame.init()
        pygame.font.init()
        font = pygame.font.SysFont('Arial', 15)
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Inverted Double Pendulum')

        x1 = y[:,1]     # position
        x3 = y[:,3]     # angle pendulum 1
        x5 = y[:,5]     # angle pendulum 2

        x1 = x1*SCALE   # scaling factor for graphical simulation
        x3 = pi-x3      # bcs straight-up position is equal to pi
        x5 = pi-x5      # same here
        
        while running:
            for time, pos, theta1, theta2 in zip(TIMES, x1, x3, x5):
                screen.fill(WHITE)

                time_txt = font.render("Time : %.1fs"%time, 1, (0, 0, 0))
                screen.blit(time_txt, (10, 10))

                x1_txt = font.render("Cart Position : %.1f m"%(pos/SCALE), 1, (0, 0, 0))
                screen.blit(x1_txt, (10, 30))

                x1_txt = font.render("Angle Pendulum 1 : %.0f°"%(180-(theta1 * 180/pi)), 1, (0, 0, 0))
                screen.blit(x1_txt, (10, 50))

                x1_txt = font.render("Angle Pendulum 2 : %.0f°"%(180-(theta2 * 180/pi)), 1, (0, 0, 0))
                screen.blit(x1_txt, (10, 70))

                cart_x = int((BASE_X - CART_WIDTH/2) + pos)

                link_1_x1 = int(cart_x + CART_WIDTH/2)
                link_1_y1 = int(BASE_Y + CART_HEIGHT/2)
                link_1_x2 = int(link_1_x1 + L1*sin(theta1))
                link_1_y2 = int(link_1_y1 + L1*cos(theta1))

                link_2_x1 = link_1_x2
                link_2_y1 = link_1_y2 
                link_2_x2 = int(link_1_x2 + L2*sin(theta2))
                link_2_y2 = int(link_1_y2 + L2*cos(theta2))

                pygame.draw.line(screen, GROUND, (int(SCALE*X_INIT+BASE_X), link_1_y1-5), (int(SCALE*X_INIT+BASE_X), link_1_y1+5), 3) # start
                pygame.draw.line(screen, GROUND, (int(SCALE*X_REF+BASE_X), link_1_y1-5), (int(SCALE*X_REF+BASE_X), link_1_y1+5), 3)   # stop
                pygame.draw.line(screen, GROUND, (0, link_1_y1), (SCREEN_WIDTH, link_1_y1), 2)                              # ground
                pygame.draw.rect(screen, RED, (cart_x, BASE_Y, CART_WIDTH, CART_HEIGHT))                                    # cart
                pygame.draw.line(screen, GREY, (link_1_x1, link_1_y1), (link_1_x2, link_1_y2), 2)                           # link 1
                pygame.draw.line(screen, GREY, (link_2_x1, link_2_y1), (link_2_x2, link_2_y2), 2)                           # link 2
                pygame.draw.circle(screen, GREEN, (link_1_x2, link_1_y2), 10)                                               # pendulum 1
                pygame.draw.circle(screen, BLUE, (link_2_x2, link_2_y2), 10)                                                # pendulum 1
                
                pygame.display.flip()

                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        running = False
                        break

                sleep(DT)

        pygame.quit()

    def train(self, net):
        self.is_train = True
        self.net = net
        ret = self.integrate()
        return self.J

    def run_adaptive(self, net):
        self.net = net
        ret = self.integrate()
        return ret



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
        pendulum.reset()
        cost = pendulum.train(net)
        #fitness = score
        #print ("\tGenome ID:", genome_id, "| Fitness:", fitness)
        #print(genome_id, cost)
        fit.append(cost)
        genome.fitness = cost


def train():
    p = Population(config, eval)
    winner = p.selection(5)
    winner.save("model/"+str(p.generation)+"_%.1f"%winner.fitness)
    #p.save("generation_"+str(p.generation))
    plt.plot(range(0, len(fit)), fit)
    plt.show()

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
    #print(pendulum.stepinfo(data[:,0], data[:,1], 0))
    #genome.draw(config)
    #pendulum.plot(data)
    pendulum.simulation(data)


if __name__ == '__main__':
    config = Config('config')
    #train()
    run('5_64.1')
    
