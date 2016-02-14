#! /usr/bin/env python
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import time
import math

def plot_dist(data):

    print "shape", data.shape

    fig = plt.figure()


    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim3d(-0.25, 0.25)
    ax.set_ylim3d(-0.25, 0.25)
    ax.set_zlim3d(-0.25, 0.25)
    ax.view_init(-90, -90)
    xs = data[:,0]
    ys = data[:,1]
    zs = data[:,2]

    ax.scatter(xs, ys, zs, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    angle = 45

    x = np.linspace(-0.25,0.25,10)
    y = np.linspace(-0.25*math.sin(angle),0.25*math.sin(angle),10)
    xv, yv = np.meshgrid(x,y)
    z = np.linspace(0.25*math.cos(angle),-0.25*math.cos(angle),10)
    z = np.repeat([z], 10, axis=0)
    z = np.transpose(z)
    ax.plot_wireframe(xv,yv,z)

    xline=((-0.25,0.25),(0,0),(0,0))
    ax.plot(xline[0],xline[1],xline[2],'grey')
    yline=((0,0),(-0.25,0.25),(0,0))
    ax.plot(yline[0],yline[1],yline[2],'grey')
    zline=((0,0),(0,0),(-0.25,0.25))
    ax.plot(zline[0],zline[1],zline[2],'grey')

    # ax.view_init(-90, -90)
    ax.view_init(-130, -50)


    plt.show()


def test():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # X, Y, Z = axes3d.get_test_data(0.1)

    angle = 55

    x = np.linspace(-0.25,0.25,10)
    y = np.linspace(-0.25*math.sin(angle),0.25*math.sin(angle),10)
    xv, yv = np.meshgrid(x,y)
    z = np.linspace(0.25*math.cos(angle),-0.25*math.cos(angle),10)
    # print z
    z = np.repeat([z], 10, axis=0)
    z = np.transpose(z)
    # print z
    # z = np.zeros([10,10])
    ax.plot_wireframe(xv,yv,z)
    plt.show(block=False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(-130, -50)

    ax.set_xlim3d(-0.25, 0.25)
    ax.set_ylim3d(-0.25, 0.25)
    ax.set_zlim3d(-0.25, 0.25)

    # plt.draw()
    plt.show(block=False)
    plt.draw()
        # time.sleep(1)
    plt.show()

if __name__ == '__main__':
    test()
