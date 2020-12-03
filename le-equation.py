#Author: Pedro Abritta 
#University of South Florida
#Course: Stellar Astrophysics
#Professor: Dr Alexander McCormick

#Program to integrate lane-Emden equation numerically - depends on n (polytopic index) - using Runge-Kutta 4th order method
#LE: y" + 2y'/x (y^n)(x^2) = 0 such ' means d/dx


#-----------------------Necessary Packages---------------------------------------
import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------------

#--------------------------Constants----------------------------------------------

#Defining important constants we will need
step_size = 0.01 #step size
mu = 0.61 #mean atomic mass
mh = 1.66e-27 #atomic mass unit
R = 1.38e-23 #Boltzmann constant

# Core numbers for the Sun
pc = 1.5e5 #density
temp_c = 15e6 #temperature
press_c = 26.5e15 #pressure

#Arrays carrying the polytropes we are solving for
ximax = np.array([3.6537, 9.53581])
poly = np.array([1.5,3.5])
#-----------------------------------------------------------------------------------


#--------------------------------- Functions ----------------------------------------- 
def Pressure(a, n, K):
    aux1 = pow(a, n+1)
    aux2 = pow(pc, 1 + 1/n)
    return aux1*aux2*K/press_c

def Temp(a, n, k):
    aux1 = pow(pc, 1/n)
    return a*aux1*mu*mh*k/(temp_c*R)

def opacity(n):
    a = (n+1)/n
    aux1 = pow(pc, a)
    return press_c/aux1


#given by project set up - it does not need any other parameters, so I will make it a one-parameter function
def yprime(a):
     return a

#given by project set up
def zprime(x,y,z,n):
    aux1 = pow(y,n)
    aux2 = 2*z/x
    aux3 = (-1)*aux1 - aux2
    return aux3


# Because our problem has two differential equations, we need two sets of variables (k,l)
# Because our yprime depends only on z, we are omitting x and y dependence on the RKS, but it does exist
def RKS(xmax, stepsize, polyindex):
    x = np.arange(0.02, xmax, stepsize)
    size = len(x)
    y = np.zeros(size)
    y[0]= 1
    z = np.zeros(size)
    z[0] = 1
    for i in range (0,size-2):
        k1 = stepsize*yprime(z[i])
        l1 = stepsize*zprime(x[i],y[i],z[i],polyindex)
        k2 = stepsize*yprime(z[i] + l1/2)
        l2 = stepsize*zprime(x[i] + stepsize/2, y[i] + k1/2 ,z[i] + l1/2, polyindex)
        k3 = stepsize*yprime(z[i] + l2/2)
        l3 = stepsize*zprime(x[i] + stepsize/2, y[i] + k2/2 ,z[i] + l2/2, polyindex)
        k4 = stepsize*yprime(z[i] + l3)
        l4 = stepsize*zprime(x[i] + stepsize, y[i] + k3 ,z[i] + l3, polyindex)
        y[i+1] = y[i] + (k1/6) + (k2/3) + (k3/3) + (k4/6)
        z[i+1] = z[i] + (l1/6) + (l2/3) + (l3/3) + (l4/6)
    return y

#------------------- Xi arrays -----------------------------------
x15 = np.arange(0.02, ximax[0], step_size)
size15 = len(x15)
xprime15 = x15/ximax[0]

x35 = np.arange(0.02, ximax[1],step_size)
size35 = len(x35)
xprime35 = x35/ximax[1]
#----------------------------------------------------------------

#----------------- Theta arrays ---------------------------------
y115 = RKS(ximax[0],step_size,poly[0])
op15 = opacity(1.5)

y135 = RKS(ximax[1],step_size,poly[1])
op35 = opacity(3.5)
#----------------------------------------------------------------

#------------------ Pressure arrays -----------------------------
press15 = np.zeros(size15)
for i in range(0,size15):
    press15[i] = Pressure(y115[i], 1.5, op15)

press35 = np.zeros(size35)
for i in range(0,size35):
    press35[i] = Pressure(y135[i], 3.5, op35)
#----------------------------------------------------------------

#------------------- Temperature arrays -------------------------
temp15 = np.zeros(size15)
for i in range(0,size15):
    temp15[i] = Temp(y115[i], 1.5, op15)

temp35 = np.zeros(size35)
