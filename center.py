#!/usr/bin/python3
# /*******************************************************************************************************************************************************/
#  Chaotic Date: 19 sep
#  Copy left (c) 2015, Mohsen Ramezani  All lefts reserved.
#  Code implemented in Python 3.4.1 [GCC 4.9.1 20140930 (Red Hat 4.9.1-11)] on linux
#  (x86-pc-linux-gnu) Run under a Intel Core i74710HQ CPU @ 3.50GHz  machine with 3.4 GiB RAM.      
#  O(N^1) algorithm for following typical parameters (N=slices=100 000) uses 50.3 MB of RAM and
#  O(N^2) algorithm for following typical parameters (N=slices) only for find gauss linking numbers (the program didn't use hight memory only use cashe !!)

# if you use linking number please give slices<10 000

# for find better help please see the refrace mention in program
# /*******************************************************************************************************************************************************/
##############################################
##########################              import
from scipy.integrate import odeint
#from pylab import *
from numpy import var
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time as tim
#import matplotlib as mpl
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
##############################################
##########################   initial condition
ru              =            np.array(1,float) #free parameter
epsel           =  np.array(1.*10**(-5),float) #scanty for not converge integral
t               =                            0 #null time usec in function
t_max           =                        200.  #maximum time in each run
slices          =                      10000   #slices per each run
ddt=np.array((t_max/np.array(slices,float))**2)
d_t=np.array(t_max/np.array(slices,float))
time            =np.linspace(0.0,t_max,slices) #vector of time
yinit1          =         np.array([1.,1.,1.]) #initial condition
nff             =         np.zeros((3), float) #sum of fft value of all place
y               =  np.zeros((slices,3), float) #place in 3D

acceleration    = np.zeros((slices,3), float)  # acceleration of curve
jerk            = np.zeros((slices,3), float)  # jerk of curve
tortion         = np.zeros( (slices) , float)  # tortion of curve
Itencity        = np.zeros( (slices) , float)  # fluxion of linking numbers per time
I2              = np.zeros( (slices) , float)  # linking number in time for curve

ffy             =  np.zeros((slices,3), float) #fft of place in 3D
intffy          =  np.zeros((slices,3), float) #(0 or 1 for fft with the thereshold) fft of place in 3D
tresh           =  np.array(.01      , float)
sum_intffty     =  np.array( 0               )
seq             =  np.zeros((  1000  ), float ) #sequence of network 
ssm             =  0                           #number of sequence 
f               =  np.zeros((slices,3), float) #vector field for each place
hor             =  np.zeros((slices  ), float) #number of each place bitwine 0 _17
adjenmatrix     =  np.zeros( (17,17)  , float) #adjencency matrix for each ru
##############################################
###########################           function
#######
def cr(x,y):
    return((x[1]*y[2]-x[2]*y[1])**2+(x[2]*y[0]-x[0]*y[2])**2+(x[0]*y[1]-x[1]*y[0])**2) # nurm of crossing 2 vectors
#######
def Parallelepiped2(r1,r2,f1,f2):
    dr= r2-r1
    var_var=np.array((dr[1]*f1[2])*f2[0]+(dr[0]*f1[1])*f2[2]+(dr[2]*f1[0])*f2[1]-(dr[2]*f1[1])*f2[0]-(dr[1]*f1[0])*f2[2]-(dr[0]*f1[2])*f2[1])
    return var_var       # value of Parallelepiped  
def Parallelepiped1(dr,f1,f2):
    var_var=np.array((dr[1]*f1[2])*f2[0]+(dr[0]*f1[1])*f2[2]+(dr[2]*f1[0])*f2[1]-(dr[2]*f1[1])*f2[0]-(dr[1]*f1[0])*f2[2]-(dr[0]*f1[2])*f2[1])
    return var_var       # value of Parallelepiped
#######                 rediuce lorenz vector field
def field(y,t):
    return(y[1]-y[0],-y[0]*y[2],y[0]*y[1]-ru)                                             #3D vector field funtion
def df(f,y,t):
    return(f[1]-f[0],-f[0]*y[2]-f[2]*y[0],f[0]*y[1]+f[1]*y[0])                            #3D Derivative per time depended by field
def ddf(a,f,y,t):
    return(a[1]-a[0],-(a[0]*y[2]+2.*f[0]*f[2]+a[2]*y[0]),a[0]*y[1]+2.*f[0]*f[1]+a[1]*y[0])#2 field 3D Derivative per time depended by field
#######
def signf(f=[0.,0.,0.],y=[0,0,0]):
    sf=np.piecewise(f, [f < 0, f >= 0], [0, 1])
    sy=np.piecewise(y, [y < 0, y >= 0], [0, 1])
    return(sf[0]+sf[1]*2+sf[2]*4+sy[2]*8+1)      #sign of field + place(z)
##############################################
###########################       function with loop
####### writhe number \ref{3D Shape Analysis of Intracranial Aneurysms Using the Writhe Number as a Discriminant for Rupture,A LEXANDRA L AURIC}
def writhe_number(y,slices):
    for ll in range(slices):
        yv=y[ll,:]
        t=time[ll]
        fi=f[ll,:]
        Ar=df(fi,yv,t)
        acceleration[ll]=Ar
        Jr=ddf(Ar,fi,yv,t)
        jerk[ll]=Jr
        #print(jerk[ll,:],acceleration[ll,:],f[ll,:],y[ll,:],time[ll])
        yek=np.ones(3,float)
        tortion[ll]=.5*Parallelepiped1(jerk[ll,:],acceleration[ll,:],f[ll,:])/(Parallelepiped1(yek,acceleration[ll,:],f[ll,:])+d_t)*d_t/np.pi
####### Gauss Linking number \ref{gauss linking number Ricca, Renzo L. Journal of Knot Theory and Its Ramifications 20.10 (2011): 1325-1343.}
def  find_linking_number(y,slices):
    I2[0]=np.array(0,float)
    I2[-1]=np.array(0,float)
    for i in range(slices):
        r1=y[i]
        f1=field(y[i],time[i])
        v=np.array(0.0)
        for ii in range(slices-1):
            r2=y[ii]
            f2=field(y[ii],time[ii])
            vv=Parallelepiped2(r1,r2,f1,f2)
            dr=r2-r1
            ddr=np.dot(dr,dr)
            vv=vv/(ddr+epsel)**1.5
            v=v+vv        
        #Itencity[i]=np.array(v*ddt*slices)
        Itencity[i]=(np.array(v*ddt*slices/(4.*np.pi*np.pi*np.pi)*1.01))
        I2[i]=I2[i-1]+int(Itencity[i])
def print_number(ru,time,I2,tortion,slices):
    for i in range(slices):print(ru,time[i],I2[i],tortion[i],file=file5)
##############################################
#                                   opens file

#file2=open("place_in_graph",'w')
file3=open("matrix2",'w')
file4=open("matrix3",'w')
file1=open("fft_location",'w')
#file5=open("numbers",'w')
file6=open("possible",'w')
file7=open("seq",'w')
##############################################
#                                      program
for ruu in range(1,1000):
    ru=float(ruu)/1000.
    seq=np.zeros((  1000  ), float)
    print(ru,ssm)
    adjenmatrix =  np.zeros( (17,17)  , float)
    for i in range(30):
        y=odeint(field,yinit1,time)
        yinit1=y[-1]
    ########### fft of orbit to find the close orbits!!!!!!!! ###############
    ffy[:,0]=np.absolute(np.fft.fft(y[:,0]))
    ffy[:,1]=np.absolute(np.fft.fft(y[:,1]))
    ffy[:,2]=np.absolute(np.fft.fft(y[:,2]))
    nff[0]=(np.dot(ffy[:,0],ffy[:,0]))**.5
    nff[1]=(np.dot(ffy[:,1],ffy[:,1]))**.5
    nff[2]=(np.dot(ffy[:,2],ffy[:,2]))**.5
    for i in range(slices):
        ffy[i,0]=ffy[i,0]/nff[0]
        ffy[i,1]=ffy[i,1]/nff[1]
        ffy[i,2]=ffy[i,2]/nff[2]
        print(ru,i,ffy[i,0],ffy[i,1],ffy[i,2],y[i,0],y[i,1],y[i,2],file=file1)
    intffy=np.piecewise(ffy, [ffy < tresh, ffy >= tresh], [0, 1])
    sum_intffty=0
    for i in range(50):
        sum_intffty=sum_intffty+intffy[i,0]
    if ( sum_intffty < 10 ):
        print("#######",ru,sum_intffty)
        print(ru,file=file6)
#        find_linking_number(y,slices)
#        writhe_number(y,slices)
#        print_number(ru,time,I2,tortion,slices)
    ssm=0
    for ss in range(slices):
        f[ss,:]=np.array(field(y[ss,:],time[ss]))
        hor[ss]=signf(f[ss,:],y[ss,:])
        if (hor[ss]!=hor[ss-1]):
            seq[ssm%1000]=hor[ss]
            ssm=ssm+1
            #print(hor[ss],file=file2)
            adjenmatrix[hor[ss],hor[ss-1]]=adjenmatrix[hor[ss],hor[ss-1]]+1
    #adjenmatrix=np.piecewise(adjenmatrix, [adjenmatrix <= 0, adjenmatrix > 0], [0, 1])
    for i in range(17):
        for j in range(17):
            print(ru,i,j,adjenmatrix[i,j],file=file3)
            print(ru,i,j,adjenmatrix[i,j],file=file4)
    print(file=file3)
    print(file=file3)
    print(' '.join(str(x) for x in seq),file=file7)
##############################################
#                                   close file
file1.close
#file2.close
file3.close
file4.close
#file5.close
file6.close
file7.close
