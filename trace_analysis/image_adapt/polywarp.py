"""
polywarp.py
Trey Wenger - August 2015

Implementation of IDL's polywarp.pro
Shamelessly copied, well tested against IDL procedure

"""
#import sys
import numpy as np
#from scipy.optimize import curve_fit

def polywarp(xy_out,xy_in,degree=3):
    """
    originally polywarp(Xout,Yout,Xin,Yin,degree=3)
    Fit a function of the form
    Xout = sum over i and j from 0 to degree of: kx[i,j] * Xin^j * Yin^i 
    Yout = sum over i and j from 0 to degree of: ky[i,j] * Xin^j * Yin^i
    Return kx, ky
    len(xo) must be greater than or equal to (degree+1)^2
    """
    x_out=xy_out[:,0]
    y_out=xy_out[:,1]
    x_in=xy_in[:,0]
    y_in=xy_in[:,1]
    
    if len(x_in) != len(y_in) or len(x_in) != len(x_out) or len(x_in) != len(y_out):
        print("Error: length of xo, yo, xi, and yi must be the same")
        return

    if len(x_in) < (degree+1.)**2.:
        print ("Error: length of arrays must be greater than (degree+1)^2")
        return

    # ensure numpy arrays
    x_in = np.array(x_in)
    y_in = np.array(y_in)
    x_out = np.array(x_out)
    y_out = np.array(y_out)

    # set up some useful variables
    degree2 = (degree+1)**2
    x = np.array([x_out,y_out])
    u = np.array([x_in,y_in])
    ut = np.zeros([len(x_in),degree2])
    u2i = np.zeros(degree+1)

    for i in range(len(x_in)):
        u2i[0] = 1.
        zz = u[1,i]
        for j in range(1,degree+1):
            u2i[j]=u2i[j-1]*zz

       # print ("u2i",u2i)

        ut[i,0:degree+1] = u2i

        for j in range(1,degree+1):
            ut[i,j*(degree+1):j*(degree+1)+degree+1]=u2i*u[0,i]**j
       # print ("ut",ut)

    uu = ut.T
  #  print( "uu",uu)
    kk = np.dot(np.linalg.inv(np.dot(uu,ut)),uu).T
   # print( "kk",kk)
    #print( "x[0,:]",x[0,:])
    kx = np.dot(kk.T,x[0,:]).reshape(degree+1,degree+1)
   # print ("kx",kx)
    ky = np.dot(kk.T,x[1,:]).reshape(degree+1,degree+1)
    #print ("ky",ky)

    return kx,ky


# def polywarp_apply(P,Q,pts1):
#      deg=len(P)-1
#      dst=np.zeros(np.shape(pts1))
#      dst[:,0]=[np.sum([P[ii,jj]*pts1[kk,0]**jj * pts1[kk,1]**ii for ii in range(deg+1) for jj in range(deg+1)]) for kk in range(len(pts1))]
#      dst[:,1]=[np.sum([Q[ii,jj]*pts1[kk,0]**jj * pts1[kk,1]**ii for ii in range(deg+1) for jj in range(deg+1)]) for kk in range(len(pts1))]
#      return(dst)


def polywarp_apply(P,Q,pts1):
     deg=len(P)-1
     dst=np.ones(np.shape(pts1))
     dst[:,0]=[np.sum([P[ii,jj]*pts1[kk,0]**ii * pts1[kk,1]**jj for ii in range(deg+1) for jj in range(deg+1)]) for kk in range(len(pts1))]
     dst[:,1]=[np.sum([Q[ii,jj]*pts1[kk,0]**ii * pts1[kk,1]**jj for ii in range(deg+1) for jj in range(deg+1)]) for kk in range(len(pts1))]
     return(dst)