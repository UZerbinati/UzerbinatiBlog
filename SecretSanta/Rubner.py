from skimage import data
from skimage import color
import matplotlib.pyplot as plt
import numpy as np
from math import log

def LABHistogram(Image,N,M):
        H = [];
        step = 100/N;
        mask = np.logical_and(Image[:,:,0] >= 0, Image[:,:,0] <= step);
        step2 = 255/M;
        mask2 = np.logical_and(Image[mask,1] >= -128, Image[mask,1] <= -128+step2);
        mask3 = np.logical_and(Image[mask][mask2,2] >= -128, Image[mask][mask2,2] <= -128+step2);
        Pixels = (Image[mask][mask2][mask3,2]).size;
        H.append(Pixels);
        for j in range(1,M):
                mask3 = np.logical_and(Image[mask][mask2,1] > -128+step2*j, Image[mask][mask2,1] <= -128+step2*(j+1));
                Pixels = (Image[mask][mask2][mask3,2]).size;
                H.append(Pixels);
        for i in range(1,M):
            mask2 = np.logical_and(Image[mask,1] > -128+step2*i, Image[mask,1] <= -128+step2*(i+1));
            mask3 = np.logical_and(Image[mask][mask2,2] >= -128, Image[mask][mask2,2] <= -128+step2);
            Pixels = (Image[mask][mask2][mask3,2]).size;
            H.append(Pixels);
            for j in range(1,M):
                mask3 = np.logical_and(Image[mask][mask2,1] > -128+step2*j, Image[mask][mask2,1] <= -128+step2*(j+1));
                Pixels = (Image[mask][mask2][mask3,2]).size;
                H.append(Pixels);
                
        for n in range(1,N):
            mask = np.logical_and(Image[:,:,0] > step*n, Image[:,:,0] <= step*(n+1));
            mask2 = np.logical_and(Image[mask,1] >= -128, Image[mask,1] <= -128+step2);
            mask3 = np.logical_and(Image[mask][mask2,2] >= -128, Image[mask][mask2,2] <= -128+step2);
            Pixels = (Image[mask][mask2][mask3,2]).size;
            H.append(Pixels);
            for j in range(1,M):
                mask3 = np.logical_and(Image[mask][mask2,1] > -128+step2*j, Image[mask][mask2,1] <= -128+step2*(j+1));
                Pixels = (Image[mask][mask2][mask3,2]).size;
                H.append(Pixels);
            for i in range(1,M):
                mask2 = np.logical_and(Image[mask,1] > -128+step2*i, Image[mask,1] <= -128+step2*(i+1));
                mask3 = np.logical_and(Image[mask][mask2,2] >= -128, Image[mask][mask2,2] <= -128+step2);
                Pixels = (Image[mask][mask2][mask3,2]).size;
                H.append(Pixels);
                for j in range(1,M):
                    mask3 = np.logical_and(Image[mask][mask2,1] > -128+step2*j, Image[mask][mask2,1] <= -128+step2*(j+1));
                    Pixels = (Image[mask][mask2][mask3,2]).size;
                    H.append(Pixels);
        #assert(sum(H)==Image[:,:,1].size)
        assert(len(H)==N*M*M)
        return H;
    
def AverageHistogram(H,tol):
    AverageH = [];
    for i in range(0,len(H)):
        if H[i] > tol:
            AverageH.append(H[i])
    return AverageH;
def MinkowskiHDistance(H,K,r):
    assert((len(H)==len(K)));
    S = 0.0;
    for i in range(0,len(H)):
        S = S+abs(H[i]-K[i])**r;
    S = S**(1/r);
    return S;
def JefferyDivergence(H,K,tolH,tolK):
    S = 0.0;
    assert(len(H)==len(K));
    for i in range(0,len(H)):
        if (H[i] > tolH and K[i] > tolK):
            M = (H[i]+K[i])/2;
            S = S + H[i]*log(H[i]/M)+K[i]*log(K[i]/M);
    return S;
def Kolmogorov(H,K):
    Hc = [];
    for i in range(0,len(H)):
        Hc.append(sum(H[0:i]));
    Kc = [];
    for i in range(0,len(K)):
        Kc.append(sum(K[0:i]));
    S=0.0;
    for i in range(0,len(K)):
        S = S +abs(Hc[i]-Kc[i]);
    return S;
