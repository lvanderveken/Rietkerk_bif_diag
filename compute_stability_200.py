import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from mpl_toolkits.mplot3d import Axes3D
from static_1Dfunction import *
import pickle

# model parameters
c = 10.0
gmax = 0.05
k1 = 5.0
k2 = 5.0
d = 0.25
alpha = 0.2
w0 = 0.2
rw = 0.2
DP = 0.1
DW = 0.1
DO = 100
param={'c':c,'gmax':gmax,'k1':k1,'d':d,'DP':DP,'alpha':alpha,'k2':k2,'w0':w0,'rw':rw,'DW':DW,'DO':DO}
L=200


#import
name_mode=['Homogeneous solution','n=1','n=2','n=3','n=4','n=5','n=6','n=7','n=8','n=9','n=10','n=11','n=12','n=13','n=14','n=15','n=16','n=17','n=18','n=19','n=7 bis','n=8 bis']
dir_name=['hom','n=1','n=2','n=3','n=4','n=5','n=6','n=7','n=8','n=9','n=10','n=11','n=12','n=13','n=14','n=15','n=16','n=17','n=18','n=19','n=7bis','n=8bis']



P_mode_tot={}
W_mode_tot={}
O_mode_tot={}
Rains_mode_tot={}

for i in range(len(dir_name)):
    with open("L200/"+dir_name[i]+"/P_mode_tot.txt", "rb") as fp:
        P_mode_tot[dir_name[i]]=pickle.load(fp)
    with open("L200/"+dir_name[i]+"/W_mode_tot.txt", "rb") as fp:
        W_mode_tot[dir_name[i]]=pickle.load(fp)
    with open("L200/"+dir_name[i]+"/O_mode_tot.txt", "rb") as fp:
        O_mode_tot[dir_name[i]]=pickle.load(fp)
    with open("L200/"+dir_name[i]+"/Rains_mode_tot.txt", "rb") as fp:
        Rains_mode_tot[dir_name[i]]=pickle.load(fp)
                                                            



#Choose the branches for wich you want to compute the stability
stab_name=['hom','n=1','n=2','n=3','n=4','n=5','n=6','n=7','n=8','n=9','n=10','n=11','n=12','n=13','n=14','n=15','n=16','n=17','n=18','n=19','n=7bis','n=8bis']
#stab_name=['hom']
Stab_mode_tot={}
Lmb_mode_tot={}

for i in range(len(stab_name)):
    print('Computing the stability for the branch '+stab_name[i])
    print('# equilibria along the branch %d'%(np.shape(Rains_mode_tot[stab_name[i]])[0]))
    Stab,Lmb=stability_range(P_mode_tot[stab_name[i]],W_mode_tot[stab_name[i]],O_mode_tot[stab_name[i]],Rains_mode_tot[stab_name[i]],L,param)
    Stab_mode_tot[stab_name[i]]=Stab
    Lmb_mode_tot[stab_name[i]]=Lmb
    


for i in range(len(stab_name)):
    print('Saving files for '+stab_name[i])
    with open("L200/"+stab_name[i]+"/Stab_mode_tot.txt", "wb") as fp:
        pickle.dump(Stab_mode_tot[stab_name[i]], fp)
    with open("L200/"+stab_name[i]+"/Lmb_mode_tot.txt", "wb") as fp:
        pickle.dump(Lmb_mode_tot[stab_name[i]], fp)
        
