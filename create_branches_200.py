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


##########################################
#Compute the homogeneous equilibria
N=200
print(N)
Rains_hom=np.concatenate((np.arange(0,0.98,0.01),np.arange(0.98,1.003,0.0001),np.arange(1.1,1.25,0.01),np.arange(1.25,1.275,0.001),np.arange(1.275,1.5,0.01)))


P_hom=np.zeros((np.shape(Rains_hom)[0],N))
W_hom=np.zeros((np.shape(Rains_hom)[0],N))
O_hom=np.zeros((np.shape(Rains_hom)[0],N))
for i in range(np.shape(Rains_hom)[0]):
    if Rains_hom[i]<1:
        a=1
    else:
        Ph,Wh,Oh=homogeneous(Rains_hom[i])
        P_hom[i]=(np.ones(N)*Ph)
        W_hom[i]=(np.ones(N)*Wh)
        O_hom[i]=(np.ones(N)*Oh)


#import the data
L=200
name=['sols1','sols2','sols3','sols4','sols5','sols6','sols7','sols8','sols9','sols10','sols11']
raw=[]
mode_import=['oneR0','twoR0','threeR0','fourR0','fiveR0','sixR0','sevenR0','eightR0','nineR0','tenR0','elevenR0','twelveR0','thirteenR0','fourteenR0','fifteenR0','sixteenR0','seventeenR0','eighteenR0','nineteenR0','sevenbisR0','eightbisR0']
n_files=[11,6,10,6,6,6,6,6,6,6,6,5,6,5,5,4,5,2,3,4,4]

var=['P','W','O']
name_mode=['Homogeneous solution','n=1','n=2','n=3','n=4','n=5','n=6','n=7','n=8','n=9','n=10','n=11','n=12','n=13','n=14','n=15','n=16','n=17','n=18','n=19','n=7 bis','n=8 bis']
dir_name=['hom','n=1','n=2','n=3','n=4','n=5','n=6','n=7','n=8','n=9','n=10','n=11','n=12','n=13','n=14','n=15','n=16','n=17','n=18','n=19','n=7bis','n=8bis']


N_mode=len(mode_import)

P_mode_tot={}
W_mode_tot={}
O_mode_tot={}
Rains_mode_tot={}

P_mode_tot[dir_name[0]]=P_hom
W_mode_tot[dir_name[0]]=W_hom
O_mode_tot[dir_name[0]]=O_hom
Rains_mode_tot[dir_name[0]]=Rains_hom




#iterate on mode
for j in range(N_mode):
    print('--------------')
    print(dir_name[j+1])
    print('# files: %d'%(n_files[j]))
    raw=[]
    #iterate on variable
    for m in range(3):
        print('Variable: '+var[m])
        raw.append([])
        print('sols1')
        raw[m]=np.loadtxt('L%s/'%(L)+'sols1'+var[m]+mode_import[j]+'.dat')
        #iterate on name
        for i in range(1,n_files[j]):
            print(name[i])
            raw[m]=np.vstack((raw[m],np.loadtxt('L%s/'%(L)+name[i]+var[m]+mode_import[j]+'.dat')))
    P_mode_tot[dir_name[j+1]]=raw[0][:,1:]
    W_mode_tot[dir_name[j+1]]=raw[1][:,1:]
    O_mode_tot[dir_name[j+1]]=raw[2][:,1:]
    Rains_mode_tot[dir_name[j+1]]=raw[0][:,0]



for i in range(len(dir_name)):
    print('Saving files for '+dir_name[i])
    with open("L200/"+dir_name[i]+"/P_mode_tot.txt", "wb") as fp:
        pickle.dump(P_mode_tot[dir_name[i]], fp)
    with open("L200/"+dir_name[i]+"/W_mode_tot.txt", "wb") as fp:
        pickle.dump(W_mode_tot[dir_name[i]], fp)
    with open("L200/"+dir_name[i]+"/O_mode_tot.txt", "wb") as fp:
        pickle.dump(O_mode_tot[dir_name[i]], fp)
    with open("L200/"+dir_name[i]+"/Rains_mode_tot.txt", "wb") as fp:
        pickle.dump(Rains_mode_tot[dir_name[i]], fp)
