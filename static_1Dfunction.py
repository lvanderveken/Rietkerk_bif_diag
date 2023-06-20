import numpy as np
from scipy.linalg import toeplitz
import copy
from scipy.integrate import simps
import matplotlib.pyplot as plt

def Diff_Fourier(N,h):
    '''
    First Differentiation Fourier matrix on a periodic domain [-pi,pi]
    '''
    column=np.insert(0.5*(-1)**(np.arange(1,N))*(1/np.tan((np.arange(1,N))*h/2)),0,0)
    D=toeplitz(column,column[np.insert(np.arange(N-1,0,-1),0,0)])
    return(D)

def Diff2_Fourier(N,h):
    '''
    Second Differentiation Fourier matrix on a periodic domain [-pi,pi]
    '''
    column=np.insert(-0.5*(-1)**np.arange(1,N)/np.sin(h*(np.arange(1,N)/2))**2,0,-np.pi**2/(3*h**2)-1/6)
    D2=toeplitz(column)
    return(D2)


def Op(L,N,Rain,P,W,O,param):
    #N = max(P.shape) #P.shape[0][0]
    h=2*np.pi/N
    #print(P.shape)
    #print(N)
    D2=Diff2_Fourier(N,h)
    # some intermediate computations
    f1 = (param['alpha']*(param['k2']*param['w0'] + P))/(param['k2'] + P)
    f2 = (param['gmax']*param['k1']*P)/(param['k1'] + W)**2
    f3 = (param['gmax']*W)/(param['k1'] + W)
    f4 = -((param['k2']*(-1 + param['w0'])*param['alpha']*O)/(param['k2'] + P)**2)
    
    dF_PdP=param['c']*f3-param['d']
    dF_PdW=param['c']*f2
    dF_PdO=np.zeros(N)
    dF_WdP=f4-f3
    dF_WdW=-param['gmax']*(param['k1']/(W+param['k1'])**2)
    dF_WdW=-param['rw']-f2
    dF_WdO=f1
    dF_OdP=-f4
    dF_OdW=np.zeros(N)
    dF_OdO=-f1
    # initialise operator matrix
    #print(dF_PdP)
    operator = np.zeros((3*N,3*N))
    # eq1
    operator[:N,:N] = np.diag(dF_PdP,0) + param['DP']*(2*np.pi/L)**2*D2
    operator[:N,N:2*N] = np.diag(dF_PdW,0) 
    operator[:N,2*N:3*N] = np.diag(dF_PdO,0)
    # eq2
    operator[N:2*N,:N] = np.diag(dF_WdP,0)
    operator[N:2*N,N:2*N] = np.diag(dF_WdW,0) + param['DW']*(2*np.pi/L)**2*D2
    operator[N:2*N,2*N:3*N] = np.diag(dF_WdO,0)
    # eq3
    operator[2*N:3*N,:N] = np.diag(dF_OdP,0)
    operator[2*N:3*N,N:2*N] = np.diag(dF_OdW,0)
    operator[2*N:3*N,2*N:3*N] = np.diag(dF_OdO,0) + param['DO']*(2*np.pi/L)**2*D2
    return(operator)


def static_eqs(L,N,Rain,P,W,O,param):
    h=2*np.pi/N
    D2=Diff2_Fourier(N,h)
    # some previous computations
    f1 = (param['gmax']*P*W)/(param['k1'] + W)
    f2 = (param['alpha']*O*(param['k2']*param['w0']+ P))/(param['k2'] + P)

    # evaluate equations
    eq1 = -param['d']*P + param['c']*f1 + param['DP']*(2*np.pi/L)**2*(D2@P)
    eq2 = f2 - param['rw']*W - f1 + param['DW']*(2*np.pi/L)**2*(D2@W)
    eq3 =  Rain- f2 + param['DO']*(2*np.pi/L)**2*(D2@O)
    
    # put together in a vector
    source = np.concatenate((eq1,eq2,eq3))
    return(source)

def homogeneous(Rain):
    # homogeneous solution for given value of Rain
    Ph = 40.0*(Rain-1)
    Wh = 5.0
    Oh = (Rain*(-4.375 + 5.0*Rain))/(-0.975 + Rain)
        
    return(Ph,Wh,Oh)

def initialGuess(Rain,kappa,mode="L",eps=0.5,n_lmb=1,N=100):
    '''
    Create the initial guess from a given rain and kappa with a choice of lower or upper branch
    '''
    
    # homogeneous solution for given value of Rain
    Ph,Wh,Oh = homogeneous(Rain)
    
    # load perturbation data from interpolated functions
    if mode == 'U':
        k = np.interp(Rain,kappa[:,0],kappa[:,1])
        dP = 1.0
        dW = np.interp(Rain,kappa[:,0],kappa[:,3])
        dO = np.interp(Rain,kappa[:,0],kappa[:,5])
        
    elif mode == 'L':
        k = np.interp(Rain,kappa[:,0],kappa[:,2])
        dP = 1.0
        dW = np.interp(Rain,kappa[:,0],kappa[:,4])
        dO = np.interp(Rain,kappa[:,0],kappa[:,6])
    
    # construct domain
    L = n_lmb*2*np.pi/k
    h = 2*np.pi/N
    x =np.arange(0,2*np.pi,h)
    D = Diff_Fourier(N,h) # differentiation matrix
    D2 = D@D
    
    # construct initial guess
    P = Ph + eps*dP*np.cos(n_lmb* x)
    W = Wh + eps*dW*np.cos(n_lmb* x)
    O = Oh + eps*dO*np.cos(n_lmb* x)
    
    return(Rain,L,N,D2,x,P,W,O,k)

def newton(L,N,Rain,P,W,O,eps_newton,param):
    h=2*np.pi/N
    source = -static_eqs(L,N,Rain,P,W,O,param);
    op = Op(L,N,Rain,P,W,O,param);
    update = np.linalg.solve(op,source)

    Pnew = P + eps_newton*update[:N]
    Wnew = W + eps_newton*update[N:2*N]
    Onew = O + eps_newton*update[2*N:3*N]
    return(Pnew,Wnew,Onew)

def newton_iterate(iterations,Rain,L,param,Pg,Wg,Og,crit_update = 1e-6,crit_source = 1e-10):
    eps_newton = 0.01
    N=Pg.shape[0]
    for it in range(iterations):
        Pnew,Wnew,Onew = newton(L,N,Rain,Pg,Wg,Og,eps_newton,param)
        delta_list = [np.max(np.abs(Pnew - Pg)),np.max(np.abs(Wnew - Wg)),np.max(np.abs(Onew - Og))]
        max_update = np.max(delta_list)
        source = static_eqs(L,N,Rain,Pnew,Wnew,Onew,param)
        max_source = np.abs(np.sum(source)/(3*N))
        if max_update < crit_update and max_source < crit_source:
            #print('max_update \n')
            #print(max_update)
            #print('max_source \n')
            #print(max_source)
            break
        elif it>=30 and max_update<0.1:
            eps_newton = 0.1
        elif it>=50 and max_update<0.01:
            eps_newton = 1.0
        elif max_update > 10:
            #print('max_update \n')
            #print(max_update)
            #print('max_source \n')
            #print(max_source)
            print("failed, Rain=%s"%(Rain))
            break
        elif it==iterations-1:
            print("failed, Rain=%s"%(Rain))
            print('End of iterations')
            break
        Pg, Wg, Og = Pnew, Wnew, Onew
    #print(max_update)
    return Rain,Pg,Wg,Og


def save_RvsPWO(sols,name,var,mode,L):
    if var=='P':
        output=np.zeros((len(sols),sols[0][1].shape[0]+1))
        for i in range(len(sols)):
            output[i]=np.append(sols[i][0],sols[i][1])
        np.savetxt('L%s/'%(L)+name+var+mode+'.dat',output)
    elif var=='W':
        output=np.zeros((len(sols),sols[0][2].shape[0]+1))
        for i in range(len(sols)):
            output[i]=np.append(sols[i][0],sols[i][2])
        np.savetxt('L%s/'%(L)+name+var+mode+'.dat',output)
    elif var=='O':
        output=np.zeros((len(sols),sols[0][3].shape[0]+1))
        for i in range(len(sols)):
            output[i]=np.append(sols[i][0],sols[i][3])
        np.savetxt('L%s/'%(L)+name+var+mode+'.dat',output)
    else:
        print('Unknown variable')


def find_kappa_U(L,n,min_Rain,max_Rain,kappa):
    valmax = max_Rain
    valmin = min_Rain
    val = (valmax+valmin)/2
    while abs(2*np.pi/np.interp(val,kappa[:,0],kappa[:,1]) - L/n) > 1e-10:
        if np.interp(val,kappa[:,0],kappa[:,1]) > n*2*np.pi/L:
            valmax = val
        else: 
            valmin = val
        val = (valmax+valmin)/2
    return(val)

def find_kappa_L(L,n,min_Rain,max_Rain,kappa):
    valmax = max_Rain
    valmin = min_Rain
    val = (valmax+valmin)/2
    while abs(2*np.pi/np.interp(val,kappa[:,0],kappa[:,2]) - L/n) > 1e-10:
        if np.interp(val,kappa[:,0],kappa[:,2]) > n*2*np.pi/L:
            valmax = val
        else: 
            valmin = val
        val = (valmax+valmin)/2
    return(val)


def water_uptakeII_Riet(W,gmax,phi):
    return(gmax*(W/(W+phi)))

def FpII_Riet(P,W,O,param):
    y=param['c']*water_uptakeII_Riet(W,param['gmax'],param['k1'])*P-param['d']*P
    return(y)

def FwII_Riet(P,W,O,param):
    y=param['alpha']*O*((P+param['k2']*param['w0'])/(P+param['k2']))-water_uptakeII_Riet(W,param['gmax'],param['k1'])*P-param['rw']*W
    return(y)

def FoII_Riet(P,O,R,param):
    y=R-param['alpha']*O*((P+param['k2']*param['w0'])/(P+param['k2']))
    return(y)





def VegModelII_Riet_Spec_1D_end_02pi(L,N,M,tmax,dt,prec,P0,W0,O0,param):
    dx=2*np.pi/(N-1)
    x=np.linspace(0,2*np.pi,N)
    M=np.shape(prec)[0]
    #Wave numbers
    #print(N)
    kx=2*np.pi*np.fft.fftfreq(N,dx)
    #dealiasing
    k_Nyq=np.abs(kx[int((N)/2)])
    k_23=2/3*k_Nyq
    msk_23=np.zeros((N))
    #print(msk_23)
    for i in range(N):
        if kx[i]<k_23:
            msk_23[i]=1
    Qmat=kx**2
    P=copy.copy(P0)
    W=copy.copy(W0)
    O=copy.copy(O0)
    Pold=copy.copy(P)
    #print(P.shape)
    for i in range(M):
        #Implicit integration in Fourier space
        p=np.fft.fft(P)
        w=np.fft.fft(W)
        o=np.fft.fft(O)
        p=p/(1+dt*Qmat*param['DP']*(2*np.pi/L)**2)
        w=w/(1+dt*Qmat*param['DW']*(2*np.pi/L)**2)
        o=o/(1+dt*Qmat*param['DO']*(2*np.pi/L)**2)
        #Non-linear part
        P=np.real(np.fft.ifft(p))+dt*FpII_Riet(P,W,O,param)
        W=np.real(np.fft.ifft(w))+dt*FwII_Riet(P,W,O,param)
        O=np.real(np.fft.ifft(o))+dt*FoII_Riet(P,O,prec[i],param)
        # Bounce solution back from 0
        P[P<0]=0
        W[W<0]=0
        O[O<0]=0
        if simps(x)/(L)<10**(-4):
            P=Pold
        Pold=P
    return(P,W,O)



def VegModelII_Riet_Spec_1D_02pi(L,N,M,tmax,dt,Dt,prec,P0,W0,O0,param,sigma=0.1,Stoch=False,P_st=1,W_st=1,O_st=1):
    #spatial grid
    dx=2*np.pi/(N)
    x=np.arange(0,2*np.pi,dx)
    #temporal grid
    t=np.linspace(0,tmax,M)
    M=np.shape(prec)[0]
    #Storing time
    #Dt=10*dt
    Mspan=int(np.round(tmax/Dt))+1
    tspan=np.linspace(0,Dt*((Mspan)-1),Mspan)
    Psol=np.zeros((Mspan,N))
    Wsol=np.zeros((Mspan,N))
    Osol=np.zeros((Mspan,N))
    ii=0
    T=0
    #Wave numbers
    #print(N)
    kx=2*np.pi*np.fft.fftfreq(N,dx)
    #dealiasing
    k_Nyq=np.abs(kx[int((N)/2)])
    k_23=2/3*k_Nyq
    msk_23=np.zeros((N))
    #print(msk_23)
    for i in range(N):
        if kx[i]<k_23:
            msk_23[i]=1
    Qmat=kx**2
    P=copy.copy(P0)
    W=copy.copy(W0)
    O=copy.copy(O0)
    Pold=copy.copy(P)
    precspan=np.zeros(Mspan)
    #print(P.shape)
    for i in range(t.shape[0]):
        if t[i]>=T:
            precspan[ii]=prec[i]
            Psol[ii,:]=P
            Wsol[ii,:]=W
            Osol[ii,:]=O
            ii=ii+1
            T=ii*Dt
            #print('T %.2f'%(T))
            #print('ii %.2f'%(ii))
        #Implicit integration in Fourier space
        p=np.fft.fft(P)
        w=np.fft.fft(W)
        o=np.fft.fft(O)
        p=p/(1+dt*Qmat*param['DP']*(2*np.pi/L)**2)
        w=w/(1+dt*Qmat*param['DW']*(2*np.pi/L)**2)
        o=o/(1+dt*Qmat*param['DO']*(2*np.pi/L)**2)
        #Non-linear part
        P=np.real(np.fft.ifft(p))+dt*FpII_Riet(P,W,O,param)
        W=np.real(np.fft.ifft(w))+dt*FwII_Riet(P,W,O,param)
        O=np.real(np.fft.ifft(o))+dt*FoII_Riet(P,O,prec[i],param)
        # Bounce solution back from 0
        P[P<0]=0
        W[W<0]=0
        O[O<0]=0
        if simps(x)/(L)<10**(-4):
            P=Pold
        Pold=P
    return(Psol,Wsol,Osol,precspan,tspan)

def VegModelII_Riet_Spec_1D_02pi_hetero_noise(L,N,M,tmax,dt,Dt,prec,P0,W0,O0,param,sigma=0.1,Stoch=False,P_st=1,W_st=1,O_st=1):
    #spatial grid
    dx=2*np.pi/(N)
    x=np.arange(0,2*np.pi,dx)
    #temporal grid
    t=np.linspace(0,tmax,M)
    M=np.shape(prec)[0]
    #Storing time
    #Dt=10*dt
    Mspan=int(np.round(tmax/Dt))+1
    tspan=np.linspace(0,Dt*((Mspan)-1),Mspan)
    Psol=np.zeros((Mspan,N))
    Wsol=np.zeros((Mspan,N))
    Osol=np.zeros((Mspan,N))
    ii=0
    T=0
    #Wave numbers
    #print(N)
    kx=2*np.pi*np.fft.fftfreq(N,dx)
    #dealiasing
    k_Nyq=np.abs(kx[int((N)/2)])
    k_23=2/3*k_Nyq
    msk_23=np.zeros((N))
    #print(msk_23)
    for i in range(N):
        if kx[i]<k_23:
            msk_23[i]=1
    Qmat=kx**2
    P=copy.copy(P0)
    W=copy.copy(W0)
    O=copy.copy(O0)
    Pold=copy.copy(P)
    precspan=np.zeros(Mspan)
    #print(P.shape)
    for i in range(t.shape[0]):
        if t[i]>=T:
            precspan[ii]=prec[i]
            Psol[ii,:]=P
            Wsol[ii,:]=W
            Osol[ii,:]=O
            ii=ii+1
            T=ii*Dt
            #print('T %.2f'%(T))
            #print('ii %.2f'%(ii))
        #Implicit integration in Fourier space
        p=np.fft.fft(P)
        w=np.fft.fft(W)
        o=np.fft.fft(O)
        p=p/(1+dt*Qmat*param['DP']*(2*np.pi/L)**2)
        w=w/(1+dt*Qmat*param['DW']*(2*np.pi/L)**2)
        o=o/(1+dt*Qmat*param['DO']*(2*np.pi/L)**2)
        #Non-linear part
        P=np.real(np.fft.ifft(p))+dt*FpII_Riet(P,W,O,param)+sigma*P_st*np.random.randn(N)*np.sqrt(dt)
        W=np.real(np.fft.ifft(w))+dt*FwII_Riet(P,W,O,param)+sigma*W_st*np.random.randn(N)*np.sqrt(dt)
        O=np.real(np.fft.ifft(o))+dt*FoII_Riet(P,O,prec[i],param)+sigma*O_st*np.random.randn(N)*np.sqrt(dt)
        # Bounce solution back from 0
        P[P<0]=0
        W[W<0]=0
        O[O<0]=0
        if simps(x)/(L)<10**(-4):
            P=Pold
        Pold=P
    return(Psol,Wsol,Osol,precspan,tspan)

def VegModelII_Riet_Spec_1D_02pi_noise(L,N,M,tmax,dt,Dt,prec,P0,W0,O0,param,sigma=0.1,P_st=1,W_st=1,O_st=1):
    #spatial grid
    dx=2*np.pi/(N)
    x=np.arange(0,2*np.pi,dx)
    #temporal grid
    t=np.linspace(0,tmax,M)
    M=np.shape(prec)[0]
    #Storing time
    #Dt=10*dt
    Mspan=int(np.round(tmax/Dt))+1
    tspan=np.linspace(0,Dt*((Mspan)-1),Mspan)
    Psol=np.zeros((Mspan,N))
    Wsol=np.zeros((Mspan,N))
    Osol=np.zeros((Mspan,N))
    ii=0
    T=0
    #Wave numbers
    #print(N)
    kx=2*np.pi*np.fft.fftfreq(N,dx)
    #dealiasing
    k_Nyq=np.abs(kx[int((N)/2)])
    k_23=2/3*k_Nyq
    msk_23=np.zeros((N))
    #print(msk_23)
    for i in range(N):
        if kx[i]<k_23:
            msk_23[i]=1
    Qmat=kx**2
    P=copy.copy(P0)
    W=copy.copy(W0)
    O=copy.copy(O0)
    Pold=copy.copy(P)
    precspan=np.zeros(Mspan)
    #print(P.shape)
    #Stochastic setup
    for i in range(t.shape[0]):
        if t[i]>=T:
            precspan[ii]=prec[i]
            Psol[ii,:]=P
            Wsol[ii,:]=W
            Osol[ii,:]=O
            ii=ii+1
            T=ii*Dt
            #print('T %.2f'%(T))
            #print('ii %.2f'%(ii))
        #Implicit integration in Fourier space
        p=np.fft.fft(P)
        w=np.fft.fft(W)
        o=np.fft.fft(O)
        p=p/(1+dt*Qmat*param['DP']*(2*np.pi/L)**2)
        w=w/(1+dt*Qmat*param['DW']*(2*np.pi/L)**2)
        o=o/(1+dt*Qmat*param['DO']*(2*np.pi/L)**2)
        #Non-linear part
        P=np.real(np.fft.ifft(p))+dt*FpII_Riet(P,W,O,param)+sigma*P_st*np.random.randn()*np.sqrt(dt)
        W=np.real(np.fft.ifft(w))+dt*FwII_Riet(P,W,O,param)+sigma*W_st*np.random.randn()*np.sqrt(dt)
        O=np.real(np.fft.ifft(o))+dt*FoII_Riet(P,O,prec[i],param)+sigma*O_st*np.random.randn()*np.sqrt(dt)
        # Bounce solution back from 0
        P[P<0]=0
        W[W<0]=0
        O[O<0]=0
        if simps(x)/(L)<10**(-4):
            P=Pold
        Pold=P
    return(Psol,Wsol,Osol,precspan,tspan)


def spectre_fourier(u,dx,ax):
    N=u.shape[0]
    UU=np.fft.fft(u)
    f=np.fft.fftfreq(N,dx)
    if N%2==0:
        f=f[:int(N/2)]
        UU=UU[:int(N/2)]
    else:
        f=f[:int((N+1)/2)]
        UU=UU[:int((N+1)/2)]
    #fig=plt.figure(figsize=(20,20))
    #ax=plt.gca()
    ax.stem(2*np.pi*f,np.abs(UU)/(np.max(np.abs(UU))),basefmt='black',use_line_collection=True)
    ax.set_xlabel('frequency')
    #plt.show()

def spectre_fourier_acc(u,dx,ax,n):
    N=u.shape[0]
    uu=np.zeros(n*N)
    for i in range(n):
        uu[i*N:(i+1)*N]=u
    UU=np.fft.fft(uu)
    f=np.fft.fftfreq(n*N,dx)
    if n*N%2==0:
        f=f[:int(n*N/2)]
        UU=UU[:int(n*N/2)]
    else:
        f=f[:int((n*N+1)/2)]
        UU=UU[:int((n*N+1)/2)]
    #fig=plt.figure(figsize=(20,20))
    #ax=plt.gca()
    ax.stem(2*np.pi*f,np.abs(UU)/(np.max(np.abs(UU))),basefmt='black',use_line_collection=True)
    ax.set_xlabel('frequency')
    #plt.show()

    
    


def VegModelII_Riet_Spec_1D_end_02pi_inv(L,N,M,tmax,dt,prec,P0,W0,O0,param):
    dx=2*np.pi/(N-1)
    x=np.linspace(0,2*np.pi,N)
    M=np.shape(prec)[0]
    #Wave numbers
    #print(N)
    kx=2*np.pi*np.fft.fftfreq(N,dx)
    #dealiasing
    k_Nyq=np.abs(kx[int((N)/2)])
    k_23=2/3*k_Nyq
    msk_23=np.zeros((N))
    #print(msk_23)
    for i in range(N):
        if kx[i]<k_23:
            msk_23[i]=1
    Qmat=kx**2
    P=copy.copy(P0)
    W=copy.copy(W0)
    O=copy.copy(O0)
    Pold=copy.copy(P)
    #print(P.shape)
    for i in range(M):
        #Implicit integration in Fourier space
        p=np.fft.fft(P)
        w=np.fft.fft(W)
        o=np.fft.fft(O)
        p=p/(1-dt*Qmat*param['DP']*(2*np.pi/L)**2)
        w=w/(1-dt*Qmat*param['DW']*(2*np.pi/L)**2)
        o=o/(1-dt*Qmat*param['DO']*(2*np.pi/L)**2)
        #Non-linear part
        P=np.real(np.fft.ifft(p))-dt*FpII_Riet(P,W,O,param)
        W=np.real(np.fft.ifft(w))-dt*FwII_Riet(P,W,O,param)
        O=np.real(np.fft.ifft(o))-dt*FoII_Riet(P,O,prec[i],param)
        # Bounce solution back from 0
        P[P<0]=0
        W[W<0]=0
        O[O<0]=0
        if simps(x)/(L)<10**(-4):
            P=Pold
        Pold=P
    return(P,W,O)


def num_int_stab(Rains,P_mode,W_mode,O_mode,ind,n,eps,param):
    
    #Spatial grid
    L=100
    N=100
    dx=2*np.pi/N
    x=np.arange(0,2*np.pi,dx)
    #Temporal grid
    tmax=6000
    M=tmax*10+1
    dt=tmax/(M-1)
    Dt=10
    #precipitation
    R=Rains[ind]
    prec=R*np.ones(M)
    #initial condition
    lmb,vec=stability_eigen(P_mode[ind],W_mode[ind],O_mode[ind],Rains[ind],L,param)
    P0=P_mode[ind,:][0]+eps*np.real(vec[:N,n])
    W0=W_mode[ind,:][0]+eps*np.real(vec[N:2*N,n])
    O0=O_mode[ind,:][0]+eps*np.real(vec[2*N:,n])
    P_full,W_full,O_full,prec,t=VegModelII_Riet_Spec_1D_02pi(L,N,M,tmax,dt,Dt,prec,P0,W0,O0,param)
    return(P_full,W_full,O_full,x,t)

def show_stability(Rains,P_mode,W_mode,O_mode,Stab,Lmb,ind,name_mode,param,L):
    lmb,vec=stability_eigen(P_mode[ind],W_mode[ind],O_mode[ind],Rains[ind],L,param)
    N=P_mode.shape[1]
    #print(N)
    x=np.arange(0,L,L/N)
    fig,ax=plt.subplots(2,4,figsize=(15,10))
    ax[0,0].plot(Rains,np.real(Lmb),label=name_mode)
    ax[0,0].plot(Rains[ind],np.real(Lmb[ind]),marker='o')
    ax[0,0].legend()
    ax[0,0].set_title('Real(max(eigenvalues))')
    ax[1,0].plot(np.real(lmb)[:20],np.imag(lmb)[:20],marker='o',linestyle='none',markersize=4)
    ax[1,0].set_xlim((-0.25,0.1))
    ax[1,0].set_xlabel('$\Re (\lambda)$')
    ax[1,0].set_ylabel('$\Im (\lambda)$')
    ax[1,0].vlines(0,-1,1,color='black')
    ax[1,0].set_ylim((-0.003,0.003))
    for i in range(3):
        n=i+1
        ax[0,i+1].plot(x,np.real(vec[:N,i]),label='Biomass')
        ax[0,i+1].set_title('n=%s $\lambda $=  %.3f+(%.3f)i '%(n,lmb[i].real,lmb[i].imag))
        ax[0,i+1].plot(x,np.real(vec[N:2*N,i]),label='Soil water')
        ax[0,i+1].plot(x,np.real(vec[2*N:,i]),label='Surface water')
        ax[0,i+1].legend()
        ax[1,i+1].plot(x,np.imag(vec[:N,i]),label='Biomass')
        ax[1,i+1].set_title('n=%s $\lambda $=  %.6f+(%.6f)i '%(n,lmb[i].real,lmb[i].imag))
        ax[1,i+1].plot(x,np.imag(vec[N:2*N,i]),label='Soil water')
        ax[1,i+1].plot(x,np.imag(vec[2*N:,i]),label='Surface water')
        ax[1,i+1].legend()

        
def show_stability_single(Rains,P,W,O,param,L):
    lmb,vec=stability_eigen(P,W,O,Rains,L,param)
    N=P.shape[0]
    #print(N)
    x=np.arange(0,L,L/N)
    #print(x)
    fig,ax=plt.subplots(2,4,figsize=(15,10))
    #ax[0,0].plot(Rains,np.real(Lmb),label=name_mode)
    #ax[0,0].plot(Rains[ind],np.real(Lmb[ind]),marker='o')
    #ax[0,0].legend()
    #ax[0,0].set_title('Real(max(eigenvalues))')
    ax[0,0].plot(np.real(lmb)[:20],np.imag(lmb)[:20],marker='o',linestyle='none',markersize=4)
    ax[0,0].set_xlim((-0.25,0.1))
    ax[0,0].set_xlabel('$\Re (\lambda)$')
    ax[0,0].set_ylabel('$\Im (\lambda)$')
    ax[0,0].vlines(0,-1,1,color='black')
    ax[0,0].set_ylim((-0.003,0.003))
    for i in range(3):
        n=i+1
        ax[0,i+1].plot(x,np.real(vec[:N,i]),label='Biomass')
        ax[0,i+1].set_title(' n=%s $\lambda $=  %.6f+(%.6f)i '%(n,lmb[i].real,lmb[i].imag))
        ax[0,i+1].plot(x,np.real(vec[N:2*N,i]),label='Soil water')
        ax[0,i+1].plot(x,np.real(vec[2*N:,i]),label='Surface water')
        ax[0,i+1].legend()
        ax[1,i+1].plot(x,np.imag(vec[:N,i]),label='Biomass')
        ax[1,i+1].set_title('n=%s $\lambda $=  %.6f+(%.6f)i '%(n,lmb[i].real,lmb[i].imag))
        ax[1,i+1].plot(x,np.imag(vec[N:2*N,i]),label='Soil water')
        ax[1,i+1].plot(x,np.imag(vec[2*N:,i]),label='Surface water')
        ax[1,i+1].legend()

        
def plot_dynamical(Rains_mode_tot,P_mode_tot,Stab_mode_tot,n_mode,ind,P_full,x,L,t,name_mode,color_mode,N_mode):
    P0=P_full[0,:]
    xx,tt=np.meshgrid(x*(L/(2*np.pi)),t)
    fig=plt.figure(figsize=(15,10))
    fig.suptitle('L= %.1f, R = %.5f'%(L,Rains_mode_tot[n_mode][ind]))
    ax1=fig.add_subplot(2,3,1,projection='3d')
    ax1.plot_surface(xx,tt,P_full)
    ax2=fig.add_subplot(2,3,2)
    ax2.plot(x*(L/(2*np.pi)),P0,linewidth=2.,label='initial')
    ax2.plot(x*(L/(2*np.pi)),P_full[-1,:],linewidth=2.,label='final')
    ax2.set_title('Biomass ')
    ax2.legend()
    ax3=fig.add_subplot(2,3,4)
    spectre_fourier_acc(P0,(x[1]-x[0])*(L/(2*np.pi)),ax3,20)
    ax3.set_title('Fourier initial')
    ax4=fig.add_subplot(2,3,5)
    spectre_fourier_acc(P_full[-1,:],(x[1]-x[0])*(L/(2*np.pi)),ax4,20)
    ax4.set_title('Fourier final')
    ax5=fig.add_subplot(2,3,3)
    ax5.plot(Rains_mode_tot[n_mode][ind],np.mean(P0),marker='^',color='orange',linestyle='none',label='Initial')
    ax5.plot(Rains_mode_tot[n_mode][ind],np.mean(P_full[-1,:]),marker='v',color='orange',linestyle='none',label='Final')
    ax6=fig.add_subplot(2,3,6)
    ax6.plot(Rains_mode_tot[n_mode][ind],np.max(P0),marker='^',color='orange',linestyle='none',label='Initial')
    ax6.plot(Rains_mode_tot[n_mode][ind],np.max(P_full[-1,:]),marker='v',color='orange',linestyle='none',label='Final')
    for i in range(N_mode):
        ax5.plot(np.ma.masked_where(True^Stab_mode_tot[i], Rains_mode_tot[i]),np.ma.masked_where(True^Stab_mode_tot[i], np.mean(P_mode_tot[i],axis=1)),marker='o',linestyle='none',markersize=1,color=color_mode[i])
        ax5.plot(np.ma.masked_where(Stab_mode_tot[i], Rains_mode_tot[i]),np.ma.masked_where(Stab_mode_tot[i], np.mean(P_mode_tot[i],axis=1)),markersize=8,color=color_mode[i],label=name_mode[i])
        ax6.plot(np.ma.masked_where(True^Stab_mode_tot[i], Rains_mode_tot[i]),np.ma.masked_where(True^Stab_mode_tot[i], np.max(P_mode_tot[i],axis=1)),marker='o',linestyle='none',markersize=1,color=color_mode[i])
        ax6.plot(np.ma.masked_where(Stab_mode_tot[i], Rains_mode_tot[i]),np.ma.masked_where(Stab_mode_tot[i], np.max(P_mode_tot[i],axis=1)),markersize=8,color=color_mode[i],label=name_mode[i])
    ax5.set_xlim(0.4,1.4)
    ax5.set_ylim(0,20)
    ax5.set_title('Mean Biomass $[g.m^{-2}]$')
    ax5.set_xlabel('Rain $[mm.d^{-1}$]')
    ax5.legend()
    ax6.set_xlim(0.4,1.4)
    ax6.set_ylim(0,20)
    ax6.set_title('Maximum Biomass $[g.m^{-2}]$')
    ax6.set_xlabel('Rain $[mm.d^{-1}$]')
    #plt.savefig('fig1.svg')
    #ax6.legend()
    return(fig)

def plot_dynamical_transient(Rains_mode_tot,P_mode_tot,Stab_mode_tot,n_mode,ind,P_full,R_full,x,L,t,name_mode,color_mode,N_mode):
    P0=P_full[0,:]
    xx,tt=np.meshgrid(x*(L/(2*np.pi)),t)
    fig=plt.figure(figsize=(15,10))
    fig.suptitle('L= %.1f, R = %.5f'%(L,Rains_mode_tot[n_mode][ind]))
    #ax1=fig.add_subplot(2,3,1,projection='3d')
    ax1=fig.add_subplot(2,3,1)
    #ax1.plot_surface(xx,tt,P_full)
    c=np.linspace(-.001,np.max(np.max(P_full)),201)
    contB=ax1.contourf(xx,tt,P_full,c)
    ax1.set_xlabel('x $[m]$')
    ax1.set_ylabel('time $[day]$')
    cax = plt.axes([0.4, 0.4,0.2,0.01])    
    ticks_cbar=np.linspace(0,int(np.max(np.max(P_full))),5)
    cbar=plt.colorbar(contB,cax=cax,orientation='horizontal',ticks=ticks_cbar)
    cbar.ax.set_title('Biomass $[g.m^{-2}]$')
    ax1.set_xlabel('x $[m]$')
    ax1.set_ylabel('time $[day]$')
    #ax1.set_zlabel('Biomass $[g.m^{-2}]$')
    ax2=fig.add_subplot(2,3,2)
    ax2.plot(x*(L/(2*np.pi)),P0,linewidth=2.,label='initial')
    ax2.plot(x*(L/(2*np.pi)),P_full[-1,:],linewidth=2.,label='final')
    ax2.set_xlabel('x $[m]$')
    ax2.set_title('Biomass $[g.m^{-2}]$')
    ax2.legend()
    ax3=fig.add_subplot(2,3,4)
    ax3.plot(t,R_full,label='Rain')
    ax3.set_ylabel('Rain $[mm.d^{-1}$]')
    ax31=ax3.twinx()
    ax31.plot(t,np.mean(P_full,axis=1),color='green',label='Mean Biomass')
    ax31.set_ylabel('Mean biomass $[g.m^{-2}]$')
    ax31.legend(bbox_to_anchor=(1,1))
    ax3.legend(bbox_to_anchor=(1,0.9))
    ax3.set_xlabel('time $[day]$')
    ax5=fig.add_subplot(2,3,3)
    ax6=fig.add_subplot(2,3,6)
    ax5.plot(R_full,np.mean(P_full,axis=1),color='orange',linestyle='solid',marker='D',markersize=3)
    ax5.plot(R_full[0],np.mean(P0),marker='o',color='lime',linestyle='none',label='Initial',markersize=5)
    ax5.plot(R_full[-1],np.mean(P_full[-1,:]),marker='^',color='lime',linestyle='none',label='Final',markersize=5)
    ax6.plot(R_full,np.max(P_full,axis=1),color='orange',linestyle='solid',marker='D',markersize=3)
    ax6.plot(R_full[0],np.max(P0),marker='o',color='lime',linestyle='none',label='Initial',markersize=5)
    ax6.plot(R_full[-1],np.max(P_full[-1,:]),marker='^',color='lime',linestyle='none',label='Final',markersize=5)
    for i in range(N_mode):
        ax5.plot(np.ma.masked_where(True^Stab_mode_tot[i], Rains_mode_tot[i]),np.ma.masked_where(True^Stab_mode_tot[i], np.mean(P_mode_tot[i],axis=1)),linestyle='dashed',markersize=1,color=color_mode[i])
        ax5.plot(np.ma.masked_where(Stab_mode_tot[i], Rains_mode_tot[i]),np.ma.masked_where(Stab_mode_tot[i], np.mean(P_mode_tot[i],axis=1)),markersize=8,color=color_mode[i],label=name_mode[i])
        ax6.plot(np.ma.masked_where(True^Stab_mode_tot[i], Rains_mode_tot[i]),np.ma.masked_where(True^Stab_mode_tot[i], np.max(P_mode_tot[i],axis=1)),linestyle='dashed',markersize=1,color=color_mode[i])
        ax6.plot(np.ma.masked_where(Stab_mode_tot[i], Rains_mode_tot[i]),np.ma.masked_where(Stab_mode_tot[i], np.max(P_mode_tot[i],axis=1)),markersize=8,color=color_mode[i],label=name_mode[i])
    ax5.set_xlim(0.4,1.4)
    ax5.set_ylim(0,20)
    ax5.set_title('Mean Biomass $[g.m^{-2}]$')
    ax5.set_xlabel('Rain $[mm.d^{-1}$]')
    #ax5.legend()
    ax6.set_xlim(0.4,1.4)
    ax6.set_ylim(0,20)
    ax6.set_title('Maximum Biomass $[g.m^{-2}]$')
    ax6.set_xlabel('Rain $[mm.d^{-1}$]')
    #ax6.legend()
    return(fig)

def plot_dynamical_Init_and_End(Rains_mode_tot,P_mode_tot,Stab_mode_tot,n_mode,ind,P_full,R_full,x,L,t,name_mode,color_mode,N_mode):
    P0=P_full[0,:]
    xx,tt=np.meshgrid(x*(L/(2*np.pi)),t)
    fig=plt.figure(figsize=(15,5))
    fig.suptitle('L= %.1f, R = %.5f'%(L,Rains_mode_tot[n_mode][ind]))
    #ax1.set_zlabel('Biomass $[g.m^{-2}]$')
    ax1=fig.add_subplot(1,3,1)
    ax1.plot(x*(L/(2*np.pi)),P0,linewidth=2.,label='initial')
    ax1.plot(x*(L/(2*np.pi)),P_full[-1,:],linewidth=2.,label='final')
    ax1.set_xlabel('x $[m]$')
    ax1.set_title('Biomass $[g.m^{-2}]$')
    ax1.legend()
    ax2=fig.add_subplot(1,3,2)
    ax3=fig.add_subplot(1,3,3)
    ax2.plot(R_full,np.mean(P_full,axis=1),color='orange',linestyle='solid',marker='D',markersize=3)
    ax2.plot(R_full[0],np.mean(P0),marker='o',color='lime',linestyle='none',label='Initial',markersize=5)
    ax2.plot(R_full[-1],np.mean(P_full[-1,:]),marker='^',color='lime',linestyle='none',label='Final',markersize=5)
    ax3.plot(R_full,np.max(P_full,axis=1),color='orange',linestyle='solid',marker='D',markersize=3)
    ax3.plot(R_full[0],np.max(P0),marker='o',color='lime',linestyle='none',label='Initial',markersize=5)
    ax3.plot(R_full[-1],np.max(P_full[-1,:]),marker='^',color='lime',linestyle='none',label='Final',markersize=5)
    for i in range(N_mode):
        ax2.plot(np.ma.masked_where(True^Stab_mode_tot[i], Rains_mode_tot[i]),np.ma.masked_where(True^Stab_mode_tot[i], np.mean(P_mode_tot[i],axis=1)),linestyle='dashed',markersize=1,color=color_mode[i])
        ax2.plot(np.ma.masked_where(Stab_mode_tot[i], Rains_mode_tot[i]),np.ma.masked_where(Stab_mode_tot[i], np.mean(P_mode_tot[i],axis=1)),markersize=8,color=color_mode[i],label=name_mode[i])
        ax3.plot(np.ma.masked_where(True^Stab_mode_tot[i], Rains_mode_tot[i]),np.ma.masked_where(True^Stab_mode_tot[i], np.max(P_mode_tot[i],axis=1)),linestyle='dashed',markersize=1,color=color_mode[i])
        ax3.plot(np.ma.masked_where(Stab_mode_tot[i], Rains_mode_tot[i]),np.ma.masked_where(Stab_mode_tot[i], np.max(P_mode_tot[i],axis=1)),markersize=8,color=color_mode[i],label=name_mode[i])
    ax2.set_xlim(0.4,1.4)
    ax2.set_ylim(0,20)
    ax2.set_title('Mean Biomass $[g.m^{-2}]$')
    ax2.set_xlabel('Rain $[mm.d^{-1}$]')
    #ax5.legend()
    ax3.set_xlim(0.4,1.4)
    ax3.set_ylim(0,20)
    ax3.set_title('Maximum Biomass $[g.m^{-2}]$')
    ax3.set_xlabel('Rain $[mm.d^{-1}$]')
    ax3.legend(bbox_to_anchor=(1.,0.9))
    #ax6.legend()
    return(fig)

def stability_range(P,W,O,Rains,L,param):
    #évalue la stabilité des solutions le long d'un mode
    M=Rains.shape[0]
    N=P.shape[1]
    stab=np.array(np.zeros(M),dtype='bool')
    Lmb_max=np.array(np.zeros(M),dtype=complex)
    for i in range(M):
        J=Op(L,N,Rains[i],P[i],W[i],O[i],param)
        lmb,vec=np.linalg.eig(J)
        ii=np.argsort(-lmb)
        lmb=lmb[ii]
        vec=vec[:,ii]
        Lmb_max[i]=np.max(lmb)
        if np.max(lmb)>10**(-7):
            stab[i]=True
        else:
            stab[i]=False
    return(stab,Lmb_max)

def selec_rain(rain,Pmin,Pmax,Rains,P):
    rge=(P<Pmax) & (P>Pmin)
    Rains_rge=Rains[rge]
    b=np.argmax(-(Rains_rge-rain)**2)
    ii=np.where(Rains==Rains_rge[b])
    return(ii[0])
    
def stability_eigen(P,W,O,rain,L,param):
    print(max(P.shape))
    N=max(P.shape)  #P.shape[1]
    #J=Op(L,N,rain,P[0],W[0],O[0],param)
    J=Op(L,N,rain,np.squeeze(P),np.squeeze(W),np.squeeze(O),param)
    lmb,vec=np.linalg.eig(J)
    ii=np.argsort(-lmb)
    lmb=lmb[ii]
    vec=vec[:,ii]
    return(lmb,vec)

def show_equilibria(Rains_mode_tot,P_mode_tot,W_mode_tot,O_mode_tot,Stab_mode_tot,n_mode,ind,x,L,name_mode,color_mode,N_mode):
    x=x/(2*np.pi)*L
    fig,ax=plt.subplots(1,4,figsize=(20,5))
    for i in range(N_mode):
        ax[0].plot(np.ma.masked_where(True^Stab_mode_tot[i], Rains_mode_tot[i]),np.ma.masked_where(True^Stab_mode_tot[i], np.mean(P_mode_tot[i],axis=1)),linestyle='dashed',markersize=1,color=color_mode[i])
        ax[0].plot(np.ma.masked_where(Stab_mode_tot[i], Rains_mode_tot[i]),np.ma.masked_where(Stab_mode_tot[i], np.mean(P_mode_tot[i],axis=1)),markersize=8,color=color_mode[i],label=name_mode[i])  
    ax[0].set_xlim(0.4,1.4)
    ax[0].set_ylim(0,16)
    ax[0].plot(Rains_mode_tot[n_mode][ind],np.mean(P_mode_tot[n_mode][ind]),marker='^',color='black',linestyle='none',label='equilibrium',markersize=7)
    ax[0].set_title('Mean Biomass $[g.m^{-2}]$')
    ax[0].set_xlabel('Rain $[mm.d^{-1}]$')
    #ax[0].legend()
    ax[1].plot(x,np.squeeze(P_mode_tot[n_mode][ind]))
    ax[1].set_xlabel('x $[m]$')
    ax[1].set_ylabel('B $[g.m^{-2}]$')
    ax[1].set_ylim((0,20))
    ax[2].plot(x,np.squeeze(W_mode_tot[n_mode][ind]))
    ax[2].set_xlabel('x $[m]$')
    ax[2].set_ylabel('W $[mm]$')
    ax[3].plot(x,np.squeeze(O_mode_tot[n_mode][ind]))
    ax[3].set_xlabel('x $[m]$')
    ax[3].set_ylabel('O $[mm]$')
    return()
