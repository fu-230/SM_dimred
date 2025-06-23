import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import SM_parameters as param
import SM_dimred as dimred

"""
implement the CW method because perturbation theory does not work. See that m3(mu,T) largely depends on mu, and if we take mu=g3, m3^2>0 at around 135 GeV, where the classical potential minimum is at \Phi=0.
Based on hep-lat/9412091 (32--34)
"""
#d for taking derivative wrt m3**2
def mT(T,v):
    #[hep-ph/9404201 (30)]=[hep-lat/9412091 (35)]=[hep-lat/9510020 (A.3)]
    return (1/2.)*dimred.g3(T)*v
def mL(T,v):
    #[hep-ph/9404201 (30)]=[hep-lat/9412091 (35)]
    #In [hep-lat/9510020 Sec.A], they integrate out the heavy scale
    return np.sqrt(dimred.mD2(T)+mT(T,v)**2)
def m1(mu3,T,v,d):
    #[hep-ph/9404201 (30)]=[hep-lat/9412091 (35)]=[hep-lat/9510020 (A.3)]
    return np.sqrt((dimred.mH32(mu3,T)+d)+3.*dimred.lam3(T)*v**2)
def m2(mu3,T,v,d):
    #[hep-ph/9404201 (30)]=[hep-lat/9412091 (35)]=[hep-lat/9510020 (A.3)]
    return np.sqrt((dimred.mH32(mu3,T)+d)+dimred.lam3(T)*v**2)
def mG2(mu3,T,d):
    #[hep-lat/9510020 (B.10)] + contribution from mL (second term, not found in the literature)
    return (3./(8.*np.pi))*dimred.g3(T)**2*(\
        (1/2.)*dimred.g3(T)*np.sqrt(-(dimred.mH32(mu3,T)+d)/dimred.lam3(T))\
        +(1/2.)*np.sqrt(dimred.g3(T)**2*(dimred.mH32(mu3,T)+d)/dimred.lam3(T)/4.+dimred.mD2(T))\
        +2.*dimred.lam3(T)*np.sqrt(-2.*(dimred.mH32(mu3,T)+d))/dimred.g3(T)**2\
        )
def vapprox(mu3,T,d):
    #vev at 1loop corresponding to m2**2=mG**2
    return np.sqrt((mG2(mu3,T,d)-dimred.mH32(mu3,T))/dimred.lam3(T))
"""
def m2imp(mu3,T,v,d):
    #[hep-lat/9510020 (B.10)] + contribution from mL (second term, not found in the literature)
    #Vimp not implemented because we haven't computed the non-analytic term at 2loop; (B.9) + contribution from mL
    return np.sqrt(m2(mu3,T,v,d)**2-mG2(mu3,T,d))
"""
def Hbar(s1,s2,s3,mu3):
    #finite part of the sunset function
    #[hep-ph/9404201 (32)]=[hep-lat/9412091 (36)]
    return np.log(mu3/(s1+s2+s3))+1/2.
def V0(mu3,T,v,d):
    #[hep-ph/9404201 (30)]=[hep-lat/9412091 (32)]=[hep-lat/9510020 (A.4)]
    return (1/2.)*(dimred.mH32(mu3,T)+d)*v**2+(1/4.)*dimred.lam3(T)*v**4
def V1(mu3,T,v,d):
    #[hep-ph/9404201 (30)]=[hep-lat/9412091 (33)]=[hep-lat/9510020 (A.4) w/ mL=0]
    mTv=mT(T,v);mLv=mL(T,v);m1v=m1(mu3,T,v,d);m2v=m2(mu3,T,v,d) #*v: variables, *(*): functions
    return -1./(12.*np.pi)*(6.*mTv**3+3.*mLv**3+m1v**3+3.*m2v**3)
def V2(mu3,T,v,d):
    #[hep-ph/9404201 (33), if apparent typos are corrected]=[hep-lat/9412091 (34)]
    #coeff of ln mu3 is consistent with that in m3
    mTv=mT(T,v);mLv=mL(T,v);m1v=m1(mu3,T,v,d);m2v=m2(mu3,T,v,d) #*v: variables, *(*): functions
    term1=(-3./16.)*dimred.g3(T)**4*v**2*(
        2.*Hbar(m1v,mTv,mTv,mu3)-(1/2.)*Hbar(m1v,mTv,0,mu3)+Hbar(m1v,mLv,mLv,mu3)\
        +(m1v**2/mTv**2)*(Hbar(m1v,mTv,0,mu3)-Hbar(m1v,mTv,mTv,mu3))
        +(m1v**4/(4.*mTv**4))*(Hbar(m1v,0,0,mu3)+Hbar(m1v,mTv,mTv,mu3)-2.*Hbar(m1v,mTv,0,mu3))\
        -m1v/(2.*mTv)-m1v**2/(4.*mTv**2)
    )
    term2=-3.*dimred.lam3(T)**2*v**2*(
        Hbar(m1v,m1v,m1v,mu3)+Hbar(m1v,m2v,m2v,mu3)
    )
    term3=2.*dimred.g3(T)**2*mTv**2*(
        (63./16.)*Hbar(mTv,mTv,mTv,mu3)+(3./16.)*Hbar(mTv,0,0,mu3)-41./16.
    )
    term4=(-3./2.)*dimred.g3(T)**2*(
        (mTv**2-4.*mLv**2)*Hbar(mLv,mLv,mTv,mu3)-2.*mTv*mLv-mLv**2
    )
    term5=4.*dimred.g3(T)**2*mTv**2\
        +(3./8.)*dimred.g3(T)**2*(2.*mTv+mLv)*(m1v+3.*m2v)\
        +(15./4.)*dimred.lamA(T)*mLv**2\
        +(3./4.)*dimred.lam3(T)*(m1v**2+2.*m1v*m2v+5.*m2v**2)
    term6=(-3./8.)*dimred.g3(T)**2*(
        (mTv**2-2.*m1v**2-2.*m2v**2)*Hbar(m1v,m2v,mTv,mu3)+(mTv**2-4.*m2v**2)*Hbar(m2v,m2v,mTv,mu3)\
        +(m1v**2-m2v**2)**2*(Hbar(m1v,m2v,mTv,mu3)-Hbar(m1v,m2v,0,mu3))/mTv**2\
        +(m1v**2-m2v**2)*(m1v-m2v)/mTv+mTv*(m1v+3.*m2v)-m1v*m2v-m2v**2
    )
    return (1/(16.*np.pi**2))*(term1+term2+term3+term4+term5+term6)
def V(mu3,T,v,d):
    #hep-lat/9412091
    return V0(mu3,T,v,d)+V1(mu3,T,v,d)+V2(mu3,T,v,d)

"""
#plot
plt.figure(figsize=(10,6))
v=np.arange(0,10,0.1)
T=154; mu3=3.*np.exp(-0.348725)*T #can find ve at 80--154 GeV
plt.plot(v,[V(mu3,T,v[i],0) for i in range(len(v))],label=r'$V_0+V_1+V_2$')
#plt.plot(v,[V0(mu3,T,v[i],0) for i in range(len(v))],label=r'$V_0$')
#plt.plot(v,[V1(mu3,T,v[i],0) for i in range(len(v))],label=r'$V_1$')
#plt.plot(v,[V2(mu3,T,v[i],0) for i in range(len(v))],label=r'$V_2$')
plt.xlabel(r'$v$ [GeV]')
plt.ylabel(r'$V_0,V_1,V_2,$ and $V_0+V_1+V_2$')
plt.title('3d effective potential at $T=$'+str(T)+' GeV\n $v_{approx}='+str(vapprox(mu3,T,0)))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
"""


#find minimum around vapprox
def vev(mu3,T,d):
    w=1 #search reasion: vapprox-w<v<vapprox+w
    Nv=200 #resolution
    for i in range(Nv):
        v=vapprox(mu3,T,d)-w+2.*w*i/Nv
        vf=vapprox(mu3,T,d)-w+2.*w*(i+1)/Nv
        if V(mu3,T,v,d)<V(mu3,T,vf,d):
            return [v,V(mu3,T,v,d)]
            break
"""
#plot
plt.figure(figsize=(10,6))
T=np.arange(80,154,1)
plt.plot(T,[vev(3.*np.exp(-0.348725)*T[i],T[i],0)[0] for i in range(len(T))],label=r'$v_{2-loop}$')
plt.plot(T,[vapprox(3.*np.exp(-0.348725)*T[i],T[i],0) for i in range(len(T))],label=r'$v_{approx}$')
plt.xlabel(r'$T$ [GeV]')
plt.ylabel(r'$v$')
plt.title('vev, mu3 dependent')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
"""


#CW
def CW(mu3,T):
    dm32=1
    CWvev21=(vev(mu3,T,dm32)[1]-vev(mu3,T,-dm32)[1])/(2.*dm32)
    dm32=0.1
    CWvev22=(vev(mu3,T,dm32)[1]-vev(mu3,T,-dm32)[1])/(2.*dm32)
    dm32=0.01
    CWvev23=(vev(mu3,T,dm32)[1]-vev(mu3,T,-dm32)[1])/(2.*dm32)
    if (CWvev21-CWvev22)<0.01:
        if (CWvev22-CWvev23)<0.01:
            return CWvev23/T
        else:
            #too small dm32 may cause an error
            return CWvev22/T 
    else:
        if (CWvev22-CWvev23)<0.01:
            return CWvev23/T
        else:
            return CWvev21/T
def CWrun(T):
    #run from mu30 to mu3=T
    mu30=2.13*T #3.*np.exp(-0.348725)*T
    CWworun=CW(mu30,T)
    #In the second line, we normalize phi following [hep-ph/9508379 (141)]
    return CWworun+3.*dimred.g3(T)**2/(16.*np.pi**2*T)*np.log(T/mu30)\
        #*(1-(1/(16.*np.pi**2))*(-(9./4.)*dimred.g4(T)**2*dimred.Lb()+3.*param.yt2[param.Ttoi(dimred.r*T)]*dimred.Lf()))

data = [CWrun(T) for T in np.arange(90, 154)]
np.savez('data.npz', x_data=np.arange(90, 154), y_data=data)

#"""
#plot
plt.figure(figsize=(10,6))
T=np.arange(140,154,1)
plt.plot(T,[CWrun(T[i]) for i in range(len(T))],label=r'$v^2_{CW}/2T^2$')
plt.plot(T,[0.23**2*(162-T[i])/2. for i in range(len(T))], label='lattice extrapolation')
plt.xlabel(r'$T$ [GeV]')
plt.ylabel(r'$v^2_{CW}/2T^2$')
plt.title(r'Coleman--Weinberg method, $\mu_T=0.6T$, $\mu_{3,0}=2.13T\,\to\,\mu_3=T$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('CW.pdf')
plt.show()
#"""
