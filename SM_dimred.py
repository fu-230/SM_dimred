import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import SM_parameters as param

#4d renormalization scale is set to muT.
r=1 #our choice
#r=7.055 #choice of muT/T in [hep-ph/9404201]
"""
4d parameters are evaluated at the renormalization scale muT.
"""
def nu2(T):
    return param.nu2[param.Ttoi(r*T)]
def g4(T):
    return np.sqrt(param.g2[param.Ttoi(r*T)])
def gY4(T):
    return param.gY
def lam(T):
    return param.lam[param.Ttoi(r*T)]
"""
3d parameters as a function of temperature.
[hep-ph/9508379 Sec.4.2]
"""
def Lb():
    #[hep-ph/9508379 (31)]
    return 2.*np.log(r)-2.*1.953808
def Lf():
    #[hep-ph/9508379 (31)]
    #4d renormalization scale is set to muT.
    return 2.*np.log(r)-2.*0.567514
def g3(T):
    #[hep-ph/9508379 (146)]
    #the result in [hep-ph/9404201 (4)] + fermion contribution (-4./3. nf Lf) + error (+2./3.)
    return g4(T)*np.sqrt(T*(1+(g4(T)**2/(16.*np.pi**2))*(43.*Lb()/6.-4.*param.nf*Lf()/3.+2./3.)))
    #alternatively, if one-loop approx is enough, ...
    #same as [hep-ph/9404201 (4)], [hep-lat/9412091 (16)]
    #return g4(T)*np.sqrt(T)
def lam3(T):
    h=param.mH/param.mW;t=param.mt/param.mW #[hep-ph/9508379 (135)]
    #[hep-ph/9508379 (150)]
    #the result in [hep-ph/9404201 (5)] + fermion contribution (Lf (...)). Note that h**2 g4**2 = 8 lam.
    return T*lam(T)*(1-(3.*g4(T)**2/(64.*np.pi**2))*(Lb()*(6./h**2-6+2.*h**2)+Lf()*(4.*t**2-8.*t**4/h**2)-4./h**2))
    #alternatively, if one-loop approx is enough, ...
    #return T*lam(T)
def mD2(T):
    #Debye mass squared
    #[hep-ph/9508379 (161)]
    #the result in [hep-ph/9404201 (8)] + fermion contribution
    #running at g**4 order has not been computed in [hep-ph/9404201], [hep-lat/9412091], [hep-ph/9508379]
    return (1/3.)*g4(T)**2*(5/2.+param.nf)*T**2
def mDY2(T):
    #Debye mass squared
    #[hep-ph/9508379 (160)]
    return (1/6.)*gY4(T)**2*(1+(10/3.)*param.nf)*T**2
def h3(T):
    h=param.mH/param.mW;t=param.mt/param.mW #[hep-ph/9508379 (135)]
    #[hep-ph/9508379 (147)]
    #the result in [hep-ph/9404201 (6)] + fermion contribution (-4./3. nf Lf +4./3. nf -3. t**2) + error (+2./3.)
    #If we correct the error (+2./3.) in the expression of g3 [hep-ph/9404201 (4)], the same as [hep-ph/9404201 (6, second line)] + fermion contribution (-4./3. nf Lf +4./3. nf -3. t**2)
    #If we correct the error (+2./3.) in the expression of g3 [hep-ph/9404201 (4)], the same as [hep-ph/9412091 (18)] + fermion contribution (-4./3. nf Lf +4./3. nf -3. t**2)
    return (g4(T)**2*T/4.)*(1+(g4(T)**2/(16.*np.pi**2))*(43*Lb()/6.-4*param.nf*Lf()/3.+53./6.-1./3.+4.*param.nf/3.+3.*h**2/2.-3.*t**2))
    #alternatively, if one-loop approx is enough, ...
    #return g4(T)**2*T/4.
def mH32(mu3,T):
    h=param.mH/param.mW;t=param.mt/param.mW;s=param.gS/param.g #[hep-ph/9508379 (135)]
    nutil2=nu2(T)*(1-(3.*g4(T)**2/(64.*np.pi**2))*((h**2-3)*Lb()+2.*t**2*Lf()))    #[hep-ph/9508379 (154)]
    yttil2=g4(T)**2*t**2/2.*T*(1-(3.*g4(T)**2/(128.*np.pi**2))*((6.*t**2-6-64.*s**2/3.)*Lf()+2+28.*np.log(2)-12.*h**2*np.log(2)+8.*t**2*np.log(2)-64.*s**2*(4.*np.log(2)-3)/9.))    #[hep-ph/9508379 (155)]
    #[hep-ph/9508379 (156)]
    #With Lb=0, the result in [hep-ph/9404201 (9)] + fermion contribution + U(1)Y contribution at 1loop
    #With Lb=0, if we replace h3 with g3**2/4, and if we correct the error (+2./3.) in the expression of g3 [hep-ph/9404201 (4)], the result in [hep-ph/9404201 (66)]=[hep-lat/9412091 (28)]=[hep-lat/9510020 (2.2)]=[hep-lat/9612006 (2.5)] + fermion contribution + U(1)Y contribution at 2loop
    #the last line: higher order correction in [hep-lat/9612006 (2.5)] included
    return -nutil2+\
    T*(lam3(T)/2.+3.*g3(T)**2/16.+gY4(T)**2*T/16.+yttil2/4.)+\
    (T**2/(16.*np.pi**2))*(g4(T)**4*(137./96.+3.*param.nf*np.log(2)/2.+param.nf/12.)+3.*lam(T)*g4(T)**2/4.)+\
    (1./16.*np.pi**2)*(39.*g3(T)**4/16.+12.*h3(T)*g3(T)**2-6.*h3(T)**2+9.*lam3(T)*g3(T)**2-12.*lam3(T)**2+\
                       -(9./8.)*g3(T)**2*gY4(T)**2*T-(5./16.)*gY4(T)**4*T**2+3.*lam3(T)*gY4(T)**2*T)*(np.log(3.*T/mu3)-0.348725)
    #alternatively, if one-loop approx is enough, ...
    #return -nu2(T)+T**2*(lam(T)+3*g4(T)**2/8.+gY4(T)**2/8.+param.yt**2/2.)/2.
def lamA(T):
    #[hep-ph/9508379 (162)]
    #the result in [hep-ph/9404201 (16)]=[hep-lat/9412091 (19)] + fermion contribution
    #can be ignored at one-loop matching
    return T*((17-4.*param.nf)/3.)*g4(T)**4/(16.*np.pi**2)
"""
if you want to integrate over the heavy scale gT...
[hep-ph/9508379 Sec.4.3]
"""
def g3eff(T):
    #[hep-ph/9508379 (167)]=[hep-ph/9404201 (51)]
    return g3(T)*np.sqrt(1-g3(T)**2/(24.*np.pi*np.sqrt(mD2(T))))
def lam3eff(T):
    #[hep-ph/9508379 (169)]
    return lam3(T)-(1/(8.*np.pi))*(3.*h3(T)**2/np.sqrt(mD2(T))+gY4(T)**4*T**2/(16.*np.sqrt(mDY2(T)))+gY4(T)**4*T**2/(4.*(np.sqrt(mD2(T))+np.sqrt(mDY2(T)))))
    #alternatively, if one-loop approx is enough, ...
    #If we replace h3 with g3**2/4, same as [hep-ph/9404201 (52)]
    #return lam3(T)-(1/(8.*np.pi))*(3.*h3(T)**2/np.sqrt(mD2(T)))
def mH32eff(mu3,T):
    #[hep-ph/9508379 (174)]
    #If we replace h3 with g3**2/4, same as [hep-ph/9404201 (53)]
    return mH32(mu3,T)-\
    (1/(4.*np.pi))*(3*h3(T)*np.sqrt(mD2(T))+(1/4.)*gY4(T)**2*T*np.sqrt(mDY2(T)))+\
    (1/(16.*np.pi**2))*((-3.*g3(T)**4/4.+12.*h3(T)*g3(T)**2-6.*h3(T)**2)*np.log(mu3/(2.*np.sqrt(mD2(T))))+3.*h3(T)*g3(T)**2-3.*h3(T)**2)
    #alternatively, if one-loop approx is enough, ...
    #return mH32(mu3,T)-(1/(4.*np.pi))*(3*h3(T)*np.sqrt(mD2(T))+(1/4.)*gY4(T)**2*T*np.sqrt(mDY2(T)))


#plot
"""
plt.figure(figsize=(10, 6))
T=np.arange(140,170,0.1)
plt.plot(T,[g3eff(T[i])**2/T[i] for i in range(len(T))],label=r'$g_{3eff}$')
plt.plot(T,[g3(T[i])**2/T[i] for i in range(len(T))],label=r'$g_3$')
#plt.hlines(param.g**2,140,170)
plt.xscale('log')
plt.xlabel(r'$T$ [GeV]')
plt.ylabel(r'$g_3^2/T$ and $g_{3eff}^2/T$')
plt.title('to be compared with the top panel in 1508.07161 Fig.1')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#
plt.figure(figsize=(10, 6))
T=np.arange(140,170,0.2)
plt.plot(T,[lam3eff(T[i])/g3eff(T[i])**2 for i in range(len(T))],label=r'$\lambda_{3eff}/g_{3eff}^2$')
plt.plot(T,[lam3(T[i])/g3(T[i])**2 for i in range(len(T))],label=r'$\lambda_3/g_3^2$')
plt.xscale('log')
plt.xlabel(r'$T$ [GeV]')
plt.ylabel(r'$x$')
plt.title('to be compared with the 3rd top panel in 1508.07161 Fig.1')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
"""
#
plt.figure(figsize=(10, 6))
T=np.arange(140,170,0.2)
#plt.plot(T,[mH32eff(g3eff(T[i]),T[i])/g3eff(T[i])**4 for i in range(len(T))],label=r'$m_{3eff}^2/g_{3eff}^4, \mu_3=g_{3eff}$')
#plt.plot(T,[mH32(g3(T[i]),T[i])/g3(T[i])**4 for i in range(len(T))],label=r'$m_3^2/g_3^4, \mu_3=g_3$')
plt.plot(T,[mH32eff(3.*np.exp(-0.348725)*T[i],T[i])/g3eff(T[i])**4 for i in range(len(T))],label=r'$m_{3eff}^2/g_{3eff}^4, \mu_3=2.11T$')
plt.plot(T,[mH32(3.*np.exp(-0.348725)*T[i],T[i])/g3(T[i])**4 for i in range(len(T))],label=r'$m_3^2/g_3^4, \mu_3~2.11T$ chosen so that log contribution vanishes')
plt.plot(T,[(-nu2(T[i])+T[i]**2*(lam(T[i])+3*g4(T[i])**2/8.+gY4(T[i])**2/8.+param.yt**2/2.)/2.)/g3(T[i])**4 for i in range(len(T))],label=r'$-\mu^2$')
plt.xscale('log')
plt.xlabel(r'$T$ [GeV]')
plt.ylabel(r'$y$')
plt.title('to be compared with the 2nd top panel in 1508.07161 Fig.1.\n Note that the zero-crossing temperature largely depends on the choice of 4d and 3d renormalization scales.\n Our choice here is mu4=T, mu3=2.11T.')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#
"""
plt.figure(figsize=(10, 6))
T=np.arange(140,170,0.2)
plt.plot(T,[gY4(T[i])**2*T[i]/g3eff(T[i])**2 for i in range(len(T))],label=r'$g_{Y3}^2/g_{3eff}^2$')
plt.plot(T,[gY4(T[i])**2*T[i]/g3(T[i])**2 for i in range(len(T))],label=r'$g_{Y3}^2/g_3^2$')
plt.xscale('log')
plt.xlabel(r'$T$ [GeV]')
plt.ylabel(r'$z$')
plt.title('to be compared with the bottom panel in 1508.07161 Fig.1')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
"""
