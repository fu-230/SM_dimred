import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.integrate import solve_ivp

"""
4d parameters as the input. Note that their running are not taken into account. Because of that mW \neq g4*v0/2, and so on.
"""
#fermion generation number
nf=3
#W boson mass in GeV, [S. Navaset al.(Particle Data Group), Phys. Rev. D110, 030001 (2024), Sec. 10]
mW=80.4
#Z boson mass in GeV, [S. Navaset al.(Particle Data Group), Phys. Rev. D110, 030001 (2024), Sec. 10]
mZ=91.2
#Higgs mass in GeV, [S. Navaset al.(Particle Data Group), Phys. Rev. D110, 030001 (2024), Sec. 10]
mH=125.1
#on-shell weak mixing angle, [S. Navaset al.(Particle Data Group), Phys. Rev. D110, 030001 (2024), Sec. 10]
swOS=np.sqrt(0.223)
#MSbar weak mixing angle at the Z-pole, [S. Navaset al.(Particle Data Group), Phys. Rev. D110, 030001 (2024), Sec. 10]
swMSb=np.sqrt(0.231)
#choose our weak mixing angle
sw=swMSb
cw=np.sqrt(1-sw**2)
tw=sw/cw
#computed from the fine structure const, [S. Navaset al.(Particle Data Group), Phys. Rev. D110, 030001 (2024), Sec. 10]
gem=np.sqrt(4*np.pi/137.04) 
#SU2 coupling in 4D
g=gem/sw
#U1 coupling in 4D
#sometimes gY means the Yukawa coupling but here is not
gY=gem/cw
#SU3 coupling in 4D at the Z-pole, [S. Navaset al.(Particle Data Group), Phys. Rev. D110, 030001 (2024), Sec.9.4.1]
gS=np.sqrt(4.*np.pi*0.1173)
#Higgs vev in GeV, [S. Navaset al.(Particle Data Group), Phys. Rev. D110, 030001 (2024), Sec. 10]
v0=246.2
#Higgs self coupling
lam=mH**2/(2.*v0**2)
#top quark mass in GeV, [S. Navaset al.(Particle Data Group), Phys. Rev. D110, 030001 (2024), Sec. 10.2.3]
mt=172.57
#top yukawa
yt=np.sqrt(2)*mt/v0
"""
runnning of the 4d parameters. Beta functions adopted from [hep/ph/9508379 (131)--(134)].
gS and gY running: not solved
"""
#values at the Z-pole
nu2Z=mH**2/2.
g2Z=g**2
yt2Z=yt**2
lamZ=lam
#solve the RGE
def rge(t,y):
    #t=ln(renormalization scale/mZ)
    nu2,g2,yt2,lam=y
    dnu2=(1/(8.*np.pi**2))*(-9.*g2/4.+6.*lam+3.*yt2)*nu2
    dg2=(1/(8.*np.pi**2))*((8.*nf-43)/6.)*g2**2
    dyt2=(1/(8.*np.pi**2))*(9.*yt2**2/2.-9.*g2*yt2/4.-8.*gS**2*yt2)
    dlam=(1/(8.*np.pi**2))*(9.*g2**2/16.-9.*lam*g2/2.+12.*lam**2-3.*yt2**2+6.*lam*yt2)
    return [dnu2,dg2,dyt2,dlam]
#technical part
t0=0
tmax=np.log(200./mZ)
t_span=(t0,tmax)
Nt=500
t_eval=np.linspace(*t_span,Nt)
def itoT(i):
    return mZ*np.exp(t0+(tmax-t0)*i/(Nt-1))
def Ttoi(T):
    i=int((Nt-1)*(np.log(T/mZ)-t0)/(tmax-t0))
    if i<0:
        return 0
    elif i>Nt-1:
        return Nt-1
    else:
        return i
yZ=[nu2Z,g2Z,yt2Z,lamZ]
sol=solve_ivp(rge,t_span,yZ,t_eval=t_eval)
mu_vals=mZ*np.exp(t_eval)
nu2,g2,yt2,lam=sol.y

"""
#plot
plt.figure(figsize=(10, 6))
plt.plot(mu_vals,nu2,label=r'$\nu^2$')
#plt.xscale('log')
plt.xlabel(r'$\mu$ [GeV]')
plt.ylabel(r'$\nu^2$')
plt.title('Running at 1-loop SM')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#
plt.figure(figsize=(10, 6))
plt.plot(mu_vals,g2,label=r'$g^2$')
#plt.xscale('log')
plt.xlabel(r'$\mu$ [GeV]')
plt.ylabel(r'$g^2$')
plt.title('Running at 1-loop SM')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#
plt.figure(figsize=(10, 6))
plt.plot(mu_vals,yt2,label=r'$y_t^2$')
#plt.xscale('log')
plt.xlabel(r'$\mu$ [GeV]')
plt.ylabel(r'$y_t^2$')
plt.title('Running at 1-loop SM')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#
plt.figure(figsize=(10, 6))
plt.plot(mu_vals,lam,label=r'$\lambda$')
#plt.xscale('log')
plt.xlabel(r'$\mu$ [GeV]')
plt.ylabel(r'$\lambda$')
plt.title('Running at 1-loop SM')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
"""
