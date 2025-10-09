import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy.special as sc
from scipy.special import gamma, factorial

#-----------------------Predefining some special functions------------#
sqrt = np.sqrt
log10  = np.log10
pi = np.pi
exp = np.exp

#-----------------------Physical constants----------------#
c = 3e10                   #  the speed of light in cm/sec 

def integrand1(x, b1, A_inv,g):
  term1 = (np.exp(-(1+b1)*x))*(x**(-3+b1))
  term2 = A_inv*x - g**2
  term_integ1 = term1*(term2**2)
  return term_integ1

def integrand2(x, b2, A_inv,g):
  term1 = x**(-3+b2)
  term2 = A_inv*x - g**2
  term_integ2 = term1*(term2**2)
  return term_integ2

def func_FS(u_1g,u):
  Gamma = (1 + u**2)**0.5
  beta = u/Gamma
    
  Gamma_1 = (1 + u_1g**2)**0.5
  beta_1 = u_1g/Gamma_1
    
  Gamma_21 = Gamma_1*Gamma*(1-beta_1*beta)
    
  numero_FS = (1/(4*Gamma_21))*(u_1g/Gamma) - beta
  deno_FS = (1/(4*Gamma_21))*(Gamma_1/Gamma) - 1
  beta_FS = numero_FS/deno_FS
  Gamma_FS = 1/(1 - beta_FS**2)**0.5
  u_FS = beta_FS*Gamma_FS
    
  return u_FS

def func_RS(u_4g,u):
  Gamma = (1 + u**2)**0.5
  beta = u/Gamma
    
  Gamma_4 = (1 + u_4g**2)**0.5
  beta_4 = u_4g/Gamma_4
    
  Gamma_34 = Gamma_4*Gamma*(1-beta_4*beta)
    
  numero_RS = beta_4 - 4*Gamma_34*(u/Gamma_4)
  deno_RS = 1 - 4*Gamma_34*(Gamma/Gamma_4)
  beta_RS = numero_RS/deno_RS
  Gamma_RS = 1/(1 - beta_RS**2)**0.5
  u_RS = beta_RS*Gamma_RS
    
  return u_RS
    
#--------Band function in the forward shocked region-------------#
p=2.5
b1_FS = -1/2      #1/3
b2_FS = - p/2
eps_FS = 1       #0.5
x_bFS = (b1_FS-b2_FS)/(1+b1_FS)

#---------Band function in the reverse shocked region------------#
p = 2.5
b1_RS = -1/2
b2_RS = -p/2
eps_RS = 1.0
x_bRS = (b1_RS-b2_RS)/(1+b1_RS)

#---------------------Power law indices of the band function--------------#
b1 = -0.25
b2 = -1.25
x_b = (b1-b2)/(1+b1)

au_minus_1 = 1    # Proper speed contrast minus one

#----------------------Quantities pertaining to shell S1----------------#
ton_1 = 0.1     # time of ejection of shell S1

u_1 = 100       # Proper speed of leading shell S1
Gamma_1 = (1 + u_1**2)**0.5
beta_1 = u_1/Gamma_1

Delta_10 = beta_1*c*ton_1    #initial radial width of shell S1

#-----------------------Quantities pertaining to shell S4-----------------#
ton_4 = ton_1          # time of ejection of shell S4
u_4 = (au_minus_1+1)*u_1
Gamma_4 = (1 + u_4**2)**0.5
beta_4 = u_4/Gamma_4

Delta_40 = beta_4*c*ton_4    # initial radial width of shell S4 
t_ej = ton_1                # time delay between the ejection of the two shells
chi = Delta_10/Delta_40    # The ratio of initial radial width of shell S1 to S4


#---------------------In the rest frame of shell S1-----------------------#
beta_41 = (beta_4-beta_1)/(1 - beta_4*beta_1)
Gamma_41 = 1/(1 - beta_41**2)**0.5
u_41 = beta_41*Gamma_41

#---------Proper density contrast for collision of equal energy shells-------#
f_en = chi*(Gamma_1*(Gamma_1 - 1))/(Gamma_4*(Gamma_4-1))

#-----Solving for the shocked fluid proper speed in the rest frame of shell S1------#
f = f_en
numero = 2*(f**1.5)*Gamma_41- f*(1+f)
deno = 2*f*(u_41**2 + Gamma_41**2) - (1 + f**2)
u_21 = u_41*np.sqrt(numero/deno)
Gamma_21 = np.sqrt(1 + u_21**2)
beta21 = u_21/Gamma_21

#---------------------Lorentz transforming to the lab frame----------------#
u = Gamma_21*Gamma_1*(beta_1 + beta21)
Gamma = np.sqrt(1+u**2)
beta = u/Gamma

#--------------------------Proper velocity of the forward shock front-----------#
u_FS = func_FS(u_1,u)
Gamma_FS = np.sqrt(1+u_FS**2)
beta_FS = u_FS/Gamma_FS

#------------------Proper velocity of the reverse shock front------------------#
u_RS = func_FS(u_4,u)
Gamma_RS = np.sqrt(1+u_RS**2)
beta_RS = u_RS/Gamma_RS

#---------------Estimating the g factor associated with both shock fronts------#
g_RS = Gamma/Gamma_RS
g_FS = Gamma/Gamma_FS

quant_RS = (g_RS**2)/(1-g_RS**2)**3
quant_FS = (g_FS**2)/(1-g_FS**2)**3

ratio_g = quant_RS/quant_FS


Gamma_21 = Gamma_1*Gamma*(1 - beta*beta_1)
Gamma_34 = Gamma_4*Gamma*(1 - beta*beta_4)

beta_21 = u_21/Gamma_21

u_34 = np.sqrt(Gamma_34**2-1)  
beta_34 = u_34/Gamma_34

beta_FSCD = (beta_FS - beta)/(1-beta*beta_FS)
beta_RSCD = (beta - beta_RS)/(1-beta*beta_RS)

quant1 = beta_RSCD+beta_34
quant2 = beta_FSCD+beta_21
ratio_vel = quant1/quant2

#-----------------Defining very important ratios------------------#
ratio_1 = (Gamma_34+1)/(Gamma_21+1)
ratio_2 = Gamma_34/Gamma_21
ratio_3 = (Gamma_34-1)/(Gamma_21-1)

ratio_freq = (ratio_1**(-0.5))*(ratio_2**(0.5))*(ratio_3**2)
ratio_bol = (eps_RS/eps_FS)*(ratio_1**(-1))*(ratio_2)*ratio_vel
ratio_lum = (ratio_freq**(-1))*(ratio_bol)


ratio_freq_norm = 1/ratio_freq
ratio_bol_norm = 1/ratio_bol
ratio_lum_norm = 1/ratio_lum
ratio_g_norn = 1/ratio_g

#------------------------radial extent of the shocked region--------------#
t_FS = Delta_10/(c*(beta_FS - beta_1))
DeltaR_FS = beta_FS*c*t_FS

R_o = (beta_4*beta_1*c*t_ej)/(beta_4-beta_1)

t_RS = Delta_40/(c*(beta_4 - beta_RS))
DeltaR_RS = beta_RS*c*t_RS

radial_FS = DeltaR_FS/R_o
radial_RS = DeltaR_RS/R_o


num = 250
nu_barRS = 10**(-3)
Lo_RS = 1

T_barRS = np.linspace(0,5,num)

tilde_TRS = T_barRS + 1
F_nu_RS = np.zeros(len(T_barRS))

nu_bar = nu_barRS
g = g_RS
for i in range(len(F_nu_RS)):
  if(tilde_TRS[i]<1):
    y_min = 1
  else:
    y_min = 1/tilde_TRS[i]
  
  if(tilde_TRS[i]<(1+radial_RS)):
    y_max = 1
  else:
    y_max = (1+radial_RS)/tilde_TRS[i]
      
  if(tilde_TRS[i]<1):
    quant = 0
  else:
    A = nu_bar*tilde_TRS[i]       
    A_inv = 1/A
    llim = A*y_min*(1 + (g**2)*( (1/y_min) -1.0 ))
    ulim = A*y_max*(1 + (g**2)*( (1/y_max) -1.0 ))
    if(ulim<x_b):
      x_min = llim
      x_max = ulim
      I1 = quad(integrand1, x_min, x_max, args=(b1,A_inv,g))[0]
      quant1 = (np.exp(1+b1))*I1
      quant2 = 0
    elif(llim>x_b):
      x_min = llim
      x_max = ulim
      I2 = quad(integrand2, x_min, x_max, args=(b2,A_inv,g))[0]
      quant1 = 0
      quant2 = (x_b**(b1-b2))*(np.exp(1+b2))*I2
    else:
      x_max = ulim
      x_min = llim
      I1 = quad(integrand1, x_min, x_b, args=(b1,A_inv,g))[0]
      I2 = quad(integrand2, x_b, x_max, args=(b1,A_inv,g))[0]
      quant1 = (np.exp(1+b1))*I1
      quant2 = (x_b**(b1-b2))*(np.exp(1+b2))*I2
    quant = quant1+quant2
    F_nu_RS[i] = (quant)*(A**2)*(g**2/(1-g**2)**3)*tilde_TRS[i]
  

arr_ls=[":","--","-.","-"]
#nu_bar_arrRS = np.array([1e-2,1e-1,1])
nu_bar_arrRS = np.array([10**(-2),10**(-1),1])


F_nu_arrRS = np.zeros((len(nu_bar_arrRS),len(F_nu_RS)))
F_nuRS_integ = np.zeros(len(nu_bar_arrRS))

nu_bar_arrFS = np.zeros(len(nu_bar_arrRS))
nu_bar_arrFS = nu_bar_arrRS*ratio_freq
Lo_FS = Lo_RS/ratio_lum 

T_barFS = ((Gamma_FS/Gamma_RS)**2)*(beta_FS/beta_RS)*((1+beta_FS)/(1+beta_RS))*T_barRS
tilde_TFS = T_barFS+1

F_nu_FS = np.zeros(len(T_barRS))
F_nu_arrFS = np.zeros((len(nu_bar_arrFS),len(F_nu_FS)))

g = g_RS
b1 =  b1_RS
b2 = b2_RS
x_b = (b1-b2)/(1+b1)
for j in range(len(nu_bar_arrRS)):
  nu_bar = nu_bar_arrRS[j]
  #print(nu_bar)
  for i in range(len(F_nu_RS)):
    if(tilde_TRS[i]<1):
      y_min = 1
    else:
      y_min = 1/tilde_TRS[i]

    if(tilde_TRS[i]<(1+radial_RS)):
      y_max = 1
    else:
      y_max = (1+radial_RS)/tilde_TRS[i]
    
    if(tilde_TRS[i]<1):
      quant = 0
    else:
      A = nu_bar*tilde_TRS[i]
      A_inv = 1/A
      llim = A*y_min*(1 + (g**2)*( (1/y_min) -1.0 ))
      ulim = A*y_max*(1 + (g**2)*( (1/y_max) -1.0 ))
      
      if(ulim<x_b):
        x_min = llim
        x_max = ulim
        I1 = quad(integrand1, x_min, x_max, args=(b1,A_inv,g))[0]
        quant1 = (np.exp(1+b1))*I1
        quant2 = 0
      elif(llim>x_b):
        x_min = llim
        x_max = ulim
        I2 = quad(integrand2, x_min, x_max, args=(b2,A_inv,g))[0]
        quant1 = 0
        quant2 = (x_b**(b1-b2))*(np.exp(1+b2))*I2
      else:
        x_max = ulim
        x_min = llim
        I1 = quad(integrand1, x_min, x_b, args=(b1,A_inv,g))[0]
        I2 = quad(integrand2, x_b, x_max, args=(b1,A_inv,g))[0]
        quant1 = (np.exp(1+b1))*I1
        quant2 = (x_b**(b1-b2))*(np.exp(1+b2))*I2
      quant = quant1+quant2
      F_nu_RS[i] = 3*(quant)*(A**2)*(g**2/(1-g**2)**3)*tilde_TRS[i]
  
    # plt.plot(T_barRS,F_nu_RS/max(F_nu_RS),color="black",ls=arr_ls[j],lw=1.0)
    F_nu_arrRS[j] = F_nu_RS
    F_nuRS_integ[j] = np.trapz(nu_bar*F_nu_RS,T_barRS) 
    # print(F_nuRS_integ[j])

g = g_FS
b1 =  b1_FS
b2 = b2_FS
x_b = (b1-b2)/(1+b1)
for j in range(len(nu_bar_arrFS)):
  nu_bar = nu_bar_arrFS[j]
  #print(nu_bar)
  for i in range(len(F_nu_FS)):
    if(tilde_TFS[i]<1):
      y_min = 1
    else:
      y_min = 1/tilde_TFS[i]

    if(tilde_TFS[i]<(1+radial_FS)):
      y_max = 1
    else:
      y_max = (1+radial_FS)/tilde_TFS[i]
    
    if(tilde_TFS[i]<1):
      quant = 0
    else:
      A = nu_bar*tilde_TFS[i]       
      A_inv = 1/A
      llim = A*y_min*(1 + (g**2)*( (1/y_min) -1.0 ))
      ulim = A*y_max*(1 + (g**2)*( (1/y_max) -1.0 ))
      
      if(ulim<x_b):
        x_min = llim
        x_max = ulim
        I1 = quad(integrand1, x_min, x_max, args=(b1,A_inv,g))[0]
        quant1 = (np.exp(1+b1))*I1
        quant2 = 0
      elif(llim>x_b):
        x_min = llim
        x_max = ulim
        I2 = quad(integrand2, x_min, x_max, args=(b2,A_inv,g))[0]
        quant1 = 0
        quant2 = (x_b**(b1-b2))*(np.exp(1+b2))*I2
      else:
        x_max = ulim
        x_min = llim
        I1 = quad(integrand1, x_min, x_b, args=(b1,A_inv,g))[0]
        I2 = quad(integrand2, x_b, x_max, args=(b1,A_inv,g))[0]
        quant1 = (np.exp(1+b1))*I1
        quant2 = (x_b**(b1-b2))*(np.exp(1+b2))*I2
      quant = quant1+quant2
      F_nu_FS[i] = (3/ratio_lum)*(quant)*(A**2)*(g**2/(1-g**2)**3)*tilde_TFS[i]
        
    #plt.plot(T_barRS,F_nu_FS/max(F_nu_FS),color="black",ls=arr_ls[j],lw=1.0)
    #plt.xlim(0,5)
    F_nu_arrFS[j] = F_nu_FS

def plot_lightcurves():
  #plt.title(r"$(u_1,u_4)=(100,200) , \; (t_\mathrm{on1},t_\mathrm{off},t_\mathrm{on4})=(1,1,2)$",fontsize=14)
  plt.title(r"$ (t_\mathrm{on1},t_\mathrm{off},t_\mathrm{on4})=(1,1,1)$",fontsize=14) #112

  for j in range(len(nu_bar_arrRS)):
    char = np.round(np.log10(nu_bar_arrRS[j]),1)
    #char2 = np.round(np.log10(nu_bar_arrFS[j]),1)
    #print(char)
    plt.plot(T_barRS,nu_bar_arrRS[j]*F_nu_arrRS[j],color="red",ls=arr_ls[j],lw=1.0)
    plt.plot(T_barRS,nu_bar_arrRS[j]*F_nu_arrFS[j],color="blue",ls=arr_ls[j],lw=1.0)
    plt.plot(T_barRS,nu_bar_arrRS[j]*(F_nu_arrFS[j]+F_nu_arrRS[j]),color="black",ls=arr_ls[j],lw=1.0,label= char)

    #plt.loglog(T_barRS,F_nu_arrFS[j],color="green",ls=arr_ls[j],lw=1.0)

  plt.xlim(0,4)
  #plt.ylim(1e-4,1e1)
  #plt.text(5,1.0,r"$\log_{10}(\nu/\nu_\mathrm{o})$",fontsize=20)
  plt.legend(loc=[0.70,0.55],shadow=False,frameon=False,fontsize=16)
  #plt.text(3.5,0.9,r"$\log_{10}(\nu/\nu_\mathrm{o})$",fontsize = 16)
  plt.text(3.0,1.1,r"$\log_{10}(\nu/\nu_\mathrm{o})$",fontsize = 16)

  plt.text(-0.5,1.1,r"$(C)$",fontsize=18)
  plt.text(-1.2,0.5,r"$\frac{\nu F_{\nu}}{\nu_\mathrm{o} F_\mathrm{o}}$",fontsize=30)
  plt.xlabel(r"$\widebar{T}$",fontsize=25)
  #plt.axhline(y=0.2)
  plt.ylim(0,1.2)
  plt.tick_params(direction="in",labelsize="16") 
  #plt.savefig("linear_c.png",bbox_inches='tight',dpi=200)

  plt.show()


nuFnuRS1 =nu_bar_arrRS[0]* F_nu_arrRS[0]
nuFnuRS2 = nu_bar_arrRS[1]*F_nu_arrRS[1]
nuFnuRS3 =nu_bar_arrRS[2]*F_nu_arrRS[2]

nuFnuFS1 =nu_bar_arrRS[0]* F_nu_arrFS[0]
nuFnuFS2 = nu_bar_arrRS[1]*F_nu_arrFS[1]
nuFnuFS3 =nu_bar_arrRS[2]*F_nu_arrFS[2]

nuFnu1 = nu_bar_arrRS[0]*(F_nu_arrFS[0]+F_nu_arrRS[0])
nuFnu2 = nu_bar_arrRS[1]*(F_nu_arrFS[1]+F_nu_arrRS[1])
nuFnu3 = nu_bar_arrRS[2]*(F_nu_arrFS[2]+F_nu_arrRS[2])

#filename="pulses_hydroD.txt"
#data = np.array(list(zip(T_barRS,nuFnuRS1,nuFnuRS2,nuFnuRS3,nuFnuFS1,nuFnuFS2,nuFnuFS3,nuFnu1,nuFnu2,nuFnu3)))
#np.savetxt(filename, data)

nu_RS =nu_bar_arrRS[1]
nuF_RS = nu_bar_arrRS[1]*F_nu_arrRS[1]
nuF_FS = nu_bar_arrRS[1]*F_nu_arrFS[1]

#print(len(nuF_RS))
#filename="lightcurveC_comb3.txt"
#data = np.array(list(zip(T_barRS,nuF_RS,nuF_FS)))
#np.savetxt(filename, data)

def plot_lightcurves_loglog():
  plt.title(r"$(u_1,u_4)=(100,200) , \; (t_\mathrm{on1},t_\mathrm{off},t_\mathrm{on4})=(2,2,1)$",fontsize=14)

  for j in range(len(nu_bar_arrRS)):
    char = np.round(np.log10(nu_bar_arrRS[j]),1)
    #char2 = np.round(np.log10(nu_bar_arrFS[j]),1)

    plt.loglog(T_barRS,nu_bar_arrRS[j]*F_nu_arrRS[j],color="red",ls=arr_ls[j],lw=1.0)
    plt.loglog(T_barRS,nu_bar_arrRS[j]*F_nu_arrFS[j],color="blue",ls=arr_ls[j],lw=1.0)
    plt.loglog(T_barRS,nu_bar_arrRS[j]*(F_nu_arrFS[j]+F_nu_arrRS[j]),color="black",ls=arr_ls[j],lw=1.0,label= str(char))

  plt.xlim(1e-2,1e1)
  #plt.xlim(4.5,5.2)
  plt.ylim(1e-3,2)
  plt.legend(loc="best",shadow=False,frameon=False,fontsize=16)
  plt.text(1.5e-1,2e-2,r"$\log_{10}(\nu/\nu_\mathrm{o})$",fontsize = 16)
  plt.ylabel( r"$\frac{\nu F_{\nu}}{\nu_\mathrm{o} F_\mathrm{o}}$",fontsize=30)
  plt.xlabel(r"$\widebar{T}$",fontsize=25)
  #plt.axhline(y=0.2)
  plt.tick_params(direction="in",labelsize="16")
  #plt.savefig("loglog_221G.png",bbox_inches='tight',dpi=200)

  plt.show()

nu_barFS = nu_barRS*ratio_freq
Lo_FS = Lo_RS/ratio_lum 

T_barFS = ((Gamma_FS/Gamma_RS)**2)*(beta_FS/beta_RS)*((1+beta_FS)/(1+beta_RS))*T_barRS
tilde_TFS = T_barFS+1
F_nu_FS = np.zeros(len(T_barRS))

nu_bar = nu_barFS
g = g_FS
for i in range(len(F_nu_FS)):
  if(tilde_TFS[i]<1):
    y_min = 1
  else:
    y_min = 1/tilde_TFS[i]
  
  if(tilde_TFS[i]<(1+radial_FS)):
    y_max = 1
  else:
    y_max = (1+radial_FS)/tilde_TFS[i]
      
  if(tilde_TFS[i]<1):
    quant = 0
  else:
    A = nu_bar*tilde_TFS[i]       
    A_inv = 1/A
    llim = A*y_min*(1 + (g**2)*( (1/y_min) -1.0 ))
    ulim = A*y_max*(1 + (g**2)*( (1/y_max) -1.0 ))
    if(ulim<x_b):
      x_min = llim
      x_max = ulim
      I1 = quad(integrand1, x_min, x_max, args=(b1,A_inv,g))[0]
      quant1 = (np.exp(1+b1))*I1
      quant2 = 0
    elif(llim>x_b):
      x_min = llim
      x_max = ulim
      I2 = quad(integrand2, x_min, x_max, args=(b2,A_inv,g))[0]
      quant1 = 0
      quant2 = (x_b**(b1-b2))*(np.exp(1+b2))*I2
    else:
      x_max = ulim
      x_min = llim
      I1 = quad(integrand1, x_min, x_b, args=(b1,A_inv,g))[0]
      I2 = quad(integrand2, x_b, x_max, args=(b1,A_inv,g))[0]
      quant1 = (np.exp(1+b1))*I1
      quant2 = (x_b**(b1-b2))*(np.exp(1+b2))*I2
    quant = quant1+quant2
    F_nu_FS[i] =(3/ratio_lum)*(quant)*(A**2)*(g**2/(1-g**2)**3)*tilde_TFS[i]
  
  #print(i,T_bar[i],F_nu[i])
        
