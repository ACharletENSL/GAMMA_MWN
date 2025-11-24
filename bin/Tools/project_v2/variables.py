# contains relevant content for simulation to variable translation
# shall be replaced by a registry later on

from phys_functions import *

# Functions list
cool_vars = ["V4", "dt", "inj", "fc", "gm", "gb", "gM",
  "num", "nub", "nuM", "Lp", "L", "V4r", "tc"]
var2func = {
  "T":(derive_temperature, ['rho', 'p'], []),
  "h":(derive_enthalpy, ['rho', 'p'], []),
  "ad":(derive_adiab, ['rho', 'p'], []),
  "lfac":(derive_Lorentz, ['vx'], []),
  "u":(derive_proper, ['vx'], []),
  "ShSt":(derive_shockStrength, ['rho', 'p'], ['rhoscale']),
  "Ei":(derive_Eint_lab, ['x', 'dx', 'rho', 'vx', 'p'], ['R0', 'rhoscale', 'geometry']),
  "Ek":(derive_Ekin_lab, ['x', 'dx', 'rho', 'vx', 'p'], ['R0', 'rhoscale', 'geometry']),
  "ek":(derive_Ekin, ['rho', 'vx'], ['rhoscale']),
  "ei_lab":(derive_Eint, ['rho', 'vx', 'p'], ['rhoscale']),
  "ei":(derive_Eint_comoving, ['rho', 'p'], ['rhoscale']),
  "B":(derive_B_comoving, ['rho', 'p'], ['rhoscale', 'eps_B']),
  "B2":(derive_B2_comoving, ['rho', 'p'], ['rhoscale', 'eps_B']),
  "syn":(derive_syn_cooling, ['rho', 'p'], ['rhoscale', 'eps_B']),
  "res":(derive_resolution, ['x', 'dx'], []),
  "Pmax":(derive_Pmax, ['rho', 'p'], ['rhoscale', 'eps_B', 'xi_e']),
  "Pemax":(derive_Pemax, ['rho', 'p'], ['rhoscale', 'eps_B', 'xi_e']),
  "Pnum":(derive_Pnum,['x', 'dx', 'rho', 'vx', 'p'], ['rhoscale', 'R0', 'eps_B', 'xi_e', 'psyn', 'geometry']),
  "Lp_num":(derive_localLum, ['rho', 'vx', 'p', 'x', 'dx', 'dt', 'gmin', 'gmax', 'gb'],
    ['rhoscale', 'eps_B', 'psyn', 'xi_e', 'R0', 'geometry']),
  "L":(derive_obsLum, ['rho', 'vx', 'p', 'x', 'dx', 'dt', 'gmin', 'gmax', 'gb'],
    ['rhoscale', 'eps_B', 'psyn', 'xi_e', 'R0', 'geometry']),
  "Lum":(derive_Lum, ['x', 'dx', 'rho', 'vx', 'p', 'gmin'],
    ['R0', 'rhoscale', 'psyn', 'eps_B', 'eps_e', 'geometry']),
  "Lth":(derive_Lum_thinshell, ['x', 'dx', 'rho', 'vx', 'p'],
    ['R0', 'rhoscale', 'psyn', 'eps_B', 'eps_e', 'xi_e', 'geometry']),
  "Lth2":(derive_L_thsh_corr, ['t', 'x', 'dx', 'rho', 'vx', 'p'],
    ['R0', 'rhoscale', 'psyn', 'eps_B', 'eps_e', 'xi_e', 'geometry']),
  "Lbol":(derive_Lbol_comov, ['x', 'rho', 'p'], ['R0', 'rhoscale', 'eps_e','geometry']),
  "Lp": (derive_Lp_nupm, ['x', 'rho', 'p'], ['R0', 'rhoscale', 'psyn', 'eps_B', 'eps_e', 'xi_e', 'geometry']),
  "Ep_th":(derive_Epnu_thsh, ['x', 'dx', 'rho', 'vx', 'p'],
    ['R0', 'rhoscale', 'psyn', 'eps_B', 'eps_e', 'xi_e', 'geometry']),
  "EpnuvFC":(derive_Epnu_vFC, ['x', 'dx', 'rho', 'vx', 'p', 'gmin'],
    ['R0', 'rhoscale', 'psyn', 'eps_B', 'eps_e', 'xi_e', 't0', 'z', 'geometry']),
  "EpnuSC":(derive_Epnu_SC, ['x', 'dx', 'dt', 'rho', 'p',],
    ['R0', 'rhoscale', 'psyn', 'eps_B', 'xi_e', 'geometry']),
  "Fpk_th":(derive_Fpeak_analytic, ['x', 'dx', 'rho', 'vx', 'p', 'gmin'],
    ['R0', 'rhoscale', 'psyn', 'eps_B', 'eps_e', 'zdl', 'geometry']),
  "Fpk_n":(derive_Fpeak_numeric, ['x', 'dx', 'rho', 'vx', 'p', 'gmin'],
    ['R0', 'rhoscale', 'psyn', 'eps_B', 'eps_e', 'zdl', 'geometry']),
  "n":(derive_n, ['rho'], ['rhoscale']),
  "Ne":(derive_Ne, ['x', 'dx', 'vx', 'rho'], ['R0', 'rhoscale', 'geometry']),
  "Pfac":(derive_emiss_prefac, [], ['psyn']),
  "Dop":(derive_DopplerRed_los, ['vx'], ['z']),
  "obsT":(derive_obsTimes, ['t', 'x', 'vx'], ['t0', 'z']),
  "Tth":(derive_obsAngTime, ['t', 'x', 'vx'], ['t0', 'z']),
  "Tej":(derive_obsEjTime, ['t', 'x', 'vx'], ['t0', 'z']),
  "Ton":(derive_obsOnTime, ['t', 'x', 'vx'], ['t0', 'z']),
  "V3":(derive_3volume, ['x', 'dx'], ['R0', 'geometry']),
  "V3p":(derive_3vol_comoving, ['x', 'dx', 'vx'], ['R0', 'geometry']),
  "V4":(derive_4volume, ['x', 'dx', 'dt'], ['R0', 'geometry']),
  "V4r":(derive_rad4volume, ['x', 'dx', 'rho', 'vx', 'p', 'dt', 'gb'],
    ['rhoscale', 'eps_B', 'R0', 'geometry']),
  "tc":(derive_tcool, ['rho', 'vx', 'p', 'gmin'], ['rhoscale', 'eps_B']),
  "tc1":(derive_tc1, ['rho', 'vx', 'p'], ['rhoscale', 'eps_B']),
  "gma_m":(derive_gma_m, ['rho', 'p'], ['rhoscale', 'psyn', 'eps_e', 'xi_e']),
  # "gma_c":(derive_gma_c, ['t', 'rho', 'vx', 'p'], ['t0', 'rhoscale', 'eps_B']),
  "gma_M":(derive_gma_M, ['rho', 'p'], ['rhoscale', 'eps_B']),
  "nup_B":(derive_cyclotron_comoving, ['rho', 'p'], ['rhoscale', 'eps_B']),
  "nu_B":(derive_cyclotron, ['rho', 'vx', 'p'], ['rhoscale', 'eps_B', 'z']),
  "nup_m":(derive_nup_from_gma, ['rho', 'p', 'gmin'], ['rhoscale', 'eps_B']),
  "nu_m":(derive_nu_from_gma, ['rho', 'vx', 'p', 'gmin'], ['rhoscale', 'eps_B', 'z']),
  "nup_M":(derive_nup_from_gma, ['rho', 'p', 'gmax'], ['rhoscale', 'eps_B']),
  "nu_M":(derive_nu_from_gma, ['rho', 'vx', 'p', 'gmax'], ['rhoscale', 'eps_B', 'z']),
  "nup_c":(derive_nup_c, ['rho', 'p'], ['R0', 'lfac', 'rhoscale', 'eps_B']),
  "nu_c":(derive_nu_c, ['rho', 'vx', 'p'], ['R0', 'lfac', 'rhoscale', 'eps_B', 'z']),
  "nup_m2":(derive_nup_m, ['rho', 'p'], ['rhoscale', 'psyn', 'eps_B', 'eps_e', 'xi_e']),
  "nu_m2":(derive_nu_m, ['rho', 'vx', 'p'], ['rhoscale', 'psyn', 'eps_B', 'eps_e', 'xi_e', 'z']),
  # "nup_c":(derive_nup_c, ['t', 'rho', 'vx', 'p'], ['rhoscale', 'eps_B']),
  "gmM":(derive_edistrib, ['rho', 'p'], ['rhoscale', 'psyn', 'eps_B', 'eps_e', 'xi_e']),
}

# Legends and labels
var_exp = {
  "time":"$t$", "x":"$r$", "r":"$r$", "dx":"$dr$", "rho":"$\\rho$", "vx":"$\\beta$", "p":"$p$", "v":"$\\beta$",
  "D":"$\\gamma\\rho$", "sx":"$\\gamma^2\\rho h$", "tau":"$\\tau$", "gma_m":"$\\gamma_m$",
  "trac":"tracer", "trac2":"wasSh", "Sd":"shock id", "gmin":"$\\gamma_{min}$", "gmax":"$\\gamma_{max}$", "zone":"",
  "T":"$\\Theta$", "h":"$h$", "lfac":"$\\Gamma$", "u":"$\\Gamma\\beta$", "u_i":"$\\Gamma\\beta$", "u_sh":"$\\gamma\\beta$",
  "Ei":"$e_{int}$", "Ekin":"$e_k$", "Emass":"$\\rho c^2$", "dt":"dt", "res":"dr/r", "Tth":"$T_\\theta$",
  "numax":"$\\nu'_{max}$", "numin":"$\\nu'_{min}$", "B":"$B'$", "Pmax":"$P'_{max}$",
  "gm":"$\\gamma_0$", "gb":"$\\gamma_1$", "gM":"$\\gamma_2$", "Lp_num":"$L'$", "L":"$L$",
  "Lth":"$\\tilde{L}_{\\nu_m}$", "Lum":"$\\tilde{L}_{\\nu_m}$", "Lp":"$L'$", "Lbol":"$L'_{bol}$",
  "num":"$\\nu'_m$", "nub":"$\\nu'_b$", "nuM":"$\\nu'_M$", "inj":"inj", "fc":"fc", "nu_m":"$\\nu_m$","nu_m2":"$\\nu_m$",
  "Ton":"$T_{on}$", "Tth":"$T_{\\theta,k}$", "Tej":"$T_{ej,k}$", "tc":"$t_c$",
  "V4":"$\\Delta V^{(4)}$", "V3":"$\\Delta V^{(3)}$", "V4r":"$\\Delta V^{(4)}_c$",
  "i":"i", 'i_sh':'i$_{sh}$', 'i_d':'i$_d$', 'ish':'i$_{sh}$', "syn":"syn"
  }
units_CGS  = {
  "time":" (s)", "x":" (cm)", "dx":" (cm)", "rho":" (g cm$^{-3}$)", "vx":"", "p":" (Ba)",
  "D":"", "sx":"", "tau":"", "trac":"", "trac2":"", "Sd":"", "gmin":"", "gmax":"", "u_sh":"",
  "T":"", "h":"", "lfac":"", "u":"", "u_i":"", "zone":"", "dt":" (s)", "res":"", "Tth":" (s)", "Lbol":" (erg)",
  "Ei":" (erg cm$^{-3}$)", "Ek":" (erg cm$^{-3}$)", "M":" (g)", "Pmax":" (erg s$^{-1}$cm$^{-3}$)", "Lp":" (erg s$^{-1}$)",
  "numax":" (Hz)", "numin":" (Hz)", "B":" (G)", "gm":"", "gb":"", "gM":"", "L":" (erg s$^{-1}$) ", "Lum":" (erg s$^{-1}$) ",
  "num":" (Hz)", "nub":" (Hz)", "nuM":" (Hz)", "inj":"", "fc":"", "tc":" (s).", "Ton":" (s)", "syn":"",
  "Tth":" (s)", "Tej": " (s)", "V4":" (cm$^3$s)", "V3":" (cm$^3$)", "V4r":" (cm$^3$s)", "i":"", "nu_m":" (Hz)", "nu_m2":" (Hz)",
  }
var_label = {'R':'$r$ (cm)', 'v':'$\\beta$', 'u':'$\\gamma\\beta$', 'u_i':'$\\gamma\\beta$', 'u_sh':'$\\gamma\\beta$',
  'f':"$n'_3/n'_2$", 'rho':"$n'$", 'rho3':"$n'_3$", 'rho2':"$n'_2$", 'Nsh':'$N_{cells}$', "Lbol":"$L'_{bol}$",
  'V':'$V$ (cm$^3$)', 'Nc':'$N_{cells}$', 'ShSt':'$\\Gamma_{ud}-1$', "Econs":"E",
  'ShSt ratio':'$(\\Gamma_{34}-1)/(\\Gamma_{21}-1)$', "Wtot":"$W_4 - W_1$",
  'M':'$M$ (g)', 'Msh':'$M$ (g)', 'Ek':'$E_k$ (erg)', 'Ei':'$E_{int}$ (erg)',
  'E':'$E$ (erg)', 'Etot':'$E$ (erg)', 'Esh':'$E$ (erg)', 'W':'$W_{pdV}$ (erg)',
  'Rct':'$(r - ct)/R_0$ (cm)', 'D':'$\\Delta$ (cm)', 'u_i':'$\\gamma\\beta$',
  'vcd':"$\\beta - \\beta_{cd}$", 'epsth':'$\\epsilon_{th}$', 'pdV':'$pdV$', "syn":"syn"
  }


nonlog_var = ['u', 'trac', 'Sd', 'zone', 'trac2']

def get_normunits(xnormstr, rhonormsub):
  rhonormsub  = '{' + rhonormsub +'}'
  CGS2norm = {" (s)":"$/T_0$",  " (s).":"$/t_{c,RS}$", " (g)":" (g)", " (Hz)":f"$/\\nu'_0$", " (eV)":" (eV)", " (G)":"$/B'_3$",
  " (cm)":" (ls)" if xnormstr == 'c' else "$/"+xnormstr+"$", " (g cm$^{-3}$)":f"$/\\rho_{rhonormsub}$",
  " (Ba)":f"$/\\rho_{rhonormsub}c^2$", " (erg cm$^{-3}$)":f"$/\\rho_{rhonormsub}c^2$", " (erg)":"$/L'_{bol}$",
  " (erg s$^{-1}$)":"$/L'_0$"," (erg s$^{-1}$) ":"$/L_0$", " (cm$^3$s)":"$/V_0t_0$", " (cm$^3$)":"$/V_0$",
  " (cm$^3$)":" (cm$^3$)", " (erg s$^{-1}$cm$^{-3}$)":"$/P'_0$"}
  return {key:(CGS2norm[value] if value else "") for (key,value) in units_CGS.items()}

def get_varscaling(var, env):
  rhoNorm = env.rhoscale
  pNorm = rhoNorm*c_**2
  unit  = units_CGS[var]
  var_scale = 1.
  if unit == " (g cm$^{-3}$)":
    var_scale = rhoNorm
  elif unit == " (Ba)" or unit == " (erg cm$^{-3}$)":
    var_scale = pNorm
  elif unit == " (erg)":
    var_scale = env.Ek1
  else:
    var_scale = 1.
  return var_scale
