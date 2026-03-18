'''
Analyze emission in the thinshell approximation
Contains:
  - get_radiation_vFC
  - get_peaks_from_data
  - analyze_nubk
'''

from scipy.optimize import least_squares

from IO import *
from obs_functions import *
from radiation_thinshell import *
from phys_constants import *
from fits_hydro import *
from analysis_hydro import extract_data_thinshell
from peak_modeling import peaks_function_fromFit

### Extract peaks and save
def extract_fittingData(key):
  '''
  Extract data behind shock fronts from a sim, save in files
  '''

  # small utilitary to unpack results
  def foo(item):
    try:
      return item[0]
    except:
      return item

  
  env = MyEnv(key)
  zs = [1, 4]
  extract_data_thinshell(key, cells=zs, noOut=True)
  for z, front in zip(zs, ['FS', 'RS']):
    data = open_rundata(key, z)
    # drop rows with x == 0
    data = data.loc[~(data['x'] == 0.)]
    data.attrs['key'] = key
    fname = get_fitsfile(key, z)
    N = getattr(env, f'Nsh{z}')
    analyzed = get_anglefits(data, NT=2*N, Nnu=300, returnAll=True)
    analyzed = np.hstack(analyzed)
    logau = np.log10(env.a_u - 1.)
    analyzed = np.insert(analyzed, 0, logau)
    np.savetxt(fname, analyzed)

# get the fits for xi (effective angle approx)
def get_anglefits(data, Tmax=5, NT=450, Nnu=200, returnAll=False,
  i_set=1000, dfindex=True):

  nuobs, Tobs, env = obs_arrays_peakcentred(data.attrs['key'], NT=NT, Nnu=Nnu)
  t_max = data.iloc[-1].t
  front = 'RS' if data.iloc[0].trac < 1.5 else 'FS'
  g = getattr(env, 'g'+front)
  g2 = g*g

  # fits from hydro
  print('Fitting procedure')
  d_fit = cellsBehindShock_fromData(data)
  popt_lfac2, popt_ShSt, _ = get_hydrofits_shell(data, i_set, dfindex)
  popt_nu, popt_L = get_radfits_shell(d_fit)

  # break time
  Tmax0 = Tmax
  Tobsf = get_variable(d_fit.iloc[-1], 'Ton', env)
  i_f = np.searchsorted(Tobs, Tobsf)
  # in case chosen Tmax is too low
  while i_f == len(Tobs):
    Tmax += Tmax0
    nuobs, Tobs, env = obs_arrays_peakcentred(data.attrs['key'], Tmax=Tmax, NT=NT, Nnu=Nnu)
    i_f = np.searchsorted(Tobs, Tobsf)
  T = 1 + (Tobs - env.Ts)/env.T0
  Tf = T[i_f]

  # get peaks
  tnu = nuobs/env.nu0
  print('Calculating peaks of ' + front)
  nF = run_nuFnu_vFC(d_fit, nuobs, Tobs, env)
  nupk_data, nFpk_data = extract_peaks(tnu, nF)
  print('Done')

  # peaks function
  func_peaks = peaks_function_fromFit(Tf, g2, popt_lfac2, popt_nu, popt_L)

  # fitting procedure
  def residuals(params, T, nupk_data, nFpk_data):
    k, xi_sat, s = params
    nupk_model, nFpk_model = func_peaks(T, k, xi_sat, s)
    r1 = nupk_model - nupk_data
    r2 = nFpk_model - nFpk_data  
    return np.concatenate([r1, r2])

  p0 = [0.3, 6., 2.]
  bounds = ([0.01, 0.1, 0.1], [10., 100., 200.])
  res = least_squares(residuals, p0, bounds=bounds, args=(T, nupk_data, nFpk_data))
  popt_xi = res.x

  if returnAll:
    return Tf, t_max, popt_lfac2, popt_ShSt, popt_nu, popt_L, popt_xi
  else:
    return popt_xi


### Produce radiation and peaks
def get_radiation_vFC(key, front='RS', norm=True,
  T_max=5, lognu_min=-3, lognu_max=2, NT=450, Nnu=500,
  noOut=False):
  '''
  \nu F_\nu (\nu, T) from front ('RS' or 'FS')
  Create file with normalized observed frequency, time and flux
  '''

  fpath = get_radfile_thinshell(key, front)
  exists = os.path.isfile(fpath)
  samesize = True
  if exists:
    obs = np.load(fpath)
    nu, T, nF = obs['nu'], obs['T'], obs['nF']
    if (len(nu) != Nnu) or (len(T) != NT):
      samesize = False

  if (not exists) or (not samesize):
    nu, T, env = obs_arrays(key, False, Tmax, NT, lognu_min, lognu_max, Nnu)
    z, nu0, T0 = (4, env.nu0, env.T0) if (front == 'RS') else (1, env.nu0FS, env.T0FS)
    data = open_rundata(key, z)
    if type(data) == bool:
      analyze_run(key, savefile=True, noOut=True)
      data = open_rundata(key, z)
    data = replace_early_withfit(data)
    nF = run_nuFnu_vFC(data, nu, T, env, norm)
    if norm:
      nu /= nu0
      T = (T - env.Ts)/T0
    np.savez(fpath, nu=nu, T=T, nF=nF)
  if not noOut:
    return nu, T, nF
  

def extract_peaks(nu, nF):
  '''
  Extract the peaks of nuFnu
  '''
  NT = nF.shape[0]
  nu_pk, nF_pk = np.zeros((2, NT))
  for j in range(NT):
    nF_t = nF[j]
    i = np.argmax(nF_t)
    nu_pk[j] += nu[i]
    nF_pk[j] += nF_t[i]
  return nu_pk, nF_pk

def get_peaks_from_data(key, front='RS', withT=False):
  '''
  Obtain peak frequency and flux from simulated data, in normalized units
  from front ('RS' or 'FS')
  
  '''

  env = MyEnv(key)
  NT = env.Nsh4 if (front == 'RS') else env.Nsh1
  nu, T, nF = get_radiation_vFC(key, front=front, norm=True, NT=NT, Nnu=500)
  NT = len(T)
  nu_pk, nF_pk = extract_peaks(nu, nF)
  if withT:
    return T + 1., nu_pk, nF_pk
  else:
    return nu_pk, nF_pk


def analyze_nubk(key):
  '''
  nu_bk/nu_hf vs Delta R/R0 and ton/toff
  Recalculates flux to avoids array size mismatch + better precision
  '''

  
  z = 4  # add analysis for FS later
  filepath = get_radfile_activity(key, 'RS')
  env = MyEnv(key)
  beta = getattr(env, f'beta{z}')
  data = open_rundata(key, z)
  data = replace_early_withfit(data)
  data = data.drop_duplicates(subset='i', keep='first')
  #data = extrapolate_early(data, env)
  df0 = openData(key, 0)

  # crossed radius by the shock and corresponding ton/toff
  dR = (data.x * c_/env.R0) - 1.
  indices = data.i.astype(int).to_list()
  x0 = df0.loc[df0['i'].isin(indices)].to_numpy()
  ton = ((env.R0 - (x0 * c_))/(c_ * env.beta4))/env.toff
  
  # arrays of obs time and frequency only near peaks
  Nj = len(data)
  last = data.iloc[-1]
  Tlast = last.t + env.t0 - last.x
  lognu_min = np.log10(get_variable(last, "nu_m2", env)/env.nu0) - 1
  Tmax = ((Tlast + env.Ts)/env.T0) * 2
  nuobs, Tobs, env = obs_arrays(key, normed=False,
    Tmax=Tmax, NT=Nj, lognu_min=lognu_min, lognu_max=1, Nnu=500)
  nub = nuobs/env.nu0

  # calculate flux
  nF = run_nuFnu_vFC(data, nuobs, Tobs, env, norm=True)
  # find peaks
  nu_pk, nF_pk = np.zeros((2, Nj))
  for j in range(Nj):
    nF_t = nF[j]
    i = np.argmax(nF_t)
    nu_pk[j] += nub[i]
    nF_pk[j] += nF_t[i]

  nu_bk, nu_hf = np.zeros((2, Nj))

  # for j in range(Nj):
  #   row = data.iloc[j]
  #   # radius crossed by the shock when reaching the shell
  #   i = row.i.astype(int)
  #   dR[j] += (row.x * c_/env.R0) - 1.
  #   # corresponding launch time by the source
  #   ton[j] += (env.R0 - (df0.at[i, 'x'] * c_))/(c_ * env.beta4)
  #   ton[j] /= env.toff
  #   # add flux contribution
  #   nF += nub * get_Fnu_vFC(row, nuobs, Tobs, env, norm=True)
  #   # find peak flux and corresponding freq in the current obs. flux
  
  # for j in range(Nj):
  #   nF_t = nF[j]
  #   i = np.argmax(nF_t)
  #   nu_pk[j] = nub[i]
  #   nF_pk[j] = nF_t[i]

  for j in range(Nj):
    Ton = get_variable(data.iloc[j], 'Ton', env)
    ion = np.searchsorted(Tobs, Ton)
    nu_bk[j] += nu_pk[ion]
    imax = np.argmax(nF[j])
    nu_Fmax = nu_pk[imax]
    nF_bk = nF_pk[imax]
    i_hf = np.searchsorted(nF_pk[nu_pk>nu_bk], .5*nF_bk) - 1
    nu_hf[j] += nu_pk[nu_pk>nu_bk][i_hf]
  
  nu_bkhf = nu_bk / nu_hf

  np.savez(filepath, dR=dR, ton=ton, nu_bkhf=nu_bkhf)
  return dR, ton, nu_bk, nu_hf, nu_pk, nF_pk
    
