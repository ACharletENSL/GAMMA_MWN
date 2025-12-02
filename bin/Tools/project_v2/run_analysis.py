'''
Analyzing a run (extracting hydro, producing radiation..)
'''

from IO import *
from radiation import *
from phys_constants import *
from hydro_fits import get_hydrofits_shell, replace_withfit

def thinshell_radiation(key, z, nuobs, Tobs):
  '''
  Derives observed flux from shock front (z=4: RS, z=1: FS)
    in thinshell approximation
  '''
  return 0.


# Extracting hydro data
#### Thinshell approx: only cells downstream are interesting
def critical_radii(key, z):
  '''
    Rc:     power-law behavior to constant transition (Tsph)
    Rf:     crossing time
  '''
  m, x_sph = extract_plaws(key, z)[0]
  Rc = x_sph**(-1/m)
  Rf = get_Rcrossing(key, z)
  return Rc, Rf


def critical_times(key, z, k=0.5):
  '''
  Important normalized observed times for peak modelization:
    Tc:     power-law behavior to constant transition
    Tf:     crossing time
    Tsat:   saturation of effective angle contribution from beaming
  '''
  Rc, Rf = critical_radii(key, z)
  Tf = Rf**(m+1)
  Tc = Rc**(m+1)
  env = MyEnv(key)
  g2 = (env.gRS if z == 4 else env.gFS)**2
  Tsat = 1 + (m+1)/(g2*k)
  return Tc, Tf, Tsat
  

def extract_plaws(key, z):
  '''
  Returns the fitting parameters for downstream LF and shock strength
  '''

  data = open_rundata(key, z)
  popt_lfac2, popt_ShSt = get_hydrofits_shell(data)
  x_lf, m, s_lf = popt_lfac2
  x_sh, n, s_sh = popt_ShSt
  return [-m, x_lf], [-n, x_sh]

def get_Rcrossing(key, z):
  '''
  Get final radius of the shock front at crossing, in units R0
  '''
  data = open_rundata(key, z)
  env = MyEnv(key)
  Rf = data.iloc[-1].x * c_ / env.R0
  return Rf


### Extract hydro data
def analyze_run(key, itmin=0, itmax=None,
    cells=[1, 2, 3, 4], savefile=True, noOut=False):
  '''
  Analyze a run, returning pandas dataframes one for each cell
  1: downstream FS, 2: CD in S2, 3: CD in S3, 4: downstream RS
  if savefile, writes it in a corresponding .csv file
  !!! if itmin != 0, starts at first iteration AFTER itmin
  '''

  fpaths = [get_runfile(key, z)[0] for z in cells]
  df0 = openData(key, it=0)
  varlist = df0.keys().to_list()
  varlist.insert(1, 'dt')
  varlist.insert(0, 'it')
  its = np.array(dataList(key, itmin, itmax))
  if itmin:
    its = [it for it in its if i>itmin]
  Nc = len(cells)
  Nj = len(its)
  Nk = len(varlist)
  datas = np.zeros((Nc, Nj, Nk))
  dics = [{} for i in range(Nc)]
  dfs = []
  for j, it in enumerate(its):
    if it % 100 == 0:
      print(f"Analyzing file of it = {it}")
    df, t = openData_withtime(key, it)
    for i, z in enumerate(cells):
      cell = df_get_frontsnCD(df, z)
      if cell.empty:
        values = np.zeros(Nk)
        values[0:2] = it, t
      else:
        values = cell.to_numpy(copy=True)
        # leave room for dt
        values = np.insert(values, 1, 0.)
        values = np.insert(values, 0, it)
      datas[i, j] += values

  # add dt
  for i, z in enumerate(cells):
    t = datas[i,:,1]
    dt = np.gradient(t)
    datas[i,:,2] += dt
    dics[i] = {varlist[k]:datas[i,:,k] for k in range(Nk)}
    df = pd.DataFrame.from_dict(dics[i]).set_index('it')
    dfs.append(df)
    if savefile:
      df.to_csv(fpaths[i])
  if not noOut:
    return dfs


### Produce radiation and peaks
def get_radiation_vFC(key, front='RS', norm=True,
  Tmax=5, NT=450, lognu_min=-3, lognu_max=2, Nnu=500):
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
    nF = run_nuFnu_vFC(data, nu, T, env, norm)
    if norm:
      nu /= nu0
      T = (T - env.Ts)/T0
      np.savez(fpath, nu=nu, T=T, nF=nF)
  return nu, T, nF


def get_peaks_from_data(key, front='RS'):
  '''
  Obtain peak frequency and flux from simulated data, in normalized units
  from front ('RS' or 'FS')
  
  '''

  env = MyEnv(key)
  NT = env.Nsh4 if (front == 'RS') else env.Nsh1
  nu, T, nF = get_radiation_vFC(key, front=front, norm=True, NT=NT, Nnu=500)
  NT = len(T)
  nu_pk, nF_pk = np.zeros((2, NT))
  for j in range(NT):
    nF_t = nF[j]
    i = np.argmax(nF_t)
    nu_pk[j] += nu[i]
    nF_pk[j] += nF_t[i]
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
  data = replace_withfit(data)
  data = data.drop_duplicates(subset='i', keep='first')
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
    i_hf = np.searchsorted(nF_pk[nu_pk>nu_Fmax], .5*nF_bk)
    nu_hf[j] += nu_pk[nu_pk>nu_bk][i_hf]
  
  nu_bkhf = nu_bk / nu_hf

  np.savez(filepath, dR=dR, ton=ton, nu_bkhf=nu_bkhf)
  return dR, ton, nu_bk, nu_hf, nu_pk, nF_pk
    
