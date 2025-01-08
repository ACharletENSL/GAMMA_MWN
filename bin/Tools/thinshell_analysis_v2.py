# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains a rewrite of what is performed in
thinshell_analysis, but without the dependancy on regions id
This includes some rewrites of equivalent data_IO
'''

from thinshell_analysis import *

# identify shocks and downstream
def df_get_shockindices(df, n=4, m=1):
  '''
  Returns array of tuples, for each shock the indices of downstream, front, and upstream
  '''
  sh = df.loc[(df['Sd']==1.) & (df['trac']>0.)]
  out = []

  if sh.empty:
    return out

  splits = np.flatnonzero(np.diff(sh.index)!=1)
  sub_arrs = np.split(sh.index, splits+1)
  shlist = [df.iloc[iarr] for iarr in sub_arrs]

  for sh in shlist:
    iL, iR = int(sh.index.min()) - 1, int(sh.index.max()) + 1
    # check which side is down/upstream from densities
    if df.iloc[iL]['rho'] > df.iloc[iR]['rho']:
      # forward shock
      iD, iF, iU = iL-n, iR-1, iR+m
    else:
      # reverse shock
      iD, iF, iU = iR+n, iL+1, iL-m
    out.append((iD, iF, iU))
  return out


# extract variables (hydro and radiation, have equivalents to Daigne & Mochokovitch 00)
def analyze_run_thinshell(key, Nsh=2, itmin=0, itmax=None):
  '''
  Applies df_get_all_thinshell to all data files, creates Nsh data files (one per shock)
  Make sure varlist is the same between the two!
  '''
  varlist = ['t', 'r', 'dr', 'rho', 'vx', 'lfac', 'p', 
    'ShSt', 'lfacsh', 'Ton', 'Tth', 'Tej', 'B', 'gma_m', 'nu_m', 'Lp', 'Lth']

  outpath = get_contribspath(key)
  nested = [[] for _ in range(Nsh)] # create Nsh empty lists to fill
  its_asindex = [[] for _ in range(Nsh)]

  its = dataList(key, itmin, itmax)[0:]
  if itmin == 0:
    its = its[1:]
  Nf = len(its)
  istop = -1
  skips = 0
  calc_its = []

  for it in its:
    print(f"Open file {it}")
    df = openData(key, it)
    data = df_get_all_thinshell(df)

    # check number of identified shocks in data
    if data.empty:
      # skip file if empty, break loop after too many skips
      skips +=1
      if skips > 10 and len(data)>0:
        break
      continue

    elif len(data) == Nsh:
      # everything's good
      for i, out, its_out in zip(range(Nsh), nested, its_asindex):
        out.append(data.iloc[i].to_numpy())
        its_out.append(it)

    
    elif len(data) < Nsh:
      # some shocks are missing (have crossed or maybe undetected at this time from noise)
      try:
        last_ids = [out[-1][0] for out in nested]
        for i, row in data.iterrows():
          # add to list with closest shock id
          i_in = int(find_closest(np.array(last_ids), i))
          nested[i_in].append(data.iloc[i].to_numpy())
          its_asindex[i_in].append(it)
      except IndexError:
        for i, row in data.iterrows():
          # add to list with closest shock id
          nested[i].append(data.iloc[i].to_numpy())
          its_asindex[i].append(it)
    
    else:
      # we have more shocks than expected!
      print(f"Detected {len(data)} shocks, expected {Nsh}!")
      print("Need to implement")
      break
      # last_ids = [out[-1][0] for out in nested]
      # ish_list = data.index.to_list()
      
      # to_check = [[] for _ in range(Nsh)]
      # # regarder de quel anciens id les nouveaux sont plus proches

      # distances = []
      # for ish in enumerate(ish_list):
      #   # compare new shocks to previous shock ids
      #   i_cl = int(find_closest(np.array(last_ids), ish))
      #   id_cl = last_ids[i_cl]
      #   dist_cl = np.abs(id_cl-i_sh)
      #   distances[i_cl].append(dist_cl)
     
  # add corresponding dt and put data in Nsh files
  for i, data, its_out in zip(range(Nsh), nested, its_asindex):
    df_out = pd.DataFrame(data=data, columns=varlist, index=its_out)
    times = df_out['t']
    dts = [(times[j+1]-times[j-1])/2. for j in range(1,len(times)-1)]
    dts.insert(0, (3*times[0]-times[1])/2)
    dts.append((3*times[-1]-times[-2])/2)
    df_out.insert(1, "dt", dts)
    path = outpath + f'sh{i}.csv'
    df_out.to_csv(path)




def df_get_all_thinshell(df):
  '''
  Like in the v1 but using a different shock ID not built on zones and not checking values at CD
  '''

  varlist = ['t', 'r', 'dr', 'rho', 'vx', 'lfac', 'p', 
    'ShSt', 'lfacsh', 'Ton', 'Tth', 'Tej', 'B', 'gma_m', 'nu_m', 'Lp', 'Lth']
  

  ind_list = df_get_shockindices(df)
  if not len(ind_list):
    # empty index list, no shocks identified
    out = pd.DataFrame(columns=varlist)
    return out

  i_out = []
  data = []
  for i_arr in ind_list:
    # shock location at front, values taken downstream
    iD, iF, iU = i_arr
    t, r = df.iloc[iF][['t', 'x']]
    dr, rho, D, v, p = df.iloc[iD][['dx', 'rho', 'D', 'vx', 'p']]
    lfac = D/rho

    # relevant timescales in the obs frame are derived using downstream material velocity
    cell = df.iloc[iF].copy()
    cell['vx'] = v
    Ton, Tth, Tej = get_variable(cell, 'obsT')

    # shock strength, shock LF from R24 C1
    rhou, Du, vu = df.iloc[iU][['rho', 'D', 'vx']]
    lfacu = Du/rhou
    Drat = Du/D
    vsh = (Drat*vu - v)/(Drat - 1)
    ShSt = lfac*lfacu*(1-v*vu) - 1.
    lfacsh = derive_Lorentz(vsh)

    # emission
    B, gma, nu, Lp, Lth = get_vars(df.iloc[iD], ['B', 'gma_m', 'nu_m2', 'Lp', 'Lth'])
    
    # append data
    vals_out = np.array([t, r, dr, rho, v, lfac, p,
      ShSt, lfacsh, Ton, Tth, Tej, B, gma, nu, Lp, Lth])
    data.append(vals_out)
    i_out.append(iF)

  out = pd.DataFrame(data=data, columns=varlist, index=i_out)
  return out


# extract contributions, 1 per cell (bcause thinshell) (may include extrapolation at start)
def extract_contribs(key, extrapolate=True, smoothing=False):
  '''
  Extracts data per shock
  '''

  file_path = get_fpath_thinshell(key)
  outpath = get_contribspath(key)
  env = MyEnv(key)
  run_data = pd.read_csv(file_path, index_col=0)
  its = run_data.index.to_list()
  varlist = ['r', 'ish', 'v', 'rho', 'p', 'ShSt', 'Ton', 'Tej', 'Tth', 'lfac', 'nu_m', 'Lth', 'Lp']
  for sh in ['RS', 'FS']:
    keylist = ['time'] + [var + f'_{{{sh}}}' for var in varlist]
    ids = run_data[f'ish_{{{sh}}}'].to_numpy()
    its_split = np.split(its, np.flatnonzero(np.diff(ids))+1)
    its_contribs = []
    for it_group in (its_split[1:] if sh=='RS' else its_split): 
      its_contribs.append(it_group[0])
    df = run_data.loc[its_contribs][keylist].copy(deep=True)
    df['it']=its_contribs
    if extrapolate:
      df = extrapolate_early(df, env)
    # clean potential holes
    r = df[f'r_{{{sh}}}']
    holes = np.where(r == 0)[0]
    #print('holes at it', holes)
    for i in holes:
      if i == len(df)-1:
        df.iloc[i] = df.iloc[i-1]
      else:
        df.iloc[i] = .5*(df.iloc[i-1] + df.iloc[i+1])
    if smoothing:
      df = smooth_contribs(df, env)
    df.to_csv(outpath+sh+'.csv', index_label='it')


# calculate peaks freq and flux



# plotting functions
### simulation snapshots

### extracted vars (and any derived ones) with t/r/T

### contributions (basically vars but with added points for precise identification)

### flux (peaks, spectrum, lightcurve, time-integrated spectrum), with possible choice of bands