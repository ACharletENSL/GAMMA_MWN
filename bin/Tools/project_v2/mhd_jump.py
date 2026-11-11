# -*- coding: utf-8 -*-
'''
Jump conditions of a perpendicular (relativistic) MHD shock running into a cold magnetized plasma
(Paper I, Sect. 5, 2026-11-11). Units: m c^2 = 1, upstream proper density n1 = 1.

Upstream (1): cold, proper field b1 = B1/sqrt(4 pi), magnetization sigma = b1^2/(n1 m c^2).
Downstream (2): proper density n2 = r, proper internal energy e2 = theta r (theta = thermal energy per
particle / m c^2), pressure p2 = (hat gamma - 1) e2, hat gamma = (4 v + 1)/(3 v), v = 1 + theta
(Kumar & Granot 2003), proper field b2 = r b1 (flux freezing, perpendicular field).

Given the Lorentz factor of the upstream relative to the downstream, G (= Gamma_34), the shock-frame
conservation of particles, energy and momentum, with E continuous, give r and theta:
  n1 G1 b1 = n2 G2 b2 (particles),  b/n conserved (flux freezing),
  (w1 + b1^2) G1^2 beta1 = (w2 + b2^2) G2^2 beta2                       (energy)
  (w1 + b1^2) G1^2 beta1^2 + p1 + b1^2/2 = (w2 + b2^2) G2^2 beta2^2 + p2 + b2^2/2   (momentum)
with w = n + e + p. sigma -> 0: theta = G - 1 and r = (hat gamma G + 1)/(hat gamma - 1) (hydrodynamics).
No solution (no shock) when the upstream is not super-fast-magnetosonic in the shock frame.
'''

import numpy as np
from scipy.optimize import brentq


def _resid(beta2, G, sig):
  b34 = np.sqrt(1 - 1/G**2)
  beta1 = (beta2 + b34)/(1 + beta2*b34)
  G1, G2 = 1/np.sqrt(1 - beta1**2), 1/np.sqrt(1 - beta2**2)
  r = G1*beta1/(G2*beta2)
  b2sq = r**2*sig
  FE = (1 + sig)*G1**2*beta1
  FM = (1 + sig)*G1**2*beta1**2 + sig/2
  w2 = FE/(G2**2*beta2) - b2sq
  X = (w2 - r)/r                                    # hat gamma theta
  if X <= 0:
    return np.nan, r, -1.
  v = ((3 + 3*X) + np.sqrt((3 + 3*X)**2 + 16))/8    # 4 v^2 - 3(1 + X) v - 1 = 0
  hg = (4*v + 1)/(3*v)
  e2 = r*(v - 1)
  p2 = (hg - 1)*e2
  return FM - ((w2 + b2sq)*G2**2*beta2**2 + p2 + b2sq/2), r, v - 1


def jump(G, sig, n=400):
  '''(r, theta) of the downstream, or None if there is no shock'''
  if G <= 1 + 1e-12:
    return None
  bs = np.geomspace(1e-6, 1 - 1e-9, n)
  vals = np.array([_resid(b, G, sig)[0] for b in bs])
  roots = []
  for i in range(n - 1):
    a, c = vals[i], vals[i + 1]
    if np.isfinite(a) and np.isfinite(c) and a*c < 0:
      b0 = brentq(lambda b: _resid(b, G, sig)[0], bs[i], bs[i + 1], xtol=1e-14)
      _, r, th = _resid(b0, G, sig)
      if r > 1 + 1e-9 and th > 0:
        roots.append((r, th))
  if not roots:
    return None
  return max(roots, key=lambda x: x[0])               # the compressive (shock) branch


if __name__ == '__main__':
  print('hydro limit (sigma = 1e-12): theta/(G-1), r vs (hg G + 1)/(hg - 1)')
  for G in (1.001, 1.05, 1.5, 3., 10.):
    r, th = jump(G, 1e-12)
    v = G; hg = (4*v + 1)/(3*v)
    print(f'  G={G:6.3f}  theta/(G-1)={th/(G - 1):.6f}  r={r:.5f}  hydro r={(hg*G + 1)/(hg - 1):.5f}')
  print('magnetized: sigma, G -> r, theta/(G-1), eps_B = r sigma/(2 theta), strong-shock eps_B = 2 G sigma/(G-1)')
  for G in (1.0175, 1.08, 0.65 + 1, 2.):
    for s in (1e-4, 1e-3, 1e-2, 3e-2):
      j = jump(G, s)
      if j is None:
        print(f'  G-1={G - 1:.4f} sigma={s:.0e}: no shock (M_A^2 = 2(G-1)/sigma = {2*(G - 1)/s:.3g})')
      else:
        r, th = j
        print(f'  G-1={G - 1:.4f} sigma={s:.0e}: r={r:.3f} theta/(G-1)={th/(G - 1):.3f} eps_B={r*s/(2*th):.3g} '
              f'(strong-shock {2*G*s/(G - 1):.3g})')
