# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains the classes to hold variables to plot
WIP
'''

'''
Idée :
pouvoir ajouter les variables à plot de façon simple, sans avoir à naviguer entre 4 fichiers différents chaque fois
doit contenir :
- nom
- nom (LaTeX) pour le y label
- unité(s)
- valeur(s) de référence/scaling
- fonction génératrice (si valeur dérivée)
- valeurs théoriques/fonction génératrice dans le modèle

Question ouverte : Comment je gère les timeseries d'une cellule ?
-> switch dans le générateur?
de quoi j'ai besoin pour cell timeseries? -> pareil que snapshot mais i cellule en plus
'''

vardict_cell = {  # dictionnary of the variables classes 
  "rho":CellRho,
}

class CellVar:

  def __init__(self, var, key):
    self = vardict_cell[var](key)


class CellRho:

  def __init__(self, key):
    env = MyEnv(key)