# -*- coding: utf-8 -*-
# @Author: acharlet

'''
New script to redo rad plots for the article
to do:
1. get_run_radiation pour générer fichiers dans les 4 cas
  -> fichiers différents Band et BPL
2. plots comparant les 4
'''

from plotting_scripts_new import *
plt.ion()

cols_dic = {'RS':'r', 'FS':'b', 'RS+FS':'k', 'CD':'purple',
  'CD-':'mediumvioletred', 'CD+':'mediumpurple', '3':'darkred', '2':'darkblue'}
spsh_cols = {'Band':'g', 'plaw':'teal'}