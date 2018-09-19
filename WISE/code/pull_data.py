from GetData import WISE_LC
import pandas as pd
import numpy as np
import time

'''
THIS IS THE SIMPLE SCRIPT TO RUN TO GET DATA FOR ANALYSIS
'''

# the input file to get from
df = pd.read_csv('bright_clean_w1w2_gt3.csv')

# some selection cut(s)
cut = np.where((df['phot_bp_mean_mag'] - df['phot_rp_mean_mag'] < 1) & (df['w1mpro'] > 8))[0]

# sort the objects in order of brightness
ss = np.argsort(df['w1mpro'].values[cut])



for k in range(0, ss.size):
    print(str(k)+', WISE '+df['original_ext_source_id'].values[cut[ss[k]]], time.ctime())

    tmp = WISE_LC('WISE '+df['original_ext_source_id'].values[cut[ss[k]]])
    if tmp==1: # if successfully pulled LC, sleep to be polite for 2 seconds
        time.sleep(2)


print('> data has been downloaded, basic figures created.')
