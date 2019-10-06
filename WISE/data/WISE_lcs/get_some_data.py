from GetData import WISE_LC
import pandas as pd
import numpy as np
import time

'''
THIS IS THE SIMPLE SCRIPT TO RUN TO GET DATA FOR ANALYSIS
'''
"""
# the input file to get from
df = pd.read_csv('massive.csv')

span = 0
# span = 1

half = int(len(df)/2)

if span == 0:
    rng0 = 1262
    rng1 = half

if span == 1:
    rng0 = 4388
    rng1 = len(df)

for k in range(rng0, rng1):
    print(str(k)+'/'+str(rng1)+', WISE '+df['original_ext_source_id'].values[k], ', '+df['CommonName'].values[k], time.ctime())

    tmp = WISE_LC('WISE '+df['original_ext_source_id'].values[k], title=df['CommonName'].values[k], alldata=True)
    if tmp==1: # if successfully pulled LC, sleep to be polite for 2 seconds
        time.sleep(2)
"""
tmp = WISE_LC('WISE J050128.62-701120.2', title='J050128.62-701120.2', alldata=True)

print('> data has been downloaded, basic figures created. HAVE FUN! (& ping Trevor)')
