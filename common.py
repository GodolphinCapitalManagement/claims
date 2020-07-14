# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: dev
#     language: python
#     name: dev
# ---

import datetime
import pathlib

# +
import pandas as pd
import numpy as np
import pytoml

from fredapi import Fred

# +
START_DATE = datetime.date(1995, 1, 1)
CRISIS_START_DATE = datetime.date(2020, 3, 14)
HOME_DIR = str(pathlib.Path.home())

with open(HOME_DIR + "/.config/gcm/gcm.toml", "rb") as f:
    config = pytoml.load(f)
    FRED_API_KEY = config["api_keys"]["fred"]


# +
def states_data(suffix, states, start_date, fred):
    ''' gets data from FRED for a list of indices '''

    
    indices = [x + suffix if x != "US" else "ICSA" for x in states]
    df = []
    for v in indices:
        x =  pd.Series(
                fred.get_series(
                    v, observation_start=start_date), name=v
            )
        df.append(x)

    y_ts = pd.concat(df, axis=1)
    y_ts.columns = states

    return y_ts

def get_labor_data(states, itd=True):
    ''' get labor market data from STL '''
    
    fred = Fred(api_key=FRED_API_KEY)

    ur_raw = states_data("UR", states, START_DATE, fred)
    ur = ur_raw.diff().iloc[-1, ]

    ur_df = ur.to_frame(name="ur").reset_index()
    ur_df.rename(columns={"index": "state"}, inplace=True)
    
    ic_raw = states_data("ICLAIMS", states, START_DATE, fred)
    if  itd:
        ic = ic_raw.loc[CRISIS_START_DATE:, :].sum(axis=0)
    else:
        ic = ic_raw.rolling(window=4).sum().iloc[-1, :]

    ic_df = ic.to_frame(name="ic").reset_index()
    ic_df.rename(columns={"index": "state"}, inplace=True)
    
    all_df = pd.merge(ur_df, ic_df, on="state")
    
    cc_raw = states_data("CCLAIMS", states, START_DATE, fred)
    cc_df = cc_raw.iloc[-1].to_frame(name="cc").reset_index()
    cc_df.rename(columns={"index": "state"}, inplace=True)
    
    all_df = pd.merge(all_df, cc_df, on="state")
    w_52_pct_chg_df = ic_raw.pct_change(periods=52)
    
    return ic_raw, all_df, ic_raw.index[-1].date(), w_52_pct_chg_df
# -


