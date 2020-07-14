# -*- coding: utf-8 cspell:disable -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: dev
#     language: python
#     name: dev
# ---

# +
import joblib
import datetime
from pathlib import Path

import pytoml
import us
# -

home = str(Path.home())

# %run "/home/gsinha/admin/db/dev/Python/projects/models/claims/common.py"

START_DATE = datetime.date(1995, 1, 1)
with open(home + "/.config/gcm/gcm.toml", "rb") as f:
    config = pytoml.load(f)
    FRED_API_KEY = config["api_keys"]["fred"]

# +
# %%time

fred_fname = "../data/fred_data"
states = [x.abbr for x in us.STATES] + ["DC", "US"]

ic_df, fred_df, ic_date, w_52_pct_chg_df = get_labor_data(states)
ic_df = ic_df.interpolate()

with open(fred_fname + ".pkl", "wb") as f:
    joblib.dump(
        {"ic_df": ic_df, "fred_df": fred_df, "w_52_pct_chg_df": w_52_pct_chg_df, "ic_date": ic_date}, f
    )
# -

