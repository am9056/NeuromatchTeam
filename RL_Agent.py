#%%
import numpy as np
import matplotlib.pyplot as plt
import os, requests

#%%
#@title Data retrieval
fname = []
for j in range(3):
  fname.append('steinmetz_part%d.npz'%j)
url = ["https://osf.io/agvxh/download"]
url.append("https://osf.io/uv3mw/download")
url.append("https://osf.io/ehmw2/download")

for j in range(len(url)):
  if not os.path.isfile(fname[j]):
    try:
      r = requests.get(url[j])
    except requests.ConnectionError:
      print("!!! Failed to download data !!!")
    else:
      if r.status_code != requests.codes.ok:
        print("!!! Failed to download data !!!")
      else:
        with open(fname[j], "wb") as fid:
          fid.write(r.content)

alldat = np.array([])
for j in range(len(fname)):
  alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz'%j, allow_pickle=True)['dat']))

# select just one of the recordings here. 11 is nice because it has some neurons in vis ctx.
dat = alldat[11]
print(dat.keys())

#%%

# Rascola-Wagner Model of Value
def value(r, alpha):
    v = np.full(r.shape, np.nan)
    v[0] = 0

    for t in np.arange(len(r)-1):
        v[t+1] = v[t] + alpha*(r[t] - v[t])

    return v


V = value(dat['feedback_type'], 0.1)
plt.plot(dat['feedback_type'], 'r--')
plt.plot(V)

plt.show()

#%%

