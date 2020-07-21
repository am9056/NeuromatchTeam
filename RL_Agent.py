#%%
import numpy as np
import matplotlib.pyplot as plt
import os, requests

#%% Download Data

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

# R = np.ones((1, 1000))[0]
R = dat['feedback_type']

V = value(R, 0.1)
plt.plot(R, 'r--')
plt.plot(V)

plt.show()

#%%
total_contrasts = dat["contrast_left"] - dat["contrast_right"]
gotrials = np.logical_and(dat["contrast_left"]!=0, dat["contrast_right"]!=0)
go_contrasts = total_contrasts[gotrials]

ntrials = len(go_contrasts)

beta = 30.

prob_left = np.exp(beta*go_contrasts)/\
            (np.exp(-beta*go_contrasts)[0] + np.exp(beta*go_contrasts))

response = dat["response"][gotrials]
response[response == -1] = 0
plt.plot(response)
plt.plot(prob_left)
plt.show()

#%%

print(dat['feedback_type'].shape)
plt.plot(dat['feedback_type'])

mean_smoothed_response = my_moving_window(dat['feedback_type'], window = 20)
plt.plot(mean_smoothed_response)

x_points, gamma = np.arange(0,20,1), .1

exp_function = gamma* np.exp(-gamma * x_points)

value = np.convolve(exp_function,dat['feedback_type'],mode = 'same')
plt.plot(value)
# plt.figure
# plt.plot(exp_function)

#%%

idx = np.logical_and(dat['contrast_left'] != 0 , dat['contrast_right'] != 0)
# print(idx)
total_contrast = dat['contrast_left'][idx] - dat['contrast_right'][idx]
total_response = dat['response'][idx]
noisy_response = total_response + 0.2 * np.random.rand(len(total_response))
plt.scatter(total_contrast,noisy_response)

contrasts = np.unique(total_contrast)
mean_response = np.zeros_like(contrasts)
for i,c in enumerate(contrasts):
  idx2 = total_contrast == c
  mean_response[i] = np.mean(total_response[idx2])

plt.plot(contrasts, mean_response)

#%%

# Softmax Function
fcv = total_contrast * value[idx]
beta = 3
PCL = np.exp(beta*fcv)/ (np.exp(-beta*fcv) + np.exp(beta*fcv))
plt.scatter(total_contrast,PCL)