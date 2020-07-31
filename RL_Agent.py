#%%
import numpy as np
import matplotlib.pyplot as plt
# import os, requests
# from scipy.optimize import minimize

#%% Functions


def load_data():
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
      alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz'%j,
                                          allow_pickle=True)['dat']))

    return alldat


# Rascola-Wagner Model of Value
def gen_value(alpha, r):
    v = np.full(r.shape, np.nan)
    v[0] = 0

    for t in np.arange(len(r)-1):
        v[t+1] = v[t] + alpha*(r[t] - v[t])

    return v


def nll(alpha, beta, choice_, dat):
    go_trials = np.logical_or(dat["contrast_left"] != 0,
                              dat["contrast_right"] != 0)

    total_contrasts_ = dat["contrast_left"] - dat["contrast_right"]
    total_contrasts = total_contrasts_[go_trials]

    choice = choice_[go_trials]
    choice[choice == -1] = 0

    y_hat = np.exp(beta*total_contrasts) / \
            (np.exp(-beta*total_contrasts) + np.exp(beta*total_contrasts)) +\
            np.finfo(float).eps # avoids zero y_hats

    lik = y_hat**choice*(1-y_hat)**(1-choice)
    return -np.sum(np.log(lik))


def softmax_data_gen(params, dat):
    (alpha, beta) = params

    v = gen_value(alpha, dat['feedback_type'])

    idx = np.logical_or(contrast_left != 0, contrast_right != 0)
    total_contrast = contrast_left[idx] - contrast_right[idx]

    fcv = total_contrast * v[idx]
    pcl = np.exp(beta * fcv) / (np.exp(-beta * fcv) + np.exp(beta * fcv))

    myrand = np.random.rand(1, len(pcl))[0]
    fake_choice = np.zeros_like(pcl)

    for i in range(len(pcl)):
        if myrand[i] < pcl[i]:
            fake_choice[i] = 1  # Choose right
        else:
            fake_choice[i] = -1  # Choose left

    return fake_choice


#%%

total_response = response[idx]
noisy_response = total_response + 0.2 * np.random.rand(len(total_response))

contrasts = np.unique(total_contrast)
mean_response = np.zeros_like(contrasts)
for i, c in enumerate(contrasts):
    idx2 = total_contrast == c
    mean_response[i] = np.mean(total_response[idx2])

plt.figure()
plt.scatter(total_contrast, noisy_response)
plt.plot(contrasts, mean_response)
plt.xlabel('Contrast')
plt.ylabel('Original Choices')
plt.title('Original Data')

mean_fake_response = np.zeros_like(contrasts)
for i, c in enumerate(contrasts):
    idx2 = total_contrast == c
    mean_fake_response[i] = np.mean(fake_choice[idx2])

plt.figure()
plt.scatter(total_contrast,
            fake_choice + 0.2 * np.random.rand(len(fake_choice)))
plt.plot(contrasts, mean_fake_response)
plt.xlabel('Contrast')
plt.ylabel('Generated Choices')
plt.title('Generated Data')

plt.figure()
plt.scatter(total_contrast, PCL)
plt.xlabel('Contrast')
plt.ylabel('P(C = L)')
plt.title('Softmax Function')

# Today's needs:
# 1. Recover generative parameters
#   a. make some good figures
# 2. Cross validation?
# 3. Fit mice

#%% Let's just jump in

# Generate Synthetic Data
synth_data = softmax_data_gen((3, 0.3), dat)

nll(alpha, beta, choice_, dat)

#%%


def my_exp(beta, X):
    return np.exp(np.dot(X, beta.T)[0])


def multi_nom_probs(betas, dat):
    # Make design matrix - contrast left, contrast right, diff in contrast
    X = np.array([dat["contrast_left"],
                  dat["contrast_right"],
                  (dat["contrast_left"] - dat["contrast_right"]) * \
                  (dat["contrast_left"] + dat["contrast_right"])])
    X = X.T

    # reshape betas
    betas = betas.reshape(3, 3)
    beta1 = betas[:, 0]
    beta2 = betas[:, 1]
    beta3 = betas[:, 2]

    # Preallocate probablities
    probs = np.zeros_like(X)

    # softmax function for each choice (left, right, nogo)
    for n in np.arange(len(X)):
        p_l = my_exp(X[n, np.newaxis], beta1) / \
              np.sum([my_exp(X[n, np.newaxis], beta1),
                      my_exp(X[n, np.newaxis], beta2),
                      my_exp(X[n, np.newaxis], beta3)])

        p_r = my_exp(X[n, np.newaxis], beta2) / \
              np.sum([my_exp(X[n, np.newaxis], beta1),
                      my_exp(X[n, np.newaxis], beta2),
                      my_exp(X[n, np.newaxis], beta3)])

        p_ng = my_exp(X[n, np.newaxis], beta3) / \
               np.sum([my_exp(X[n, np.newaxis], beta1),
                       my_exp(X[n, np.newaxis], beta2),
                       my_exp(X[n, np.newaxis], beta3)])

        probs[n, :] = [p_l, p_r, p_ng]

    return probs


def nll_multinom(betas, ys, dat):
    probs = multi_nom_probs(betas, dat) # Get probablities from above

    # One hot encoding of choice (l, r, nogo)
    Y = np.zeros((len(ys), 3))
    Y[ys == 1, 0] = 1
    Y[ys == -1, 1] = 1
    Y[ys == 0, 2] = 1

    # Calculate log likelihood using (ignoring multinomial coeff)?
    lik = np.prod(probs**Y, axis=1)
    return -np.sum(np.log(lik))


def gen_synth_data(betas, dat):
    probs = multi_nom_probs(betas, dat)

    choices = np.zeros_like(dat["response"])

    for n in np.arange(len(dat["response"])):

        r = np.random.rand()
        p = np.cumsum(probs[n, :])

        if r < p[0]:
            choices[n] = 1
        elif p[0] < r < p[1]:
            choices[n] = -1
        else:
            choices[n] = 0

    return choices


#%%
# betas = np.random.rand(9)*100 - 50
# synth_data = gen_synth_data(betas, dat)
# data_to_fit = synth_data

data_to_fit = dat["response"]

test = minimize(lambda x: nll_multinom(x, data_to_fit, dat),
                x0=np.random.rand(9))

plt.plot(gen_synth_data(test.x, dat), 'r.')
plt.plot(data_to_fit, 'k.')

plt.title(str(np.mean(gen_synth_data(test.x, dat) == data_to_fit)))
plt.show()
