#%%
import numpy as np
import matplotlib.pyplot as plt
import os, requests
from scipy.optimize import minimize
from scipy.optimize import Bounds

#%% Load Data


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


alldat = load_data()
dat = alldat[11]

#%% Functions


def make_ys(responses):
    '''
    Generate a one-hot encoding of the response
    Inputs:
        responses - a 1xNtrials vector of the mouse's response
                    1 = left, -1 = right, 0 = no go
    Returns:
         ys - 1xN vector with 1 on left trials and 0 else
    '''
    ys = np.zeros_like(responses)
    ys[responses == 1] = 1 # Left Choices

    return ys


def get_value(alpha, feedback, v0=0):
    nsteps = len(feedback)-1
    V = np.full(feedback.shape, np.nan)
    V[0] = v0

    for t in np.arange(nsteps):
        V[t+1] = V[t] + alpha*(feedback[t] - V[t])

    return V


def get_prob_left(beta, contrasts, value):
    '''
    Generates choice probabilities according to soft-max

    Inputs:
        beta - beta for soft-max (either float or 1x2 np array)
        contrasts - left and right contrasts for each trial (2xNtrials)

    Returns:
         choice_prob - the choice probabilities for each trial
    '''

    # Get relevant contrasts
    l = contrasts[0, :]
    r = contrasts[1, :]

    q = beta[0] * (r-l)
    v = beta[1] * value
    x = q + v

    # Probability of left, right, nogo using softmax
    pl = 1 / (1 + np.exp(x))

    return pl


def calculate_nll(beta, responses, contrasts, value):
    ys = make_ys(responses)
    pl = get_prob_left(beta, contrasts, value)

    lik = (pl ** ys) * ((1 - pl) ** (1 - ys))
    return -np.sum(np.log(lik))


def nll_fit(alpha, beta, data, fittype):
    if fittype == 'no_value':
        values_ = np.ones_like(data["feedback_type"])
    elif fittype == 'value':
        values_ = get_value(alpha, data["feedback_type"])
    else:
        raise NameError('Not a fit type')

    contrasts_ = np.array([data["contrast_left"], data["contrast_right"]])
    responses_ = data["response"]

    idx = responses_ != 0

    contrasts = contrasts_[:, idx]
    responses = responses_[idx]
    values = values_[idx]

    return calculate_nll(beta, responses, contrasts, values)


def fit_model(data, fittype):
    if fittype == 'no_value':
        fit = minimize(lambda x: nll_fit(1, x, data, fittype),
                       x0=15*np.random.rand(2))

        opt_params = fit.x

    elif fittype == 'value':
        bounds = Bounds([0, -np.inf, -np.inf], [1.0, np.inf, np.inf])
        fit = minimize(lambda x: nll_fit(x[0], [x[1], x[2]], data, fittype),
                       x0=np.random.rand(3),
                       bounds=bounds)

        opt_params = fit.x
    else:
        raise NameError('Not a fit type')

    return opt_params


def gen_synth_responses(alpha, beta, data, fittype):
    if fittype == 'no_value':
        values_ = np.ones_like(data["feedback_type"])
    elif fittype == 'value':
        values_ = get_value(alpha, data["feedback_type"])
    else:
        raise NameError('Not a fit type')

    contrasts_ = np.array([data["contrast_left"], data["contrast_right"]])
    responses_ = data["response"]

    idx = responses_ != 0

    contrasts = contrasts_[:, idx]
    values = values_[idx]

    prob_left = get_prob_left(beta, contrasts, values)

    my_rand = np.random.rand(1, len(prob_left))[0]

    synth_responses = np.zeros_like(prob_left)
    synth_responses[my_rand < prob_left] = 1
    synth_responses[my_rand >= prob_left] = -1

    return synth_responses


def plot_psychometric(contrasts, responses, label):
    contrasts_diffs = np.unique(contrasts)
    pl = np.full(contrasts_diffs.shape, np.nan)

    for i, c in enumerate(contrasts_diffs):
        usethese = contrasts == c
        pl[i] = np.mean(responses[usethese] == 1)

    plt.plot(contrasts_diffs, pl, 'o-', label=label)
    return pl


#%%
data = alldat[11]

contrasts_ = np.array([data["contrast_left"], data["contrast_right"]])
responses_ = data["response"]
outcomes_ = data["feedback_type"]

value_ = get_value(1, outcomes_)

idx = responses_ != 0

contrasts = contrasts_[0, idx] - contrasts_[1, idx]
responses = responses_[idx]
value = value_[idx]

plot_psychometric(contrasts, responses, 'Mouse Data')
plt.show()

#%%

unrewarded = value == -1
rewarded = value == 1

# plot_psychometric(contrasts[unrewarded], responses[unrewarded], 'Unrewarded')
# plot_psychometric(contrasts[rewarded], responses[rewarded], 'Rewarded')

params_value = [fit_model(alldat[i], 'value')[0] for i in
                np.arange(len(alldat))]

#%%

c = np.linspace(-1, 1, 1000)
p_r = 1 - 1/(1+np.exp(params_value[1]*c + params_value[2]*1))
p_ur = 1 - 1/(1+np.exp(params_value[1]*c + params_value[2]*-1))

p_1 = 1 - 1/(1+np.exp(params_value[1]*c + params_value[2]*0.5))
p_2 = 1 - 1/(1+np.exp(params_value[1]*c + params_value[2]*-0.5))
p_3 = 1 - 1/(1+np.exp(params_value[1]*c + params_value[2]*0))

plt.plot(c, p_r, color=plt.cm.seismic(0)[0:3], linewidth=2.5,
         label="Previously Rewarded")

plt.plot(c, p_1, color=plt.cm.seismic(100)[0:3], linewidth=2.5)
plt.plot(c, p_3, color=plt.cm.seismic(200)[0:3], linewidth=2.5)
plt.plot(c, p_2, color=plt.cm.seismic(300)[0:3], linewidth=2.5)

plt.plot(c, p_ur, color=plt.cm.seismic(1000)[0:3], linewidth=2.5,
         label="Previously Unrewarded")

plt.axvline(0, color='k', linestyle='--')
plt.axhline(0.5, color='k', linestyle='--')

plt.xlabel('Contrast Left - Contrast Right')
plt.ylabel('P(Choice = Left)')

plt.legend()
plt.show()

#%%

c = np.linspace(-1, 1, 1000)
p_r = 1 - 1/(1+np.exp(params_value[1]*c + params_value[2]*1))
p_ur = 1 - 1/(1+np.exp(params_value[1]*c + params_value[2]*-1))

p_1 = 1 - 1/(1+np.exp(params_value[1]*c + params_value[2]*0.5))
p_2 = 1 - 1/(1+np.exp(params_value[1]*c + params_value[2]*-0.5))
p_3 = 1 - 1/(1+np.exp(params_value[1]*c + params_value[2]*0))

plt.plot(c, p_r, color=plt.cm.seismic(0)[0:3], linewidth=2.5,
         label="Previously Rewarded")

plt.plot(c, p_1, color=[0.7, 0.7, 0.7])
plt.plot(c, p_3, color=[0.7, 0.7, 0.7])
plt.plot(c, p_2, color=[0.7, 0.7, 0.7])

plt.plot(c, p_ur, color=plt.cm.seismic(1000)[0:3], linewidth=2.5,
         label="Previously Unrewarded")

plt.axvline(0, color='k', linestyle='--')
plt.axhline(0.5, color='k', linestyle='--')

plt.xlabel('Contrast Left - Contrast Right')
plt.ylabel('P(Choice = Left)')

plt.legend()
plt.show()

#%% Compare Value + No Value Estimates
alpha_opt, beta_opt = fit_model(dat, 'value')
beta_opt_no_value = fit_model(dat, 'no_value')

contrasts_ = np.array([dat["contrast_left"], dat["contrast_right"]])
responses_ = dat["response"]
idx = responses_ != 0

responses = responses_[idx]

synth_choices_value = gen_synth_responses(alpha_opt, beta_opt, dat, 'value')
synth_choices_no_value = gen_synth_responses(None, beta_opt_no_value,
                                             dat, 'no_value')

contrast_diffs = contrasts_[0, idx] - contrasts_[1, idx]
c_values = np.unique(contrast_diffs)

p_left_synthetic_value = np.full(c_values.shape, np.nan)
p_left_synthetic_no_value = np.full(c_values.shape, np.nan)

p_left_mouse = np.full(c_values.shape, np.nan)

for i, cd in enumerate(c_values):
    usethese = contrast_diffs == cd
    p_left_mouse[i] = np.mean(responses[usethese] == 1)

    p_left_synthetic_value[i] = np.mean(synth_choices_value[usethese] == 1)

    p_left_synthetic_no_value[i] = \
        np.mean(synth_choices_no_value[usethese] == 1)

plt.plot(c_values, p_left_mouse, 'ko-', label='mouse')
plt.plot(c_values, p_left_synthetic_value, 'ro-', label='fit w/ value')
plt.plot(c_values, p_left_synthetic_no_value, 'bo-', label='fit w/o value')

plt.xlabel('Contrast Left - Contrast Right')
plt.ylabel('P(Choose Left)')

plt.title('Comparing Value and No Value Estimates')
plt.legend()

plt.show()

#%% Compare across dates
ValueFitAcrossDays = np.full((len(alldat), 2), np.nan)
NoValueFitAcrossDays = np.full((len(alldat), 1), np.nan)

for day in np.arange(len(alldat)):
    ValueFitAcrossDays[day,:] = fit_model(alldat[day], 'value')
    NoValueFitAcrossDays[day] = fit_model(alldat[day], 'no_value')

names_by_session = np.array([day['mouse_name'] for day in alldat])
name_list = np.unique(names_by_session)
value_across_days = {}
no_value_across_days = {}

for i in np.arange(len(name_list)):
    usethese = names_by_session == name_list[i]

    value_across_days[str(name_list[i])] = \
        ValueFitAcrossDays[usethese,:]
    no_value_across_days[str(name_list[i])] = \
        NoValueFitAcrossDays[usethese]



#%% Cross Valiate

names_by_session = np.array([day['mouse_name'] for day in alldat])
name_list = np.unique(names_by_session)
name_list = name_list[:-1]

crossVal = {}

for name in np.arange(len(name_list)):
    crossVal[name_list[name]] = np.where(names_by_session == name_list[name])
    usethese = np.where(names_by_session == name_list[name])[0]

    # Randomly select one session to train on
    train_indx = np.random.choice(len(usethese))
    train = usethese[train_indx]

    # Test on the other sessions
    test_indx = np.delete(usethese, train_indx)

    alpha_opt, beta_opt = fit_model(alldat[train_indx], 'value')
    beta_opt_nv = fit_model(alldat[train_indx], 'no_value')

    value_cross_val = np.full(test_indx.shape, np.nan)
    no_value_cross_val = np.full(test_indx.shape, np.nan)

    for n, i in enumerate(test_indx):
        val_resp = alldat[i]["response"][alldat[i]["response"] != 0]

        val_synth_v = gen_synth_responses(alpha_opt, beta_opt,
                                          alldat[i], 'value')
        val_synth_nv = gen_synth_responses(None, beta_opt,
                                           alldat[i], 'no_value')

        value_cross_val[n] = np.mean(val_synth_v == val_resp)
        no_value_cross_val[n] = np.mean(val_synth_nv == val_resp)

    results = {'ValueFit': value_cross_val,
               'NoValueFit': no_value_cross_val,
               'TestIndx': test_indx}

    crossVal[name_list[name]] = results

for name in np.arange(len(name_list)):

    meanValue = np.mean(crossVal[name_list[name]]['ValueFit'])
    semValue = np.std(crossVal[name_list[name]]['ValueFit']) / \
               np.sqrt(len(crossVal[name_list[name]]['ValueFit']))

    meanNoValue = np.mean(crossVal[name_list[name]]['NoValueFit'])
    semNoValue = np.std(crossVal[name_list[name]]['NoValueFit']) / \
               np.sqrt(len(crossVal[name_list[name]]['NoValueFit']))

    if name != len(name_list)-1:
        label1 = None
        label2 = None
    else:
        label1 = 'Fit With Value'
        label2 = 'Fit Without Value'

    plt.bar(x=name-0.1, height=meanValue, width=0.2, yerr=semValue,
            color='goldenrod', label=label1)
    plt.bar(x=name+0.1, height=meanNoValue, width=0.2, yerr=semNoValue,
            color='steelblue', label=label2)

plt.ylim((0, 1.01))
plt.legend(loc='best')
plt.show()

#%%

beta_opt = fit_model(alldat[11], 'no_value')

l_ = alldat[11]["contrast_left"]
r_ = alldat[11]["contrast_right"]
response_trials = alldat[11]["response"] != 0

l = l_[response_trials]
r = r_[response_trials]
responses = alldat[11]["response"][response_trials]

vals = np.unique(l-r)

pl = 1 - 1/(1 + np.exp(beta_opt*vals)) + np.finfo(float).eps

plot_psychometric(l-r, responses, 'Mouse Choice')

plt.plot(vals, pl, 'o-', label='Model Fit')

plt.legend()

plt.show()

#%%
