import numpy as np
import pandas as pd
from scipy.stats import norm


#------------------Simulating Click Data for A/B Testing------------------
N_exp = 10000
N_con = 10000

# Generating Click Data
click_exp = pd.Series(np.random.binomial(1, 0.5, size=N_exp))
click_con = pd.Series(np.random.binomial(1, 0.2, size=N_con))

# Generate Group Identifier
exp_id = pd.Series(np.repeat("exp", N_exp))
con_id = pd.Series(np.repeat("con", N_con))

df_exp = pd.concat([click_exp, exp_id], axis=1)
df_con = pd.concat([click_con, con_id], axis=1)

df_exp.columns = ["click", "group"]
df_con.columns = ["click", "group"]

df_ab_test = pd.concat([df_exp, df_con], axis=0).reset_index(drop=True)
print(df_ab_test)

X_con = df_ab_test.groupby("group")["click"].sum().loc["con"]
X_exp = df_ab_test.groupby("group")["click"].sum().loc["exp"]

print(f'Number of Clicks in Control: {X_con}')
print(f'Number of Clicks in Experimental: {X_exp}')


p_con_hat = X_con / N_con
p_exp_hat = X_exp / N_exp
print(f'Click Probability in Control Group: {p_con_hat}')
print(f'Click Probability in Experimental Group: {p_exp_hat}')

p_pooled_hat = (X_con + X_exp) / (N_con + N_exp)
pooled_variance = p_pooled_hat * (1 - p_pooled_hat) * (1/N_con + 1/N_exp)
print(f'p^_pooled is: {p_pooled_hat}')
print(f'pooled_variance is: {pooled_variance}')

SE = np.sqrt(pooled_variance)
print(f'Standard Error is: {SE}')

Test_stat = (p_con_hat - p_exp_hat)/SE
print(f'Test Statistics for 2-sample Z-test is: {Test_stat}')

alpha = 0.05
print(f'Alpha: significance level is {alpha}')


Z_crit = norm.ppf(1-alpha/2)
print(f'Z-critical value from Standard Normal distribution: {Z_crit}')

p_value = 2 * norm.sf(abs(Test_stat))
print(f'P-value of the 2-sample Z-test: {p_value:.3f}')

CI = [round((p_exp_hat - p_con_hat) - SE*Z_crit, 3), round((p_exp_hat - p_con_hat) + SE*Z_crit, 3)]
print(f'Confidence Interval of the 2 sample Z-test is: {CI}')

delta = 0.03
