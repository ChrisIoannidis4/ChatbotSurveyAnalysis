import pandas as pd
from scipy.stats import mannwhitneyu #  Mann-Whitney U test 
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import math 
responses = pd.read_excel("responses/all_results_preprocessed.xlsx")
print(responses)


# 1. Stat Analysis

# Experiment 1: Mann-Whitney U test 
satisfaction_no_tts = responses[responses['text_to_speech'] == 0]['user_satisfaction']
satisfaction_with_tts = responses[responses['text_to_speech'] == 1]['user_satisfaction']
mann_whitney_result = mannwhitneyu(satisfaction_no_tts, satisfaction_with_tts, alternative='greater')
print("Mann-Whitney U Test Results")
print("U-statistic:", mann_whitney_result.statistic)  #53.0
print("p-value:", mann_whitney_result.pvalue) #0.0009734801158020909
n1 = len(satisfaction_no_tts)
n2 = len(satisfaction_with_tts)

# Calculate rank-biserial correlation
rank_biserial_r = 1 - (2 * mann_whitney_result.statistic) / (n1 * n2)
print("Rank-Biserial Correlation (effect size):", rank_biserial_r)

# Experiment 2: Logistic Regression - binary satisfaction
X = responses[['text_to_speech']] 
y = (responses['user_satisfaction'] > 3).astype(int)  
logit_model = sm.Logit(y, X)
result = logit_model.fit_regularized(alpha=0.1, L1_wt=0)
print(result.summary())
odds_ratios = np.exp(result.params)   
print("Odds Ratios:", odds_ratios)

# 95% confidence intervals
conf = result.conf_int()  # Get confidence intervals of coefficients
conf_exp = np.exp(conf)  # Transform to odds ratio scale
conf_exp.columns = ['95% CI Lower', '95% CI Upper']
conf_exp['Odds Ratio'] = odds_ratios

print(conf_exp)


# Experiment 3:     

X = responses[['text_to_speech']]
X = sm.add_constant(X)  
y = responses['prefer_other'] 
logit_model = sm.Logit(y, X)
result = logit_model.fit_regularized(alpha=0.1, L1_wt=0)
print(result.summary())
odds_ratios = np.exp(result.params)   
print("Odds Ratios:", odds_ratios)

# 95% confidence intervals
conf = result.conf_int()  # Get confidence intervals of coefficients
conf_exp = np.exp(conf)  # Transform to odds ratio scale
conf_exp.columns = ['95% CI Lower', '95% CI Upper']
conf_exp['Odds Ratio'] = odds_ratios

print(conf_exp)



# Experiment 4: Confounding factors 

X = sm.add_constant(responses[['text_to_speech', 'humanlike']])
y = (responses['user_satisfaction'] > 3).astype(int)
logit_model = sm.Logit(y, X)
result = logit_model.fit_regularized(alpha=0.1, L1_wt=0)
print(result.summary())
odds_ratios = np.exp(result.params)   
print("Odds Ratios:", odds_ratios)
conf = np.exp(result.conf_int())
conf['OR'] = odds_ratios

# 95% confidence intervals
conf = result.conf_int()  # Get confidence intervals of coefficients
conf_exp = np.exp(conf)  # Transform to odds ratio scale
conf_exp.columns = ['95% CI Lower', '95% CI Upper']
conf_exp['Odds Ratio'] = odds_ratios

print(conf_exp)

plt.figure(figsize=(8, 6))
plt.errorbar(conf_exp['Odds Ratio'], conf_exp.index, 
             xerr=[conf_exp['Odds Ratio'] - conf_exp['95% CI Lower'], conf_exp['95% CI Upper'] - conf_exp['Odds Ratio']],
             fmt='o', color='black', ecolor='gray', capsize=3)
plt.axvline(x=1, color='gray', linestyle='--')  # Line at OR=1 for reference
plt.xlabel("Odds Ratio (OR)")
plt.title("Odds Ratios with 95% CIs for Satisfaction Predictors")
plt.show()

# Experiment 5: 

# Separate humanlike scores by text_to_speech groups
humanlike_no_tts = responses[responses['text_to_speech'] == 0]['humanlike']
humanlike_with_tts = responses[responses['text_to_speech'] == 1]['humanlike']
humanlike_test = mannwhitneyu(humanlike_no_tts, humanlike_with_tts, alternative='two-sided')
print("Mann-Whitney U Test Results")
print("U-statistic:", humanlike_test.statistic)  #53.0
print("p-value:", humanlike_test.pvalue) #0.0009734801158020909
print("Humanlike Behavior Test Results:", humanlike_test)
n1 = len(humanlike_no_tts)
n2 = len(humanlike_with_tts)

# Calculate rank-biserial correlation
rank_biserial_r = 1 - (2 * humanlike_test.statistic) / (n1 * n2)
print("Rank-Biserial Correlation (effect size):", rank_biserial_r)



# Separate waiting time scores by text_to_speech groups
waiting_no_tts = responses[responses['text_to_speech'] == 0]['waiting_time'] 
waiting_with_tts = responses[responses['text_to_speech'] == 1]['waiting_time']
waiting_time_test = mannwhitneyu(waiting_no_tts, waiting_with_tts, alternative='two-sided')
print("Mann-Whitney U Test Results")
print("U-statistic:", humanlike_test.statistic)  #53.0
print("p-value:", humanlike_test.pvalue) #0.0009734801158020909
print("Waiting TIme Behavior Test Results:", humanlike_test)
n1 = len(humanlike_no_tts)
n2 = len(humanlike_with_tts)

# Calculate rank-biserial correlation
rank_biserial_r = 1 - (2 * waiting_time_test.statistic) / (n1 * n2)
print("Rank-Biserial Correlation (effect size):", rank_biserial_r)




# 2. Visualizations
# descriptive table
descriptive_stats = responses.groupby('text_to_speech').agg(
    mean_satisfaction=('user_satisfaction', 'mean'),
    sd_satisfaction=('user_satisfaction', 'std'),
    median_satisfaction=('user_satisfaction', 'median'),
    iqr_satisfaction=('user_satisfaction', lambda x: x.quantile(0.75) - x.quantile(0.25)),
    mean_humanlike=('humanlike', 'mean'),
    sd_humanlike=('humanlike', 'std'),
    mean_waiting_time=('waiting_time', 'mean'),
    sd_waiting_time=('waiting_time', 'std')
)

print("Descriptive Statistics Table")
print(descriptive_stats)

#violin plot

# Violin plot for user satisfaction by text-to-speech group
plt.figure(figsize=(8, 6))
sns.violinplot(x='text_to_speech', y='user_satisfaction', data=responses, palette='Set2')
plt.xlabel("Text-to-Speech")
plt.ylabel("User Satisfaction")
plt.title("Distribution of User Satisfaction by Text-to-Speech Group")
plt.xticks([0, 1], ['No TTS', 'TTS'])
plt.show()


#barplot
# import numpy as np
# Calculate means and standard errors for humanlike and waiting_time
grouped_stats = responses.groupby('text_to_speech').agg(
    mean_humanlike=('humanlike', 'mean'),
    se_humanlike=('humanlike', lambda x: x.std() / math.sqrt(len(x))),
    mean_waiting_time=('waiting_time', 'mean'),
    se_waiting_time=('waiting_time', lambda x: x.std() / math.sqrt(len(x)))
).reset_index()

# Adding 95% CI columns
grouped_stats['ci_lower_humanlike'] = grouped_stats['mean_humanlike'] - 1.96 * grouped_stats['se_humanlike']
grouped_stats['ci_upper_humanlike'] = grouped_stats['mean_humanlike'] + 1.96 * grouped_stats['se_humanlike']
grouped_stats['ci_lower_waiting_time'] = grouped_stats['mean_waiting_time'] - 1.96 * grouped_stats['se_waiting_time']
grouped_stats['ci_upper_waiting_time'] = grouped_stats['mean_waiting_time'] + 1.96 * grouped_stats['se_waiting_time']

print(grouped_stats)

# Bar plot with error bars
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Humanlike
ax[0].bar([0, 1], grouped_stats['mean_humanlike'], yerr=grouped_stats['se_humanlike'], capsize=5, color=['#66c2a5', '#fc8d62'])
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(['No TTS', 'TTS'])
ax[0].set_ylabel('Mean Humanlike Rating')
ax[0].set_title('Humanlike Ratings by Text-to-Speech Group')

# Waiting Time
ax[1].bar([0, 1], grouped_stats['mean_waiting_time'], yerr=grouped_stats['se_waiting_time'], capsize=5, color=['#66c2a5', '#fc8d62'])
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(['No TTS', 'TTS'])
ax[1].set_ylabel('Mean Waiting Time')
ax[1].set_title('Waiting Time by Text-to-Speech Group')

plt.tight_layout()
plt.show()

