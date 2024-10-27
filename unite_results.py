import pandas as pd



#####################################
# voice chat disabled - group a 
 
dfa = pd.read_excel("responses/group_a.xlsx" )

dfa.rename(columns={dfa.columns[0]: "timestamp"}, inplace=True)
dfa.rename(columns={dfa.columns[1]: "gender"}, inplace=True)
dfa.rename(columns={dfa.columns[2]: "age"}, inplace=True)
dfa.rename(columns={dfa.columns[3]: "prev_experience"}, inplace=True)
dfa.rename(columns={dfa.columns[4]: "user_satisfaction"}, inplace=True)
dfa.rename(columns={dfa.columns[5]: "waiting_time"}, inplace=True)
dfa.rename(columns={dfa.columns[6]: "prefer_other"}, inplace=True)
dfa.rename(columns={dfa.columns[7]: "straightforward"}, inplace=True)
dfa.rename(columns={dfa.columns[8]: "humanlike"}, inplace=True)
dfa = dfa.assign(text_to_speech = [ 0 for i in range(dfa.shape[0])])


####################################
#voice chat enabled - group b


dfb = pd.read_excel("responses/group_b.xlsx" )

dfb.rename(columns={dfb.columns[0]: "timestamp"}, inplace=True)
dfb.rename(columns={dfb.columns[1]: "gender"}, inplace=True)
dfb.rename(columns={dfb.columns[2]: "age"}, inplace=True)
dfb.rename(columns={dfb.columns[3]: "prev_experience"}, inplace=True)
dfb.rename(columns={dfb.columns[4]: "user_satisfaction"}, inplace=True)
dfb.rename(columns={dfb.columns[5]: "voice_satisfaction"}, inplace=True)
dfb.rename(columns={dfb.columns[6]: "voice_comment"}, inplace=True)
dfb.rename(columns={dfb.columns[7]: "waiting_time"}, inplace=True)
dfb.rename(columns={dfb.columns[8]: "prefer_other"}, inplace=True)
dfb.rename(columns={dfb.columns[9]: "straightforward"}, inplace=True)
dfb.rename(columns={dfb.columns[10]: "humanlike"}, inplace=True)
dfb = dfb.assign(text_to_speech = [1  for i in range(dfb.shape[0])])


###############################
#union dataframes 

columns = sorted(set(dfa.columns).union(set(dfb.columns)))
dfa = dfa.reindex(columns=columns)
dfb = dfb.reindex(columns=columns)
 
df = pd.concat([dfa, dfb], ignore_index=True)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    print(df)

df.to_excel("responses/all_results.xlsx", index=False)

df['prefer_other'] = df['prefer_other'].map({'No': 0, 'Yes': 1})
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
df['straightforward'] = df['straightforward'].map({'No': 0, 'Yes': 1})

df.to_excel("responses/all_results_preprocessed.xlsx", index=False)
