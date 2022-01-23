from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns

DATA_PATH = '<my data path>'

df_train = pd.read_csv(Path(DATA_PATH)/'train.csv')
df_test = pd.read_csv(Path(DATA_PATH)/'test.csv')
df_members = pd.read_csv(Path(DATA_PATH)/'members.csv')
df_songs = pd.read_csv(Path(DATA_PATH)/'songs.csv')
df_songs_extra = pd.read_csv(Path(DATA_PATH)/'song_extra_info.csv')

#---------------------- Sample distributions of the training set and the testing set --------------#

### new users & songs in the tesing set
len(df_test[~df_test.msno.isin(df_train.msno)])/len(df_test) # 7%
len(df_test[~df_test.song_id.isin(df_train.song_id)])/len(df_test) # 13%

### frequency distribtuion of users in the training set versus tesing set
df = pd.merge((df_train['msno'].value_counts()/len(df_train)).to_frame().reset_index(),
         (df_test['msno'].value_counts()/len(df_test)).to_frame().reset_index(),
         on='index', how='inner')
p = sns.relplot(data=df, x='msno_x', y='msno_y')
p.set(xlabel = "msno frequency in training set", ylabel = "msno frequency in testing set")
p.figure.set_size_inches(7, 7)
np.corrcoef(df.msno_x,df.msno_y) # 0.68

### frequency distribtuion of songs in the training set versus tesing set
df = pd.merge((df_train['song_id'].value_counts()/len(df_train)).to_frame().reset_index(),
         (df_test['song_id'].value_counts()/len(df_test)).to_frame().reset_index(),
         on='index', how='inner')
p = sns.relplot(data=df, x='song_id_x', y='song_id_y')
p.set(xlabel = "song_id frequency in training set", ylabel = "song_id frequency in testing set")
p.figure.set_size_inches(7, 7)
np.corrcoef(df.song_id_x,df.song_id_y) # 0.79

#---------------------------- Temporal patterns of the data -----------------------------#

### time patterns
sample_song = df_train['song_id'].value_counts().index[0]
df_train[df_train.song_id==sample_song]
window = int(1e5)
start = int(0)
listen_ratio = []
while start<len(df_train):
    df = df_train.iloc[start:(start+window),:]
    df = df[df.song_id==sample_song]
    df = df.target.value_counts()
    if len(df)>0:
        listen_ratio.append(df[df.index==1].values[0]/sum(df))
    else:
        listen_ratio.append(np.nan)
    start += window
p = sns.relplot(data=pd.DataFrame({'time':range(len(listen_ratio)), 'ratio':listen_ratio}),
            x='time', y='ratio', kind='line') # decreasing popularity
p.set(xlabel = "timestamp of a 1e5 record window", ylabel = "frequency of target=1")


