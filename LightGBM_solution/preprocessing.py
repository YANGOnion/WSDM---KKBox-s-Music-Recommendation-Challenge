from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from feature_engine.encoding import RareLabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

def cat_convertor(x, to_type='category'):
    '''Convert object column to category column
    '''
    x = x.astype(str)
    x[x=='nan'] = np.nan
    return x.astype(to_type)

def data_preprocess(data_path, use_test=False):
    '''Data preprocessing: imputing, rare-label encoding, and category encoding
    Args:
        data_path: path for dta files
        use_test: whether to combine test data for preprocessing
    Return:
        a pandas.DataFrame of processed data
    '''
    ### read data
    df = pd.read_csv(Path(data_path)/'train.csv')
    if use_test:
        df_test = pd.read_csv(Path(data_path)/'test.csv')
        df = pd.concat((df, df_test.drop('id', axis=1)), axis=0).reset_index(drop=True)
    df_members = pd.read_csv(Path(data_path)/'members.csv')
    df_songs = pd.read_csv(Path(data_path)/'songs.csv')
    
    ### preprocess member data
    df_members[['city', 'registered_via']] = df_members[['city', 'registered_via']].apply(cat_convertor)
    df_members.loc[(df_members['bd']<15) | (df_members['bd']>80), 'bd'] = np.nan
    df_members[['registration_init_time', 'expiration_date']] = \
        df_members[['registration_init_time', 'expiration_date']].\
            apply(lambda x: pd.to_datetime(x, format='%Y%m%d'))
    date_min = df_members['registration_init_time'].min()
    df_members[['registration_init_time', 'expiration_date']] = \
        df_members[['registration_init_time', 'expiration_date']].\
            apply(lambda x: (x-date_min).astype('timedelta64[D]').astype(int))
    df_members['register_period'] = df_members['expiration_date'] - df_members['registration_init_time']

    ### preprocess song data
    df_songs[['language']] = df_songs[['language']].apply(cat_convertor)
    df_songs['genre_ids'] = df_songs['genre_ids'].str.extract('^([^\\|]+)\\|*?')
    df_songs['artist_name'] = df_songs['artist_name'].str.extract('^([^\\|]+)\\|*?')

    ### merge member & song data into training data
    df = df.merge(df_members, how='left', on='msno')
    df = df.merge(df_songs[['song_id', 'artist_name', 'song_length', 'genre_ids', 'language']], 
                            how='left', on='song_id')
    df['artist_name'] = df['artist_name'].fillna('others')
    for col in df.columns:
        if df[col].dtypes=='object':
            df[col] = df[col].astype('category')
        elif df[col].dtypes=='int64':
            df[col] = df[col].astype('int32')
        elif df[col].dtypes=='float64':
            df[col] = df[col].astype('float32')

    ### preprocessing: fillna & encoding
    fea_cat = np.setdiff1d(df.dtypes[df.dtypes == 'category'].index, ['msno', 'song_id', 'artist_name'])
    fea_con = np.setdiff1d(df.dtypes[df.dtypes != 'category'].index, ['target'])
    enc = ColumnTransformer([
        ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='others'),
                              RareLabelEncoder(tol=1e-4, n_categories=30),
                              OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=999)), fea_cat),
        ('con', SimpleImputer(strategy='mean'), fea_con)
    ])
    data = enc.fit_transform(df)
    df = pd.concat((df[['msno', 'song_id', 'artist_name', 'target']], pd.DataFrame(data)), axis=1)
    df.columns = np.concatenate((['msno', 'song_id', 'artist_name', 'target'], fea_cat, fea_con))
    df[fea_cat] = df[fea_cat].astype(int).astype('category')
    df[fea_con] = df[fea_con].astype('float32')
    
    return df