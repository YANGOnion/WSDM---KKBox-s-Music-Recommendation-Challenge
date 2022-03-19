
import pandas as pd
import numpy as np
import gc
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#----------------------------------- feature engineering ----------------------------------#

def get_id_profile(df, item, fea_con_profile, fea_cat_profile):
    df_profile = pd.DataFrame({item: df[item], item+'_count': df.groupby(item)[item].transform('count')})
    for fea in fea_con_profile:
        df_profile[item+'_'+fea+'_mean'] = df.groupby(item)[fea].transform('mean')
        df_profile[item+'_'+fea+'_std'] = df.groupby(item)[fea].transform('std')
    for fea in fea_cat_profile:
        df_profile[item+'_'+fea+'_ratio'] = df.groupby([item, fea])[fea].transform('count')/df_profile[item+'_count']
    for col in df_profile.columns:
        if col!=item:
            df_profile[col].fillna(0, inplace=True)
            df_profile[col] = df_profile[col].astype('float32')
    return df_profile


def create_feature(df):
    df['timestamp'] = pd.Series(range(len(df))).astype('float32')
    
    item = 'msno'
    fea_con_profile = ['song_length']
    fea_cat_profile = ['genre_ids', 'language', 'registered_via', 'source_screen_name', 'source_system_tab', 'source_type']
    df_profile_msno = get_id_profile(df, item, fea_con_profile, fea_cat_profile)
    item = 'song_id'
    fea_con_profile = ['bd', 'expiration_date', 'register_period', 'registration_init_time']
    fea_cat_profile = ['city', 'gender', 'registered_via', 'source_screen_name', 'source_system_tab', 'source_type']
    df_profile_song_id = get_id_profile(df, item, fea_con_profile, fea_cat_profile)

    df = pd.concat((df, df_profile_msno.loc[:,df_profile_msno.columns!='msno']), axis=1)
    df = pd.concat((df, df_profile_song_id.loc[:,df_profile_song_id.columns!='song_id']), axis=1)
    return df

#----------------------------------- embbeding mapper ----------------------------------#

class WildDict:
    '''Dictionary with wild key
    '''
    
    def __init__(self):
        self.d = {}
        self.wild_value = None
    
    def set_map(self, key, value):
        self.d[key] = value
    
    def set_wild(self, wild_value):
        '''match unknown key with a wild value
        '''
        self.wild_value = wild_value
    
    def get_map(self, key):
        if key in self.d:
            return self.d[key]
        else:
            return self.wild_value

class Mapper:
    '''Mapping feature fields to be ordinal
    '''
    
    def __init__(self, df, fea_con, thres):
        self.n_fields = len(df.columns)
        self.mapper = {}
        self.fea_con = fea_con
        offset = 0
        for col in df.columns:
            if col in fea_con:
                self.mapper[col] = offset
                offset += 1
            else:
                counts = df.loc[:, col].value_counts()
                values = sorted(counts[counts>=thres].index)
                new_dict = WildDict()
                for j, elem in enumerate(values, offset):
                    new_dict.set_map(elem, j)
                new_dict.set_wild(j+1)
                self.mapper[col] = new_dict
                offset = j+2
        self.n_features = offset
    
    def _get_col_index(self, col, col_values):
        if col in self.fea_con:
            return pd.Series(np.repeat(self.mapper[col], len(col_values)))
        else:
            return col_values.apply(self.mapper[col].get_map)
    
    def _get_col_value(self, col, col_values):
        if col in self.fea_con:
            return col_values
        else:
            return pd.Series(np.ones(len(col_values)))
    
    def get_index(self, new_df, fea_con):
        X_index = pd.concat([self._get_col_index(col, new_df.loc[:, col]) for col in new_df.columns], axis=1).values
        return X_index.astype('int32')
    
    def get_value(self, new_df, fea_con):
        X_value = pd.concat([self._get_col_value(col, new_df.loc[:, col]) for col in new_df.columns], axis=1).values
        return X_value.astype('float32')
    
#----------------------------------- create data loader ----------------------------------#

def data_preprocess(df, rare_thres, valid_size, y_name):
    '''create training, validating & testing tensors
    '''
    
    ### embed mapping
    fea_con = np.setdiff1d(df.columns[df.dtypes=='float32'].values, [y_name])
    mapper = Mapper(df.drop(y_name, axis=1), fea_con, rare_thres)
    X_index = mapper.get_index(df.drop(y_name, axis=1), fea_con)
    X_value = mapper.get_value(df.drop(y_name, axis=1), fea_con)
    y_data = df[y_name].values.reshape(-1, 1)
    
    ### spliting
    X_test_index = X_index[pd.isnull(df[y_name])]
    X_test_value = X_value[pd.isnull(df[y_name])]
    X_index = X_index[~pd.isnull(df[y_name])]
    X_value = X_value[~pd.isnull(df[y_name])]
    y_data = y_data[~pd.isnull(df[y_name])]
    X_train_index, X_valid_index, y_train, y_valid = train_test_split(X_index, y_data, 
                                                          test_size=valid_size, 
                                                          shuffle=False)
    X_train_value, X_valid_value, y_train, y_valid = train_test_split(X_value, y_data, 
                                                          test_size=valid_size, 
                                                          shuffle=False)
    del X_index, X_value, y_data
    gc.collect()
    
    ### standardization
    ind_con = np.where(df.drop(y_name, axis=1).dtypes=='float32')[0]
    ind_cat = np.setdiff1d(np.array(range(len(df.columns)-1)), ind_con)
    enc = StandardScaler()
    enc.fit(X_train_value[:, ind_con])
    X_train_value = np.concatenate((X_train_value[:, ind_cat], 
                                    enc.transform(X_train_value[:, ind_con])), axis=1)
    X_valid_value = np.concatenate((X_valid_value[:, ind_cat], 
                                    enc.transform(X_valid_value[:, ind_con])), axis=1)
    X_test_value = np.concatenate((X_test_value[:, ind_cat], 
                                   enc.transform(X_test_value[:, ind_con])), axis=1)
    X_train_index = X_train_index[:, np.concatenate((ind_cat, ind_con))]
    X_valid_index = X_valid_index[:, np.concatenate((ind_cat, ind_con))]
    X_test_index = X_test_index[:, np.concatenate((ind_cat, ind_con))]
    
    ### to tensor
    X_train_index = torch.tensor(X_train_index).type(torch.int)
    X_train_value = torch.tensor(X_train_value).type(torch.float32)
    X_valid_index = torch.tensor(X_valid_index).type(torch.int)
    X_valid_value = torch.tensor(X_valid_value).type(torch.float32)
    X_test_index = torch.tensor(X_test_index).type(torch.int)
    X_test_value = torch.tensor(X_test_value).type(torch.float32)
    y_train = torch.tensor(y_train).type(torch.FloatTensor)
    y_valid = torch.tensor(y_valid).type(torch.FloatTensor)
    return mapper.n_fields, mapper.n_features, \
        X_train_index, X_train_value, y_train, \
            X_valid_index, X_valid_value, y_valid, \
                X_test_index, X_test_value

