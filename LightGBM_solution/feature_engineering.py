import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

def feature_generating(df):
    '''Generating features: a time feature, user & song profile features, and SVD features
    Args:
        df: a pandas.DataFrame that has been preprocessed
    Return:
        a pandas.DataFrame of processed data
    '''
    ### timestamp feature
    df['timestamp'] = pd.Series(range(len(df))).astype('float32')
    
    ### get profile features
    def get_id_profile(item, fea_con_profile, fea_cat_profile):
        '''Create profile features for members or songs
        Args:
            item: 'msno' or 'song_id'
            fea_con_profile: a list of continuous variables
            fea_cat_profile: a list of categorical variables
        '''
        df_profile = pd.DataFrame({item: df[item], item+'_count': df.groupby(item)[item].transform('count')})
        for fea in fea_con_profile:
            df_profile[item+'_'+fea+'_mean'] = df.groupby(item)[fea].transform('mean')
            df_profile[item+'_'+fea+'_std'] = df.groupby(item)[fea].transform('std')
        for fea in fea_cat_profile:
            df_profile[item+'_'+fea+'_ratio'] = df.groupby([item, fea])[fea].transform('count')/df_profile[item+'_count']
        for col in df_profile.columns:
            if col!=item:
                df_profile[col] = df_profile[col].astype('float32')
        return df_profile

    # user profile
    item = 'msno'
    fea_con_profile = ['song_length']
    fea_cat_profile = ['genre_ids', 'language', 'registered_via', 'source_screen_name', 'source_system_tab', 'source_type']
    df_profile_msno = get_id_profile(item, fea_con_profile, fea_cat_profile)

    # song profile
    item = 'song_id'
    fea_con_profile = ['bd', 'expiration_date', 'register_period', 'registration_init_time']
    fea_cat_profile = ['city', 'gender', 'registered_via', 'source_screen_name', 'source_system_tab', 'source_type']
    df_profile_song_id = get_id_profile(item, fea_con_profile, fea_cat_profile)
    
    ### get latent features
    # counts concurrence
    df_latent = df[['msno', 'song_id', 'artist_name']]
    df_latent['times_song_id'] = df_latent.groupby(['msno', 'song_id'])['msno'].transform('count')
    df_latent['times_artist_name'] = df_latent.groupby(['msno', 'artist_name'])['msno'].transform('count')

    # map index
    all_msno = sorted(df_latent['msno'].unique())
    all_msno = dict(zip(all_msno, range(len(all_msno))))
    all_song_id = sorted(df_latent['song_id'].unique())
    all_song_id = dict(zip(all_song_id, range(len(all_song_id))))
    all_artist_name = sorted(df_latent['artist_name'].unique())
    all_artist_name = dict(zip(all_artist_name, range(len(all_artist_name))))
    df_latent['ind_msno'] = df_latent['msno'].map(all_msno)
    df_latent['ind_song_id'] = df_latent['song_id'].map(all_song_id)
    df_latent['ind_artist_name'] = df_latent['artist_name'].map(all_artist_name)

    # SVD for msno~song_id
    mat_song_id = csr_matrix((df_latent['times_song_id'], (df_latent['ind_msno'], df_latent['ind_song_id'])), 
                              shape=(len(all_msno), len(all_song_id)))
    svd_song_id = TruncatedSVD(n_components=50, random_state=0)
    svd_song_id.fit(mat_song_id)
    svd_song_id.explained_variance_ratio_.sum() # 0.20
    latent_msno1 = pd.DataFrame(svd_song_id.transform(mat_song_id))
    latent_msno1.columns = ['svd_msno1_'+str(i) for i in range(latent_msno1.shape[1])]
    latent_msno1 = pd.concat((pd.DataFrame({'msno': all_msno.keys()}), latent_msno1), axis=1)
    latent_msno1['msno'] = latent_msno1['msno'].astype('category')
    latent_msno1.iloc[:, 1:] = latent_msno1.iloc[:, 1:].astype('float32')
    latent_song_id = pd.DataFrame(svd_song_id.components_.T)
    latent_song_id.columns = ['svd_song_id_'+str(i) for i in range(latent_song_id.shape[1])]
    latent_song_id = pd.concat((pd.DataFrame({'song_id': all_song_id.keys()}), latent_song_id), axis=1)
    latent_song_id['song_id'] = latent_song_id['song_id'].astype('category')
    latent_song_id.iloc[:, 1:] = latent_song_id.iloc[:, 1:].astype('float32')

    # SVD for msno~artist_name
    mat_artist_name = csr_matrix((df_latent['times_artist_name'], (df_latent['ind_msno'], df_latent['ind_artist_name'])), 
                                  shape=(len(all_msno), len(all_artist_name)))
    svd_artist_name = TruncatedSVD(n_components=25, random_state=0)
    svd_artist_name.fit(mat_artist_name)
    svd_artist_name.explained_variance_ratio_.sum() # 0.83
    latent_msno2 = pd.DataFrame(svd_artist_name.transform(mat_artist_name))
    latent_msno2.columns = ['svd_msno2_'+str(i) for i in range(latent_msno2.shape[1])]
    latent_msno2 = pd.concat((pd.DataFrame({'msno': all_msno.keys()}), latent_msno2), axis=1)
    latent_msno2['msno'] = latent_msno2['msno'].astype('category')
    latent_msno2.iloc[:, 1:] = latent_msno2.iloc[:, 1:].astype('float32')
    latent_artist_name = pd.DataFrame(svd_artist_name.components_.T)
    latent_artist_name.columns = ['svd_artist_name_'+str(i) for i in range(latent_artist_name.shape[1])]
    latent_artist_name = pd.concat((pd.DataFrame({'artist_name': all_artist_name.keys()}), latent_artist_name), axis=1)
    latent_artist_name['artist_name'] = latent_artist_name['artist_name'].astype('category')
    latent_artist_name.iloc[:, 1:] = latent_artist_name.iloc[:, 1:].astype('float32')

    ### merge all features
    df = pd.concat((df, df_profile_msno.loc[:,df_profile_msno.columns!='msno']), axis=1)
    df = pd.concat((df, df_profile_song_id.loc[:,df_profile_song_id.columns!='song_id']), axis=1)
    df = df.merge(latent_msno1, how='left', on='msno')
    df = df.merge(latent_song_id, how='left', on='song_id')
    df = df.merge(latent_msno2, how='left', on='msno')
    df = df.merge(latent_artist_name, how='left', on='artist_name')
    
    return df