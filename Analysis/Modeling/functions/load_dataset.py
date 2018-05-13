def load_summary():
    ## %matplotlib inline

    import scipy.io as io
    import numpy as np
    data=io.loadmat('dataset_raw_sumMAT.mat' )
    ds = data['ds']
    val=ds[0,0]
    #Size of Feature vector
    sz=8

    # X -> features, y -> label
    vpn =np.zeros((ds.shape[1]))
    y =np.zeros((ds.shape[1]))
    X =np.zeros((ds.shape[1],sz))

    for i in range(0,len(X[:,0]-1)):
        tmp=ds['mean'][0,i]
        X[i, 0]=tmp['xr']
        X[i, 1] = tmp['yr']
        X[i, 2] = tmp['dur']
        X[i, 3] = tmp['pupil']
        X[i, 4] = ds['numberOfFix'][0,i][0][0].astype('int')
        X[i, 5] = ds['numberOfBlink'][0,i][0][0].astype('int')
        X[i, 6]=tmp['xl']
        X[i, 7] = tmp['yl']
        y[i] = ds['nCat'][0,i][0,0].astype(int)
        vpn[i] = ds['nVpn'][0, i][0, 0].astype(int)

    #split datasets
    perception=np.where(ds['condition'] == 'perception')[1]
    imagination=np.where(ds['condition'] == 'imagination')[1]

    y_p=y[perception]
    y_i=y[imagination]
    X_p=X[perception]
    X_i=X[imagination]
    vpn_p =vpn[perception]
    vpn_i =vpn[imagination]

    #find nan and exclude
    not_excld = np.logical_not((np.any(np.isnan(np.concatenate((X_p,X_i),axis=1)),axis=1)))
    y_p=y_p[not_excld]
    y_i=y_i[not_excld]
    X_p=X_p[not_excld]
    X_i=X_i[not_excld]
    vpn_p =vpn_p[not_excld]
    vpn_i =vpn_i[not_excld]

    return X_p,y_p,X_i,y_i,vpn_p,vpn_i