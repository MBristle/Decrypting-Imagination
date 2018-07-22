def load_summary(y_cat='nCat', group_cat='nVpn'):
    ## %matplotlib inline

    import scipy.io as io
    import numpy as np
    data = io.loadmat('dataset_raw_sumMAT.mat')
    ds = data['ds']

    # Size of Feature vector
    sz = 8

    # X -> features, y -> label
    vpn = np.zeros((ds.shape[1]))
    condition = []
    y = np.zeros((ds.shape[1]))
    X = np.zeros((ds.shape[1], sz))

    for i in range(0, len(X[:, 0] - 1)):
        tmp = ds['mean'][0, i]
        X[i, 0] = tmp['xr']
        X[i, 1] = tmp['yr']
        X[i, 2] = tmp['dur']
        X[i, 3] = 0#tmp['pupil']
        X[i, 4] = ds['numberOfFix'][0, i][0][0].astype('int')
        X[i, 5] = ds['numberOfBlink'][0, i][0][0].astype('int')
        X[i, 6] = tmp['xl']
        X[i, 7] = tmp['yl']
        y[i] = ds[y_cat][0, i][0, 0].astype(int)
        vpn[i] = ds[group_cat][0, i][0, 0].astype(int)
        condition.append(str(ds['condition'][0, i][0]))

    X_i, X_p, vpn_i, vpn_p, y_i, y_p = split_and_rm_nan(X, y, ds, vpn,np.asarray(condition))

    return X_p, y_p, X_i, y_i, vpn_p, vpn_i


def load_map(split=6, y_cat='nCat', group_cat='nVpn', load=False):
    import os.path as path
    import numpy as np
    FILENAME = 'feature_map' + '_' + group_cat + '_' + y_cat + str(split) + '.npz'
    # import Data in features X and targets y if available
    if load and path.isfile(FILENAME):
        data = np.load(FILENAME)  # load_map()
        X_p = data['X_p']
        y_p = data['y_p']
        X_i = data['X_i']
        y_i = data['y_i']
        group_p = data['group_p']
        group_i = data['group_i']
        return X_p, y_p, X_i, y_i, group_p, group_i

    # Calculate Dataset

    ## %matplotlib inline
    from functools import reduce
    import scipy.io as io
    import numpy as np
    import pandas as pd
    data = io.loadmat('dataset_raw_sumMAT.mat')

    ds = data['ds']



    # Dataset should be splited into imagination and Perception
    # Only Trails on Screen should be considered
    # split the bins by partition and bcond

    f_shape = (len(ds[0, :]),) + (split + 1, split + 1, 8)


    # X -> features, y -> label

    group = np.empty((ds.shape[1]))
    condition = []
    y = np.empty((ds.shape[1]))
    y[:] = np.nan
    X = np.empty(shape=f_shape)

    trial = 0;
    for subject in np.unique(ds['vpn']):
        for ip_cond in np.unique(ds['condition']):
            tmp_ds = ds[np.logical_and(ds['vpn'] == subject, ds['condition'] == ip_cond)]
            x_shape, y_shape = feature_shape(tmp_ds)

            # calc bins by quantil split
            x_out, x_bins = pd.qcut(np.asarray(x_shape), split, labels=False, retbins=True, duplicates='drop')
            y_out, y_bins = pd.qcut(np.asarray(y_shape), split, labels=False, retbins=True, duplicates='drop')

            for tmp in tmp_ds:
                # check if has field for processing or is empty
                if tmp['xr'].size == 0:
                    print('no Max: ', trial)
                    if (len(condition)%750) < 375 and len(condition)< 3360:
                        condition.append('imagination')
                    elif (len(condition)%750) > 375 and len(condition)< 3360:
                        condition.append('perception')
                    if ((len(condition)+15) % 750) < 375 and len(condition) > 3360:
                        condition.append('imagination')
                    elif ((len(condition)+15) % 750) > 375 and len(condition) > 3360:
                        condition.append('perception')

                    trial += 1
                else:
                    # print('trial: ', trial)
                    for k in range(tmp['xr'].size): # für jede fixation
                        for eye in ('r', 'l'):  # left right eye
                            if not np.isnan(tmp['x' + eye][k]) and tmp['x' + eye][k]>285 and tmp['x' + eye][k]<1635:
                                if tmp['y' + eye][k]>0 and tmp['y' + eye][k]<1080:
                                    # measures hitrate, eye(l/r), duration,pupil, blinks, num Of Fix
                                    curr_x = np.digitize((tmp['x' + eye][k]), x_bins) - 1
                                    curr_y = np.digitize((tmp['y' + eye][k]), y_bins) - 1
                                    curr_size = tmp['xr'].size

                                    # Hitrate relative to view
                                    X[trial, curr_y, curr_x, 0] += 1 / (curr_size*2)

                                    # Hitrate
                                    X[trial, curr_y, curr_x, 1] += 1/2

                                    # relative counts of eye
                                    X[trial, curr_y, curr_x, 2] += 1 / curr_size if eye == 'r' else 0

                                    # relative count of left eye
                                    X[trial, curr_y, curr_x, 3] += 1 / curr_size if eye == 'l' else 0

                                    # mean duration
                                    X[trial, curr_y, curr_x, 4] = ((X[trial, curr_y, curr_x, 4] * X[trial, curr_y, curr_x, 1])
                                                                   + tmp['dur'][k]) / (X[trial, curr_y, curr_x, 1] + 1)

                                    # mean time position
                                    X[trial, curr_y, curr_x, 5] =((X[trial, curr_y, curr_x, 5] * X[trial, curr_y, curr_x, 1])
                                                               + k )/ (X[trial, curr_y, curr_x, 1] + 1)

                                    # mean blinks
                                    X[trial, curr_y, curr_x, 6] = ((X[trial, curr_y, curr_x, 4] * X[trial, curr_y, curr_x, 1])
                                                                   + int(tmp['blink'][k] is not 'NONE')) / (
                                                                              X[trial, curr_y, curr_x, 1] + 1)

                                    # blinks
                                    X[trial, curr_y, curr_x, 7] += int(tmp['blink'][k] is not 'NONE')*0.5

                    y[trial] = tmp[y_cat][0, 0].astype(int)
                    group[trial] = tmp[group_cat][0, 0].astype(int)
                    condition.append(str(tmp['condition'][0]))
                    trial += 1
                    # print(X[trial,curr_y, curr_x, :])

    # [('group', 'O'), ('session', 'O'), ('block', 'O'), ('kat', 'O'), ('img', 'O'), ('trailID', 'O'), ('condition', 'O'),
    # ('xr', 'O'), ('xl', 'O'), ('yr', 'O'), ('yl', 'O'), ('dur', 'O'), ('pupil', 'O'), ('blink', 'O'),
    # ('numberOfBlink', 'O'), ('numberOfFix', 'O'), ('rating', 'O'), ('mean', 'O'), ('std', 'O'), ('nCat', 'O'),
    # ('nImg', 'O'), ('nVpn', 'O')]

    X = np.reshape(X, (len(ds[0, :]), reduce(lambda x, y: x * y, f_shape[1:])))
    X_i, X_p, group_i, group_p, y_i, y_p = split_and_rm_nan(X, y, ds, group, np.asarray(condition))

    np.savez('feature_map' + '_' + group_cat + '_' + y_cat + str(split), X_i=X_i, X_p=X_p, y_i=y_i, y_p=y_p,
             group_i=group_i, group_p=group_p)

    return X_p, y_p, X_i, y_i, group_p, group_i


def feature_shape(ds):
    import numpy as np
    x = list()
    y = list()
    i = 0;
    for tmp in ds:
        i += 1
        if tmp['xr'].size == 0:
            print('no Max: ', i)
        else:
            for curr_tmp in np.concatenate((tmp['xr'], tmp['xl'])):
                if not np.isnan(curr_tmp):
                    x.append(curr_tmp)
            for curr_tmp in np.concatenate((tmp['yr'], tmp['yl'])):
                if not np.isnan(curr_tmp):
                    y.append(curr_tmp)

    x = np.asarray(x)
    y = np.asarray(y)

    onScreen = np.logical_and(np.logical_and(x > 285, x < 1635), np.logical_and(y > 0, y < 1080))
    x = x[onScreen]
    y = y[onScreen]

    # xr, xl,yr,yl,dur, pup, blink
    y_ax = int(np.max(y)) + abs(int(np.min(y)))
    x_ax = int(np.max(x)) + abs(int(np.min(x)))
    # feature = (y_ax, x_ax, 7)
    # bins pd.qcut(np.array((0,0,2,3,40,5,9,33,42,33,6)),3, labels=False ,retbins=True,duplicates='drop')
    return x, y


def split_and_rm_nan(X, y, ds, vpn,condition):
    import numpy as np
    # split datasets
    perception = np.transpose(condition == 'perception')
    imagination = np.logical_not(perception)
    # find nan and exclude in y and x in perception and imagination set
    #not_missing = np.logical_or(condition == 'perception',condition == 'imagination')
    y_ntexcld= np.logical_and(y>0,y<16)
    not_excld = np.logical_not(np.logical_or(np.any(np.isnan(X), axis=1), np.isnan(y)))
    not_excld =np.logical_and(not_excld,y_ntexcld)
    not_excld = np.logical_and(not_excld[perception], not_excld[imagination])

    y_p = y[perception][not_excld].astype(int)
    y_i = y[imagination][not_excld].astype(int)
    X_p = X[perception][not_excld]
    X_i = X[imagination][not_excld]

    vpn_p = vpn[perception][not_excld]
    vpn_i = vpn[imagination][not_excld]
    print('missing data: ', str(((len(X)/2)-len(X_p))/(len(X)/2)))
    print('length of y_i: ',str(len(np.unique(y_i))))
    print('length of y_p: ', str(len(np.unique(y_p))))
    return X_i, X_p, vpn_i, vpn_p, y_i, y_p
