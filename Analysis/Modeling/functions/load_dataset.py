def load_summary():
    ## %matplotlib inline

    import scipy.io as io
    import numpy as np
    data = io.loadmat('dataset_raw_sumMAT.mat')
    ds = data['ds']

    # Size of Feature vector
    sz = 8

    # X -> features, y -> label
    vpn = np.zeros((ds.shape[1]))
    y = np.zeros((ds.shape[1]))
    X = np.zeros((ds.shape[1], sz))

    for i in range(0, len(X[:, 0] - 1)):
        tmp = ds['mean'][0, i]
        X[i, 0] = tmp['xr']
        X[i, 1] = tmp['yr']
        X[i, 2] = tmp['dur']
        X[i, 3] = tmp['pupil']
        X[i, 4] = ds['numberOfFix'][0, i][0][0].astype('int')
        X[i, 5] = ds['numberOfBlink'][0, i][0][0].astype('int')
        X[i, 6] = tmp['xl']
        X[i, 7] = tmp['yl']
        y[i] = ds['nCat'][0, i][0, 0].astype(int)
        vpn[i] = ds['nVpn'][0, i][0, 0].astype(int)

    X_i, X_p, vpn_i, vpn_p, y_i, y_p = split_and_rm_nan(X, y, ds, vpn)

    return X_p, y_p, X_i, y_i, vpn_p, vpn_i


def load_map(split=6, y_cat='nCat', group_cat='nVpn', load=True):
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

    import pandas as pd
    data = io.loadmat('dataset_raw_sumMAT.mat')

    ds = data['ds']

    x_shape, y_shape = feature_shape(ds)

    f_shape = (len(ds[0, :]),) + (split + 1, split + 1, 8)
    # calc bins by quantil split
    x_out, x_bins = pd.qcut(np.asarray(x_shape), split, labels=False, retbins=True, duplicates='drop')
    y_out, y_bins = pd.qcut(np.asarray(y_shape), split, labels=False, retbins=True, duplicates='drop')

    # X -> features, y -> label
    group = np.empty((ds.shape[1]))
    y = np.empty((ds.shape[1]))
    y[:] = np.nan
    X = np.empty(shape=f_shape)
    i = 0;
    for tmp in ds[0, :]:
        # check if has field for processing or is empty
        if tmp['xr'].size == 0:
            print('no Max: ', i)
            i += 1
        else:
            # print('sample: ', i)

            for k in range(tmp['xr'].size):
                for eye in ('r', 'l'):  # left right eye
                    if not np.isnan(tmp['x' + eye][k]):
                        # measures hitrate, eye(l/r), duration,pupil, blinks, num Of Fix
                        curr_x = np.digitize((tmp['x' + eye][k]), x_bins) - 1
                        curr_y = np.digitize((tmp['y' + eye][k]), y_bins) - 1
                        curr_size = tmp['xr'].size

                        # Hitrate relative to view
                        X[i, curr_y, curr_x, 0] += 1 / curr_size

                        # Hitrate
                        X[i, curr_y, curr_x, 1] += 1

                        # relative counts of eye
                        X[i, curr_y, curr_x, 2] += 1 / curr_size if eye == 'r' else 0

                        # relative count of left eye
                        X[i, curr_y, curr_x, 3] += 1 / curr_size if eye == 'l' else 0

                        # mean duration
                        X[i, curr_y, curr_x, 4] = ((X[i, curr_y, curr_x, 4] * X[i, curr_y, curr_x, 1])
                                                   + tmp['dur'][k]) / (X[i, curr_y, curr_x, 1] + 1)

                        # mean pupil
                        X[i, curr_y, curr_x, 5] = ((X[i, curr_y, curr_x, 5] * X[i, curr_y, curr_x, 1])
                                                   + tmp['pupil'][k]) / (X[i, curr_y, curr_x, 1] + 1)

                        # mean blinks
                        X[i, curr_y, curr_x, 6] = ((X[i, curr_y, curr_x, 4] * X[i, curr_y, curr_x, 1])
                                                   + int(tmp['blink'][k] is not 'NONE')) / (X[i, curr_y, curr_x, 1] + 1)

                        # blinks
                        X[i, curr_y, curr_x, 7] += int(tmp['blink'][k] is not 'NONE')

                        y[i] = tmp[y_cat][0, 0].astype(int)

                        group[i] = tmp[group_cat][0, 0].astype(int)
            i += 1
            # print(X[i,curr_y, curr_x, :])

    #[('group', 'O'), ('session', 'O'), ('block', 'O'), ('kat', 'O'), ('img', 'O'), ('trailID', 'O'), ('condition', 'O'),
     #('xr', 'O'), ('xl', 'O'), ('yr', 'O'), ('yl', 'O'), ('dur', 'O'), ('pupil', 'O'), ('blink', 'O'),
     #('numberOfBlink', 'O'), ('numberOfFix', 'O'), ('rating', 'O'), ('mean', 'O'), ('std', 'O'), ('nCat', 'O'),
     #('nImg', 'O'), ('nVpn', 'O')]


    X = np.reshape(X, (len(ds[0, :]), reduce(lambda x, y: x * y , f_shape[1:])))

    X_i, X_p, group_i, group_p, y_i, y_p = split_and_rm_nan(X, y, ds, group)
    np.savez('feature_map'+'_'+group_cat+'_'+y_cat+str(split), X_i=X_i, X_p=X_p, y_i=y_i, y_p=y_p, group_i=group_i, group_p=group_p)

    return X_p, y_p, X_i, y_i, group_p, group_i


def feature_shape(ds):
    import numpy as np
    max_x = 0
    max_y = 0
    min_x = 0
    min_y = 0
    x = list()
    y = list()
    i = 0;
    for tmp in ds[0, :]:
        i += 1
        if tmp['xr'].size == 0:
            print('no Max: ', i)
        else:
            curr_max = max(tmp['xr'].max(), tmp['xl'].max())
            max_x = curr_max if (curr_max > max_x) else max_x
            curr_max = max(tmp['yr'].max(), tmp['yl'].max())
            max_y = curr_max if (curr_max > max_y) else max_y
            curr_min = min(tmp['xr'].min(), tmp['xl'].min())
            min_x = curr_min if (curr_min < min_x) else min_x
            curr_min = min(tmp['yr'].min(), tmp['yl'].min())
            min_y = curr_min if (curr_min < min_y) else min_y
            for curr_tmp in np.concatenate((tmp['xr'], tmp['xl'])):
                if not np.isnan(curr_tmp):
                    x.append(curr_tmp)
            for curr_tmp in np.concatenate((tmp['yr'], tmp['yl'])):
                if not np.isnan(curr_tmp):
                    y.append(curr_tmp)

    # xr, xl,yr,yl,dur, pup, blink
    y_ax = int(max_y) + abs(int(min_y))
    x_ax = int(max_x) + abs(int(min_x))
    #feature = (y_ax, x_ax, 7)
    # bins pd.qcut(np.array((0,0,2,3,40,5,9,33,42,33,6)),3, labels=False ,retbins=True,duplicates='drop')
    return  x, y


def split_and_rm_nan(X, y, ds, vpn):
    import numpy as np
    # split datasets
    perception = np.transpose(ds['condition'] == 'perception')
    imagination = np.transpose(ds['condition'] == 'imagination')
    # find nan and exclude in y and x
    not_excld = np.logical_not(np.logical_or(np.any(np.isnan(X), axis=1),np.isnan(y)))
    not_excld = np.logical_and(not_excld[:int(len(not_excld) / 2)], not_excld[int(len(not_excld) / 2):])
    not_excld = np.concatenate((not_excld, not_excld), axis=0)

    y_p = y[np.logical_and(not_excld, perception[:, 0])]
    y_i = y[np.logical_and(not_excld, imagination[:, 0])]
    print('startX_P')
    X_p = X[np.logical_and(not_excld, perception[:, 0])]
    print('startX_I')
    X_i = X[np.logical_and(not_excld, imagination[:, 0])]

    vpn_p = vpn[np.logical_and(not_excld, perception[:, 0])]
    vpn_i = vpn[np.logical_and(not_excld, imagination[:, 0])]
    return X_i, X_p, vpn_i, vpn_p, y_i, y_p
