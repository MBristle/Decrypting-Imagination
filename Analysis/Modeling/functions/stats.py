
def bootstrap(data, n_it=5000, eval = 'mean'):
    #
    import numpy as np
    import matplotlib as plt
    from sklearn.utils import resample
    n_size = int(len(data))
    resample(data, n_samples=n_size)
    # run bootstrap
    stats = list()
    for i in range(n_it):
        # prepare train and test sets
        resampled_data = resample(data, n_samples=n_size)

        if eval == 'mean':
            score = np.mean(resampled_data)
        elif eval == 'median':
            score = np.median(resampled_data)

        stats.append(score)
    # plot scores
    #plt.hist(stats)
    #plt.show()
    if eval == 'mean':
        sum_score = np.mean(resampled_data)
    elif eval == 'median':
        sum_score = np.median(resampled_data)
    # confidence intervals
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = np.percentile(stats, p)
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper =  np.percentile(stats, p)
    print(eval+': %.3f and %.1fCI %.2f; %.2f' % (sum_score, alpha * 100, lower , upper ))
    return sum_score, upper, lower, stats
