#DISCLAIMER: This code is not to be resold or used for commercial purposes and the user of this code assumes all liability for the consequences of its use

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time

def ncr(n, r):
    from math import factorial as f
    ans = f(n)/(f(r)*f(n-r))
    return ans

def binomial_expand(a, n):
    coeffs = []
    for k in range(n+1):
        coeffs.append(ncr(n, k)*np.power(y, n-k))
    return coeffs

def quantile_identify(in_series, n_quantiles = 4, plot_dist = False, plot_title = '', x_axis_label = '', y_axis_label = ''):
    '''Returns series identifying the quantile each corresponding element in the input series is in
    Plotting a density plot of the series with the quantiles indicated is optional'''
    #Setting the number of quantiles to more than 1000 results in bad things down the road, so check for it now
    if n_quantiles > 1000:
        while n_quantiles > 1000:
            n_quantiles = input("There's a chance that setting the number of quantiles to be greater than 1,000 could cause the universe to explode, so let's not risk it. How many quantiles do you want?\n")

    #Check if scipy installed (for plotting)
    if plot_dist:
        try:
            from scipy.stats import gaussian_kde
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Scipy and/or Matplotlib aren't installed, so plotting the distribution with quantiles just isn't going to happen.")
            plot_dist = False
            plot_title = ''
            x_axis_label = ''
            y_axis_label = ''
            
    #Locate quantiles
    p = []
    for i in range(1, n_quantiles):
        p.append(float(i)/n_quantiles)
    qs = in_series.describe(percentiles = p)
    qsi = qs.index.tolist()
    #print qsi
    q = {}
    q[0] = 0
    for i in range(1, n_quantiles):
        index_percent = round(100*float(i)/n_quantiles, 1)
        if str(int(index_percent)) + '%' in qsi:
            #print ('Yes 1')
            ind = str(int(index_percent)) + '%'
        else:
            #print ('No 1')
            if str(index_percent) + '%' in qsi:
                #print ('Yes 2')
                ind = str(index_percent) + '%'
            else:
                #print ('No 2')
                ind = str(index_percent - 0.1) + '%'
        #print index
        q[i] = qs[ind]
    q[n_quantiles] = in_series.max() + 1

    #Plot distribution with quantiles
    if plot_dist:
        support = np.linspace(0, in_series.max(), 1000)
        d = gaussian_kde(in_series.dropna())
        plt.plot(support, d(support))
        for i in range(1, n_quantiles):
            plt.plot([q[i], q[i]], [0, d(q[i])], color = '#000000', linestyle = '--')
        if plot_title:
            plt.title(plot_title)
        if x_axis_label:
            plt.xlabel(x_axis_label)
        if y_axis_label:
            plt.ylabel(y_axis_label)
        plt.show()

    #Categorize series values based on quantiles
    def categorize(number):
        if np.isnan(number):
            quantile = number
            return quantile
        for i in range(n_quantiles):
            if number >= q[i] and number < q[i + 1]:
                quantile = i + 1
        return quantile

    #Categorize values
    out_series = in_series.apply(categorize)
    return out_series

def egoa(in_series, group_size, uneven_last = True, plot_dist = False, plot_title = False, x_axis_label = '', y_axis_label = ''):
    '''Even group ordinal aggregation. Similar to quantile identify, but all groups (except the first or last) are guaranteed to have the same size'''
    in_series = in_series.sort(inplace = False) #Sort series
    if plot_dist:
        try:
            from scipy.stats import gaussian_kde
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Scipy and/or Matplotlib aren't installed, so plotting the distribution with divisions just isn't going to happen.")
            plot_dist = False
            plot_title = ''
            x_axis_label = ''
            y_axis_label = ''
    
        support = np.linspace(0, in_series.max(), 1000)
        d = gaussian_kde(in_series)
        plt.plot(support, d(support), color = '#000000')

    n_groups = int(np.ceil(float(len(in_series))/group_size)) #Compute the number of groups based on inputs
    out_series = pd.Series(index = in_series.index)

    #Check if the last group is to be the group smaller than the others
    if uneven_last:
        for i in range(1, n_groups):
            #Define all of the elements in each group, and set their corresponting outputs to be the group number
            current_set = in_series.iloc[(i-1)*group_size:i*group_size]
            for item in current_set.index:
                out_series.loc[item] = i
            if plot_dist:
                x = in_series.iloc[i*group_size]
                plt.plot([x, x], [0, d(x)], linestyle = '--', color = '#000000')
        #Now do this for the final group
        final_set = in_series.iloc[(n_groups-1)*group_size:]
        for item in final_set.index:
            out_series.loc[item] = n_groups
    else:
        n = len(in_series)
        for i in range(1, n_groups):
            #Define all of the elements in each group, and set their corresponting outputs to be the group number
            current_set = in_series.iloc[n-i*group_size:n-(i-1)*group_size]
            for item in current_set.index:
                out_series.loc[item] = n_groups - i + 1
            if plot_dist:
                x = in_series.iloc[n-i*group_size]
                plt.plot([x, x], [0, d(x)], linestyle = '--', color = '#000000')
        #Now do this for the first group
        initial_set = in_series.iloc[:n-(n_groups-1)*group_size]
        for item in initial_set.index:
            out_series.loc[item] = 1
    if plot_title:
        plt.title(plot_title)
    if x_axis_label:
        plt.xlabel(x_axis_label)
    if y_axis_label:
        plt.ylabel(y_axis_label)
    if plot_dist:
        plt.show()
    return out_series

def plot_2d_kde(df, columns, range = None, aggregation = 1, resolution = (800, 600), colormap = plt.cm.hot, title = '', axis_labels = ('', ''), outfilepath = ''):
    '''Plots 2-dimensional Gaussian kernal density estimation plot for specified data'''
    timerstart = time.time()
    df = df[columns].dropna()

    if aggregation > 1:
        df['Q'] = egoa(df[columns[0]], aggregation)
        n_groups = df['Q'].max()
        df = df.query('Q < @n_groups')
        df = df.groupby('Q').mean()

    #Check if maximum value of x specified. If not, just have it as the maximum value of the x column
    if range:
        hist_data = df

        #Define x_min
        if range[0] == 'min':
            x_min = x.min()
        elif range[0] or range[0] == 0:
            x_min = range[0]
            hist_data = hist_data.query(columns[0] + ' >= @x_min')
        else:
            x_min = x.min()

        #Define x_max
        if range[1] == 'max':
            x_max = x.max()
        elif range[1] or range[1] == 0:
            x_max = range[1]
            hist_data = hist_data.query(columns[0] + ' <= @x_max')
        else:
            x_max = x.max()

        #Define y_min
        if range[2] == 'min':
            y_min = y.min()
        elif range[2] or range[2] == 0:
            y_min = range[2]
            hist_data = hist_data.query(columns[1] + ' >= @y_min')
        else:
            y_min = y.min()

        #Define y_max
        if range[3] == 'max':
            y_max = y.max()
        elif range[3] or range[3] == 0:
            y_max = range[3]
            hist_data = hist_data.query(columns[1] + ' <= @y_max')
        else:
            y_max = y.max()
        x = np.array(hist_data[columns[0]])
        y = np.array(hist_data[columns[1]])
    else:
        x = np.array(df[columns[0]])
        y = np.array(df[columns[1]])
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()

    print ('Defining kernel')
    X, Y = np.mgrid[x_min:x_max:resolution[0]*1j, y_min:y_max:resolution[1]*1j] #Define meshgrid
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions), X.shape) #Reshape array in order to be plotted

    print('Plotting kernel')   
    plt.imshow(np.rot90(Z), cmap = colormap, extent = [x_min, x_max, y_min, y_max], aspect = 'auto')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title(title)
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    print(time.time() - timerstart)
    if outfilepath:
        plt.savefig(outfilepath)
        plt.cla()
    else:
        plt.show()

def series_standardize(in_series):
    '''Standardizes values in a series'''
    out_series = (in_series - in_series.mean())/in_series.std()
    return out_series

def series_log_standardize(in_series):
    '''Standardizes the natural logarithm of values in a series'''
    in_series = np.log(in_series)
    out_series = (in_series - in_series.mean())/in_series.std()
    return out_series