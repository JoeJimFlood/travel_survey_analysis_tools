#DISCLAIMER: This code is not to be resold or used for commercial purposes and the user of this code assumes all liability for the consequences of its use

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def smooth(data_frame, time_series_name = str, dt = 1, filter_width = int, weights = None, plot_result = False):
    '''Smooths out time series using Fourier Transform and filter'''

    #Check if acceptable value of dt
    data_frame['time_remainder'] = data_frame[time_series_name].dropna().astype('int') % dt
    if data_frame['time_remainder'].sum() != 0:
        raise ValueError('Data do not support value of dt')
    del data_frame['time_remainder']

    time_data = data_frame[time_series_name].dropna().astype('int').value_counts() #Count number of values with each time
    times = time_data.index.tolist() #Get list of times
    for t in range(0, 1440, dt): #Check if gaps in time data. If gap exists, fill it in with x = 0 at t = t_gap
        if t not in times:
            time_data.loc[t] = 0
    time_data = time_data.sort_index() #Sort ascending

    t = np.array(time_data.index) #Define time array
    x = np.array(time_data) #Define x array
    n = len(t)

    x_transform = np.fft.fft(x) #Fast Fourier Transform on x-data
    ks = np.linspace(-n/2+1, n/2, n) #Get Fourier modes
    k = np.fft.ifftshift(ks) #Unshift Fourier modes
    filter = np.exp(-0.5*np.square(2 * k / filter_width)) #Define Gaussian filter
    x_transform_filtered = x_transform * filter #Multiply filter by transformed data
    x_filtered = np.fft.ifft(x_transform_filtered) #Use the inverse Fourier transform to transform the filtered data back into the time domain

    if plot_result:
        plt.plot(t, x)
        plt.plot(t, abs(x_filtered))
        #Use user-specified x and y axes labels
        x_axis_label = raw_input('Please enter a label for the x-axis: ')
        plt.xlabel(x_axis_label)
        y_axis_label = raw_input('Please enter a label for the y-axis: ')
        plt.ylabel(y_axis_label)
        plt.legend(['Data', 'Filtered Data'])
        plt.show()

    time_data_filtered = pd.Series(abs(x_filtered), t)
    return time_data_filtered

def find_peaks(t, x, ranges = [('AM', 0, 12), ('PM', 12, 24)]):
    '''Find peaks within specified time ranges'''

    peaks = {}
    data = pd.DataFrame.from_items([('t', t), ('x', x)])
    for i in range(len(ranges)): #Loop over time ranges

        range_name = ranges[i][0]
        min_time = ranges[i][1]
        max_time = ranges[i][2]

        #Check if time range bounds are real numbers
        if not isinstance(min_time, (int, long, float)) or not isinstance(max_time, (int, long, float)):
            raise IOError('Time range bounds need to be real numbers')

        current_range = data.query('t >= @min_time and t < @max_time') #Create subset of data over current time
        max_x = current_range['x'].max()
        current_max = current_range.query('x == @max_x')
        max_t = current_max['t'].tolist()[0]
        minute = int(round(max_t%1*60))
        peak_time = str(int(max_t)) + ':' + str(100 + minute)[1:]
        peaks.update({range_name: [max_t, max_x, peak_time]}) #Save peaks in a dictionary of lists: {Name: [peak time, peak height, time in HH:MM format]}
    return peaks

def plot_peaks(peaks, line_color, text_height):
    '''Plot given Peaks'''
    for peak in peaks:
        plt.plot(np.array([peaks[peak][0], peaks[peak][0]]), np.array([0, peaks[peak][1]]), color = line_color, linestyle = ':') #Vertical dotted line at the peak
        plt.text(peaks[peak][0], text_height, peak + u'\u2014' + peaks[peak][2], horizontalalignment = 'center', color = line_color) #Text identifying the peak

def get_dt(array):
    '''Gets the time step of an array of time values'''
    from fractions import gcd
    dt = 0
    for value in array:
        dt = gcd(dt, value) #Iteratevely Find the greatest common denomenator of each value
    if int(dt) != dt: #Check if not integer
        raise ValueError('Data must be able to support integer dt value')
    dt = int(dt) #Convert from floating point number to integer
    if dt == 0:
        dt = 1
    return dt

def create_time_plot(df_column_pairs, series_titles, plot_title, series_colors, series_linestyles, file_name, filter_width, x_limits, y_limits, peak_ranges = None, text_heights = None, peaks_to_use = None):
    '''Creates time distribution plot. Location of peaks is optional'''
    if peak_ranges:
        tx_pairs = [] #Store for later
    for i in range(len(df_column_pairs)):
        df_column_pairs[i][0][df_column_pairs[i][1]] = df_column_pairs[i][0][df_column_pairs[i][1]].dropna()
        dt = get_dt(df_column_pairs[i][0][df_column_pairs[i][1]].value_counts().index.tolist())
        tx_pair = smooth(df_column_pairs[i][0], df_column_pairs[i][1], dt, filter_width)
        tx_pair = tx_pair / tx_pair.sum() * (60.0/dt)
        t = 1./60 * np.array(tx_pair.index.tolist())
        x = np.array(tx_pair)
        plt.plot(t, x, color = series_colors[i], linestyle = series_linestyles[i])
        if peak_ranges:
            tx_pairs.append((t, x))
    plt.legend(series_titles)
    if peak_ranges and text_heights:
            if type(peak_ranges) == list:
                if peaks_to_use:
                    for i in range(len(df_column_pairs)):
                        peaks = find_peaks(tx_pairs[i][0], tx_pairs[i][1], [peak_ranges[peaks_to_use[i]]])
                        plot_peaks(peaks, series_colors[i], text_heights[i])
                else:        
                    for i in range(len(df_column_pairs)):
                        peaks = find_peaks(tx_pairs[i][0], tx_pairs[i][1], peak_ranges)
                        plot_peaks(peaks, series_colors[i], text_heights[i])
            else:
                for i in range(len(df_column_pairs)):
                    peaks = find_peaks(tx_pairs[i][0], tx_pairs[i][1])
                    plot_peaks(peaks, series_colors[i], text_heights[i])
    plt.title(plot_title)
    plt.xlabel('Hour of Day')
    plt.ylabel('Proportion of Daily Trips Generated Per Hour')
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.savefig(file_name)
    plt.cla()