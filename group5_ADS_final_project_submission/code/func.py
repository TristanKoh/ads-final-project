import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.metrics import mean_squared_error

from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict

# Defining the Augmented Dickey-Fuller Test
def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)

# Defining the Kwiatkowski–Phillips–Schmidt–Shin Test
def kpss_test(timeseries):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)

# Define plot lags function
def plot_lags(time_series):
    """
    Plots lag 1 to 30 for a time series.
    
    Parameters:
    time_series (pandas.Series): A pandas series containing the time series data.
    
    Returns:
    None
    """
    
    # Create a new figure and set the size
    fig = plt.figure(figsize=(16, 16))
    
    # Loop through lags 1 to 30
    for i in range(1, 31):
        # Create a subplot with a specific position
        ax = fig.add_subplot(6, 5, i)
        # Plot the lagged data
        lag_plot(time_series, lag=i)
        # Set the title of the subplot
        ax.set_title(f"Lag {i}")
        
    # Adjust the spacing between the subplots
    fig.tight_layout()
    
    # Show the plot
    plt.show()


# Define plotting rolling average function
def plot_rolling_average(time_series):
    """
    Plots rolling average for 7, 14, 30 and 60 days for a time series.
    
    Parameters:
    time_series (pandas.Series): A pandas series containing the time series data.
    
    Returns:
    None
    """
    
    # Define the window sizes for the rolling averages
    window_sizes = [7, 14, 30, 60]
    
    # Create a new figure and set the size
    fig = plt.figure(figsize=(12, 8))
    
    # Loop through the window sizes
    for i, window in enumerate(window_sizes):
        
        # # Calculate the rolling average
        # rolling_avg = time_series.rolling(window=window).mean()
        
        # Create a subplot with a specific position
        ax = fig.add_subplot(2, 2, i+1)
        
        # Plot the rolling average
        # ax.plot(time_series.index, rolling_avg)
        time_series.rolling(window).mean().plot(ax = ax)

        # Set the title of the subplot
        ax.set_title(f"Rolling Average ({window} days)")
        
    # Adjust the spacing between the subplots
    fig.tight_layout()
    
    # Show the plot
    plt.show()


# Define plot seasonal trend function
def plot_seasonal(x, labels, hue, df, num_rows, num_cols, width, height):
    """
    Plots seasonal trend of each day / month, with one timeseries per month / year.
    Plots a seasonal trend subplot for each category in labels.

    Parameters:
    x: Time period for x axis (string)
    labels: List of categories to iterate through, e.g. items sold (list)
    hue: Time period for each timeseries 
    df: pandas dataframe
    num_rows: number of rows for subplots
    num_cols: number of cols for subplots
    width: width of graph
    height: height of graph
    
    Returns:
    None
    """

    fig = plt.figure(figsize=(width, height))

    # Loop through the window sizes
    for i, cat in enumerate(labels):
        
        # Create a subplot with a specific position
        ax = fig.add_subplot(num_rows, num_cols, i+1)
        ax.tick_params(axis='x', rotation=90)

        # Plot the timeseries
        sns.lineplot(x = x, y = cat, data = df, hue = hue, ax = ax, errorbar = None)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

        # Set the title of the subplot
        ax.set_title(f"Seasonal trend for {cat}")
        
    # Adjust the spacing between the subplots
    fig.tight_layout()
    
    # Show the plot
    plt.show()


# Define plotting autocorrelation function
def plot_autocorrelation(labels, df, num_rows, num_cols, width, height):
    """
    Plots autocorrelation graph for each category as subplots.

    Parameters:
    labels: List of categories to iterate through, e.g. region list (list)
    df: pandas dataframe
    num_rows: number of rows for subplots
    num_cols: number of cols for subplots
    width: width of graph
    height: height of graph
    
    Returns:
    None
    """

    fig = plt.figure(figsize=(width, height))

    # Loop through the window sizes
    for i, cat in enumerate(labels):
        
        # Create a subplot with a specific position
        ax = fig.add_subplot(num_rows, num_cols, i+1)

        # Plot the timeseries
        autocorrelation_plot(df[cat], ax = ax)

        # Set the title of the subplot
        ax.set_title(f"Autocorrelation plot for {cat}")
        
    # Adjust the spacing between the subplots
    fig.tight_layout()
    
    # Show the plot
    plt.show()


def round_positive(s) :
    """
    Function to change to 0 if negative, if not round to nearest int
    """
    s[s < 0 ] = 0
    s = s.round().astype(int)

    return s

# Function to convert predictions into submission csv
def convert_to_sub_csv(preds_df, sub_df, method) :
    """
    Used to convert predictions returned by statsforecast models, returns dataframe for submission on kaggle.
    Pred_df, sub_df: pandas dataframe
    method: string matching the statsforecast method
    """
    
    df_converted = preds_df[["unique_id", "ds", method]].pivot(index = "unique_id", columns = "ds", values = method)

    # Change col names back to day ints
    day_to_d = dict(zip(list(df_converted.columns), list(sub_df.columns[1:])))
    df_converted = df_converted.rename(day_to_d, axis = 1).reset_index()

    # Round up to nearest int
    df_converted.iloc[:, 1:] = df_converted.iloc[:, 1:].round().astype(int)    

    # Sort into the original ordering by ID
    df_converted[["category", "store", "num", "region", "num_2"]] = df_converted["unique_id"].str.split("_", expand = True)
    df_converted["region"] = pd.Categorical(df_converted["region"], ["East", "Central", "West"])
    df_converted = df_converted.sort_values(by = ["region", "num_2", "category", "store", "num"])
    df_converted = df_converted.drop(["category", "store", "num", "region", "num_2"], axis =1)

    # Rename ID col
    df_converted = df_converted.rename(columns = {"unique_id" : "id"})

    return df_converted

def print_rmse(y_true, y_pred) :
    """
    Function to print the rmse of a prediction. Accepts 2 pandas series.
    """
    print(np.sqrt(mean_squared_error(y_true, y_pred)))

def plot_differencing(series, n = 8) :

    fig, axes = plt.subplots(n, 3, figsize = (20,25))

    axes[0, 0].plot(series); axes[0, 0].set_title('Original Series')
    plot_acf(series, ax=axes[0, 1])

    for i in range(1, n) :
        diff_series = series.diff(periods = i)

        axes[i, 0].plot(diff_series); axes[i, 0].set_title(f'{i}th Order Differencing')
        plot_acf(diff_series.dropna(), ax=axes[i, 1])
        plot_pacf(diff_series.dropna(), method='ywm', ax = axes[i, 2])


    plt.show()