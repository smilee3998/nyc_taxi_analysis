from typing import Dict
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta


def load_raw_csv(file_path: Path) -> pd.DataFrame:
    return parse_dataframe(pd.read_csv(file_path), year=int(file_path.stem.split("_")[0]))

def load_preprocessed_csv(file_path: Path)-> pd.DataFrame:
    return pd.read_csv(file_path, parse_dates=['tpep_pickup_datetime'], index_col='tpep_pickup_datetime')


def parse_dataframe(df: pd.DataFrame, year: int) -> pd.DataFrame:
    # remove na rows
    df = df.dropna()
    
    # convert passenger_count to integer
    df = df.astype({"passenger_count": "float32"}).astype({"passenger_count": "int8"})
    
    # Convert 'tpep_pickup_datetime' to datetime format
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')

    # Keep only rows with designated year 
    df = df[df['tpep_pickup_datetime'].dt.year == year]

    return df

def group_data_by_min(df: pd.DataFrame, minute: int) -> pd.DataFrame:
    # change the index to datetime and resample to the desired time interval (in minutes)
    df = df.set_index('tpep_pickup_datetime')
    df = df.resample(f'{minute}min').sum()
    
    df = df.reset_index()
    
    return df

def select_max_in_one_day(df: pd.DataFrame) -> pd.DataFrame:
    # select the data for the maximum passenger count in one day
    return (df.groupby(df['tpep_pickup_datetime'].dt.date))['passenger_count'].max()
    
def plot_data(df: pd.DataFrame):
    ax = df.plot(x='tpep_pickup_datetime', y='passenger_count', figsize=(16,9), x_compat=True, 
                 title="Max number of passenger count per hour in one day for the year 2022 and 2023", 
                 ylabel="Max number of passenger count per hour", xlabel="date")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    
    
def plot_anomoly_perday(df: pd.DataFrame, known_date: Dict[str, str]):
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    df.plot(y='passenger_count', ax=ax1, legend=False, sharex=True, x_compat=True)
    df.plot(y='anomaly_score', ax=ax2, legend=False, sharex=True, x_compat=True)

    anomalies = df[df['anomaly']== -1]
    
    # Visualize and label anomalies
    for time, count in zip(anomalies.index, anomalies['passenger_count']):
        date = time.to_pydatetime().date()
        ax1.axvline(x=date, color='green', linestyle='--')
        try:
            ax1.annotate(f'{known_date[str(date)]}',
                         xy=(time + timedelta(2), count))
        except KeyError as e:
            print(e)
            ax1.annotate(f'{date}', xy=(time, count))
    
    ax1.scatter(anomalies.index, anomalies['passenger_count'], label='anomaly',color='red')
    ax2.scatter(anomalies.index, anomalies['anomaly_score'], label='anomaly',color='red')
    
    # plot anomaly score threshold
    ax2.hlines(0, xmin=ax2.get_xlim()[0], xmax=ax2.get_xlim()[1], linestyles='--', colors='black')
    
    ax1.set_xticklabels([])
    ax1.set_xlabel('')
    ax1.set_ylabel('Max number of passenger count per hour per day')
    ax2.set_ylabel('Anomaly score')
    ax2.set_xlabel('date')
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    
    fig.suptitle("Anomaly detection for the year of 2022 and 2023")
    plt.show()