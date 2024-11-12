from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd
from pathlib import Path

import utils

KNOWN_DATE = {
    '2022-01-09': 'Winter Storm',
    '2022-01-29': 'Heavy Snowfall',
    '2022-12-25': 'Christmas Day',
    '2023-07-04': 'Independence Day',
    '2023-11-23': 'Thanksgiving Day', 
    '2023-12-25': 'Christmas Day'
}
    
    
def isolation_forest_detect(df: pd.DataFrame):
    passenger_count_standarize = pd.DataFrame(StandardScaler().fit_transform(df))
    model = IsolationForest(
                            contamination=1e-2,
                            max_samples=int(len(df)/10),
                            random_state=10 
                            )
    model.fit(passenger_count_standarize)
    
    df['anomaly_score']= model.decision_function(passenger_count_standarize)
    df['anomaly'] = model.predict(passenger_count_standarize)
    
    return df


if __name__ == '__main__':
    preprocessed_csv_path = Path("data/merged_max_count.csv")
    
    if not preprocessed_csv_path.exists():
        raise FileNotFoundError("")
    df = utils.load_preprocessed_csv(preprocessed_csv_path)

    df = isolation_forest_detect(df)
    utils.plot_anomoly_perday(df, known_date=KNOWN_DATE)