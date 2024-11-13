Preprocessing and Analyzing NYC Yellow taxi passenger count for 2022 and 2023 using Isolation forest 

# Steps
1. First, download the queried dataset as csv to the folder data/ from [2022dataset](https://data.cityofnewyork.us/Transportation/2022-Yellow-Taxi-Trip-Data/qp3b-zxtp/explore/query/SELECT%20%60tpep_pickup_datetime%60%2C%20%60passenger_count%60/page/filter) and [2023dataset](https://data.cityofnewyork.us/Transportation/2023-Yellow-Taxi-Trip-Data/4b4i-vvec/explore/query/SELECT%20%60tpep_pickup_datetime%60%2C%20%60passenger_count%60/page/filter)
2. install required packages `pip install -r requirements.txt`
3. Run all code in preprocess.iypnb. It first group the number of passenger per hour and do data cleaning to remove problematic data in the original dataset. Then it locates the maximum number of passenger per hour in one day to further reduce the data size. Finally it saves the processed data as csv files.
4. Run `python anomoly_detection.py`

# Results
![result](https://github.com/user-attachments/assets/9cfe44d2-cf82-4941-97f1-47c54623042e)
