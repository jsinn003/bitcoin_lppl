import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import requests
import zipfile
from datetime import datetime, timedelta
from lppls import lppls, data_loader


def extract_zip(zip_file_path, destination_folder):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        zip_file.extractall(destination_folder)


def save_data_to_csv(data_frame, csv_filename, is_new_file=False):
    data_frame.to_csv(csv_filename, mode='a', header=is_new_file, index=False)


def collect_binance_historical_data(start_date, end_date, trading_pair='BTCUSDT', timeframe='1d'):
    output_folder = 'btc_usdt_data'
    combined_data_file = f'{output_folder}/full_btc_usdt_data.csv'
    column_names = [
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ]
    save_data_to_csv(pd.DataFrame(columns=column_names), combined_data_file, is_new_file=True)
    current_date = start_date
    while current_date <= end_date:
        date_string = current_date.strftime('%Y-%m-%d')
        binance_url = (
            f'https://data.binance.vision/data/futures/um/daily/klines/'
            f'{trading_pair}/{timeframe}/{trading_pair}-{timeframe}-{date_string}.zip'
        )
        daily_zip = f'{output_folder}/{date_string}.zip'
        daily_csv = f'{output_folder}/{trading_pair}-{timeframe}-{date_string}.csv'
        try:
            response = requests.get(binance_url)
            response.raise_for_status()
            with open(daily_zip, 'wb') as file:
                file.write(response.content)
            extract_zip(daily_zip, output_folder)
            daily_data = pd.read_csv(daily_csv)
            save_data_to_csv(daily_data, combined_data_file)
            os.remove(daily_zip)
            os.remove(daily_csv)
            print(f'Successfully processed data for {date_string}')
        except requests.RequestException as error:
            print(f'Failed to process data for {date_string}: {error}')
        current_date += timedelta(days=1)
    print(f"All data has been saved to {combined_data_file}")


if not os.path.exists('btc_usdt_data'):
    os.makedirs('btc_usdt_data')

start_date = datetime(2023, 1, 6)
end_date = datetime(2025, 1, 6)
collect_binance_historical_data(start_date, end_date)

# LPPL starts here
df = pd.read_csv("btc_usdt_data/full_btc_usdt_data.csv")
df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
data_for_lppl = df[['Open time', 'Close']].copy()
data_for_lppl.set_index('Open time', inplace=True)

# plt.figure(figsize=(10, 6))
# plt.plot(df['Open time'], df['Close'], label="Bitcoin Price")
# plt.title("Bitcoin Price Movement")
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.legend()
# plt.show()

time = [t1.toordinal() for t1 in data_for_lppl.index]
price = np.log(data_for_lppl['Close'].values)
observations = np.array([time, price])

MAX_SEARCHES = 25

lppls_model = lppls.LPPLS(observations=observations)
try:
    tc, m, w, a, b, c, c1, c2, O, D = lppls_model.fit(MAX_SEARCHES)
    print("Fitted Parameters:")
    print(f"tc (Critical Time): {tc}")
    print(f"m (Exponent): {m}")
    print(f"w (Frequency): {w}")
    print(f"a (Logarithmic Growth Rate): {a}")
    print(f"b (Coefficient of Power Law): {b}")
    print(f"c (Oscillation Amplitude): {c}")
    print(f"c1: {c1}")
    print(f"c2: {c2}")
    print(f"O (LPPL Fit Quality): {O}")
    print(f"D (Goodness-of-fit): {D}")
    # lppls_model.plot_fit()
    log_predicted_price = lppls_model.lppls(np.array(time), tc, m, w, a, b, c1, c2)
    predicted_price = np.exp(log_predicted_price)
    plt.figure(figsize=(12, 6))
    plt.plot(data_for_lppl.index, data_for_lppl['Close'], label="Actual Bitcoin Price", color="blue")
    plt.plot(data_for_lppl.index, predicted_price, label="LPPL Predicted Price", color="red", linestyle="--")
    plt.title("Bitcoin Price Movement with LPPL Model Fit")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
except Exception as e:
    print("Error fitting the model:", e)

if __name__ == '__main__':
    res = lppls_model.mp_compute_nested_fits(
        workers=4,
        window_size=60,
        smallest_window_size=15,
        outer_increment=5,
        inner_increment=10,
        max_searches=10
    )
    lppls_model.plot_confidence_indicators(res)
    plt.show()