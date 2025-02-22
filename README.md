# Bitcoin Price Analysis with LPPL Model

A Python program that downloads historical Bitcoin futures data from Binance and applies the Log-Periodic Power Law (LPPL) model to analyze potential bubble patterns in the price movement.

## Features

- Downloads historical Bitcoin/USDT futures data from Binance
- Processes and combines daily trading data
- Applies LPPL model to detect bubble patterns
- Generates visualizations of actual vs. predicted prices
- Includes confidence indicators for model fitting

## Prerequisites

```bash
pip install numpy pandas scipy matplotlib requests lppls
```

## Program Structure

The program consists of two main parts:
1. Data Collection
2. LPPL Analysis

### Data Collection Functions

- `extract_zip(zip_file_path, destination_folder)`: Extracts downloaded zip files
- `save_data_to_csv(data_frame, csv_filename, is_new_file)`: Handles CSV file operations
- `collect_binance_historical_data(start_date, end_date, trading_pair, timeframe)`: Downloads and processes historical data

### LPPL Analysis

The program uses the LPPLS model to analyze Bitcoin price movements and detect potential bubble patterns. It includes:
- Data preprocessing
- LPPL model fitting
- Visualization of results
- Confidence indicator calculation

## Usage

1. Set your customized date range in the main script:
```python
start_date = datetime(2023, 1, 6)
end_date = datetime(2025, 1, 6)
```

2. Run the script:
```bash
python lppl.py
```

3. The program will:
   - Create a 'btc_usdt_data' directory
   - Download and process historical data
   - Perform LPPL analysis
   - Generate visualization plots

## Output

The program generates:
1. CSV file with historical data (`btc_usdt_data/full_btc_usdt_data.csv`)
2. Plots showing:
   - Actual vs. LPPL-predicted Bitcoin prices
   - Confidence indicators for the model fit

## LPPL Model Parameters

The model outputs several parameters:
- `tc`: Critical Time (predicted time of bubble end)
- `m`: Exponent (growth rate)
- `w`: Frequency (oscillations)
- `a`: Logarithmic growth rate
- `b`: Power law coefficient
- `c`: Oscillation amplitude
- `c1, c2`: Phase parameters
- `O`: LPPL fit quality
- `D`: Goodness-of-fit measure

## Advanced Usage

For detailed confidence analysis, the program includes the parameters below:

```python
res = lppls_model.mp_compute_nested_fits(
    workers=4,
    window_size=60,
    smallest_window_size=15,
    outer_increment=5,
    inner_increment=10,
    max_searches=10
)
```

## Error Handling

The program includes error handling for:
- Failed data downloads
- LPPL model fitting issues
- File operations

## Notes

- The program uses Binance Futures data (BTCUSDT pair)
- Default timeframe is daily ('1d')
- LPPL model parameters are optimized using multiple searches (MAX_SEARCHES = 25)
- Parallel processing is used for nested fits computation

## License

This project is open-source and available under the MIT License.
