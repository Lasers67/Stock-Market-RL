import pandas as pd
import pandas_ta as ta
import numpy as np
EMA1 = 9
EMA2 = 21
EMA3 = 42
EMA4 = 63
EMA5 = 30
EMA6 = 74
EMA7 = 100
EMA8 = 200
class Enricher:
    def get_enriched_from_df(self, symbol, df):
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        # df['symbol'] = symbol
        return self.add_indicators_signals(df, symbol)

    def get_enriched_data(self, symbol, data):
        df = pd.DataFrame(data)
        df = df.sort_values(by='timestamp')
        # print('adding enricher for', symbol)
        return self.add_indicators_signals(df, symbol)
    
    def add_indicators_signals(self, df, symbol):
        df['inr_volume'] = (df['close']*df['volume'])/1e6
        sti = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=1)
        df[['ST_10_1.0', 'STd_10_1', 'STl_10_1.0', 'STs_10_1.0']] = sti
        df['rsi_12'] = ta.rsi(df['close'], length=12)
        df['rsi_12_sma_12'] = ta.sma(df['rsi_12'], length=12)
        df['rsi_12_sma_4'] = ta.sma(df['rsi_12'], length=4)
        df['st_momo'] = df['rsi_12'] > df['rsi_12_sma_4']
        df['mt_momo'] = df['rsi_12'] > df['rsi_12_sma_12']
        df['lt_momo'] = df['rsi_12_sma_4'] > df['rsi_12_sma_12']
        df['sma_12'] = ta.sma(df['close'], length=12)
        df['ema1'] = ta.ema(df['close'], length=EMA1)
        df['ema2'] = ta.ema(df['close'], length=EMA2)
        df['ema3'] = ta.ema(df['close'], length=EMA3)
        df['ema4'] = ta.ema(df['close'], length=EMA4)
        # df['ema5'] = ta.ema(df['close'], length=EMA5)
        # df['ema6'] = ta.ema(df['close'], length=EMA6)
        # df['ema7'] = ta.ema(df['close'], length=EMA7)
        # df['ema8'] = ta.ema(df['close'], length=EMA8)
        df['ema2_ch'] = (100*(df['ema1'] - df['ema1'].shift(1))/df['ema2'].shift(1))
        df['close_ema2'] = (100*(df['close'] - df['ema2'])/df['close'].shift(1))
        df['close_above_prev_high'] = (df['close'] > df['high'].shift(1))
        df['close_below_prev_low'] = (df['close'] < df['low'].shift(1))
        df['cross_prev_hl'] = df.apply(lambda row: '1' if row['close_above_prev_high'] else '-1' if row['close_below_prev_low'] else 0, axis=1)
        df['ema_trend_up'] = ((df['ema1'] > df['ema2']) & (df['ema2'] > df['ema3']) & (df['ema3'] > df['ema4']))
        df['ema_trend_st_up'] = (df['ema1'] > df['ema2'])
        df['ema_trend_lt_up'] = (df['ema3'] > df['ema4'])
        df['ema_trend_down'] = ((df['ema1'] < df['ema2']) & (df['ema2'] < df['ema3']) & (df['ema3'] < df['ema4']))
        df['ema_trend_st_down'] = (df['ema1'] < df['ema2'])
        df['ema_trend_lt_down'] = (df['ema3'] < df['ema4'])
        df['ema_trend_st'] = df.apply(lambda row: '1' if row['ema_trend_st_up'] else '-1' if row['ema_trend_st_down'] else 0, axis=1)
        df['ema_trend_st_ch'] = (df['ema_trend_st'].shift(1) != df['ema_trend_st'])
        df['ema_trend_lt'] = df.apply(lambda row: '1' if row['ema_trend_lt_up'] else '-1' if row['ema_trend_lt_down'] else 0, axis=1)
        df['ch_p'] = round((df['close'] - df['close'].shift(1))/ df['close'].shift(1) * 100,2)
        df['gap_p'] = round((df['open'] - df['close'].shift(1))/ df['close'].shift(1) * 100,2)
        df['intra_p'] = round((df['close'] - df['open'])/ df['close'] * 100,2)
        df['range_p'] = round((df['high'] - df['low'])/ df['close'] * 100,2)
        df['st_p'] = round((df['close'] - df['ST_10_1.0'])/ df['close'] * 100,2)
        # print(df.columns)
        # df.to_csv('/Users/praprsa/Documents/enriched_data'+symbol+'.csv')
        return df

def merge_and_remove_duplicates(part1_path, part2_path):
    try:
        df1 = pd.read_csv(part1_path)
        df2 = pd.read_csv(part2_path)

        merged_df = pd.concat([df1, df2], ignore_index=True)
        merged_df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        merged_df['date'] = pd.to_datetime(merged_df['date'])
        merged_df = merged_df.sort_values(by='date', ascending=True)

        merged_df = merged_df.drop_duplicates(subset=['date'], keep='first')

        return merged_df

    except FileNotFoundError:
        print("Error: One or both CSV files not found.")
        return None
    except KeyError:
        print("Error: 'date' column not found in one or both CSV files.")
        return None

# Example usage
part1_path = 'nse_nifty_part1.csv'
part2_path = 'nse_nifty_part2.csv'
merged_file_path = 'data.csv'
enriched_file_path = 'data2.csv'

merged_df = merge_and_remove_duplicates(part1_path, part2_path)
merged_df.to_csv(merged_file_path, index=False)


enricher = Enricher()
df = pd.read_csv(merged_file_path)
enricher.get_enriched_from_df('NSE:NIFTY50_INDEX', df)
df.to_csv(enriched_file_path, index=False)