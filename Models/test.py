from DataLoader import DataLoader
import Constants
import datetime as dt
import tensorflow as tf

dl = DataLoader()
df = dl.get(Constants.GBPUSD, Constants.FOUR_HOURS)

hlc = df[['bid_high', 'bid_low', 'bid_close']]

print(hlc.head())
