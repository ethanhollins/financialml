from DataLoader import DataLoader
import Constants
import datetime as dt

dl = DataLoader()
df = dl.get(Constants.GBPUSD, Constants.FOUR_HOURS, current=True)

print(df.head())
print(df.tail())
