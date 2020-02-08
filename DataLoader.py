import Constants
import json
import os
import datetime, pytz
import requests
import pandas as pd
import numpy as np

class DataLoader(object):

	def __init__(self, source='oanda'):
		self._source = source
		self._getOptions()
		if self._source == 'oanda':
			self._headers = {
				'Authorization': 'Bearer '+self._options['key']
			}
			self._url = (
				'https://api-fxpractice.oanda.com/v3/'
				if self._options['is_demo'] else
				'https://api-fxtrade.oanda.com/v3/'
			)
		else:
			raise Exception('Data source not found.')

	def _getOptions(self):
		if os.path.exists('options.json'):
			with open('options.json', 'r') as f:
				self._options = json.load(f)[self._source]
				
		else:
			raise Exception('Options file does not exist.')

	def get(self, product, period, start=None, end=None, current=False, override=False):
		data_path = os.path.join('data/', '{}/{}'.format(product, period))
		if os.path.exists(data_path) and len(os.listdir(data_path)) > 0 and not override:
			data = self.load(product, period, start, end)
			if current:
				ts = data.index[-1]
				data = pd.concat((
					data,
					self.download(
						product, period,
						self.convertTimestampToTime(ts),
						datetime.datetime.now()
					)
				))

				start_dt = self.convertTimestampToTime(data.index[0])
				end_dt = self.convertTimestampToTime(data.index[-1])
				self.save(data, product, period, start_dt, end_dt)
			return data
		else:
			return self.download(product, period, start, end, save=True)

	def load(self, product, period, start=None, end=None):
		if not start:
			start = Constants.TS_START_DATE
		if not end:
			end = datetime.datetime.now()

		data_dir = os.path.join('data/', '{}/{}/'.format(product, period))
		frags = []
		for y in range(start.year, end.year+1):
			data_path = os.path.join(data_dir, '{}-{}.csv'.format(y, y+1))
			if os.path.exists(data_path):
				t_data = pd.read_csv(data_path, sep=' ')
				if y == end.year:
					ts_end = self.convertTimeToTimestamp(end)
					t_data = t_data.loc[t_data['timestamp'] <= ts_end]
				frags.append(t_data)
		return pd.concat(frags).set_index('timestamp')

	def download(self, product, period, start=None, end=None, save=False):
		if self._source == 'oanda':
			if not start:
				start = Constants.TS_START_DATE
			if not end:
				end = datetime.datetime.now()
			data = self._oandaDownload(product, period, start_dt=start, end_dt=end)

			if save:
				self.save(data, product, period, start, end)
			return data

	def save(self, data, product, period, start, end):
		data_dir = os.path.join('data/', '{}/{}/'.format(product, period))
		if not os.path.exists(data_dir):
			os.makedirs(data_dir)

		data = data.round(pd.Series([5]*8, index=data.columns))
		for y in range(start.year, end.year+1):
			ts_start = self.convertTimeToTimestamp(datetime.datetime(year=y, month=1, day=1))
			ts_end = self.convertTimeToTimestamp(datetime.datetime(year=y+1, month=1, day=1))
			data_path = os.path.join(data_dir, '{}-{}.csv'.format(y, y+1))
			t_data = data.loc[(ts_start <= data.index) & (data.index < ts_end)]
			if t_data.size == 0:
				continue
			t_data.to_csv(data_path, sep=' ', header=True)

	def _oandaDownload(self, product, period, tz='Europe/London', start_dt=None, end_dt=None, count=None, result={}):
		if count:
			if start_dt:
				start_str = start_dt.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
				endpoint = 'instruments/{}/candles?price=BA' \
							'&from={}&count={}&granularity={}&alignmentTimezone={}&dailyAlignment=0'.format(
								product, start_str, count, period, tz
							)
			else:
				endpoint = 'instruments/{}/candles?price=BA' \
							'&count={}&granularity={}&alignmentTimezone={}&dailyAlignment=0'.format(
								product, count, period, tz
							)
		else:
			start_str = start_dt.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
			end_str = end_dt.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
			endpoint = 'instruments/{}/candles?price=BA' \
						'&from={}&to={}&granularity={}&alignmentTimezone={}&dailyAlignment=0'.format(
							product, start_str, end_str, period, tz
						)
		print(endpoint)
		res = requests.get(
			self._url + endpoint,
			headers=self._headers
		)

		if res.status_code == 200:
			if len(result) == 0:
				result['timestamp'] = []
				result['ask_open'] = []
				result['ask_high'] = []
				result['ask_low'] = []
				result['ask_close'] = []
				result['bid_open'] = []
				result['bid_high'] = []
				result['bid_low'] = []
				result['bid_close'] = []

			data = res.json()
			candles = data['candles']

			for i in candles:

				time = datetime.datetime.strptime(i['time'], '%Y-%m-%dT%H:%M:%S.000000000Z')
				ts = self.convertUTCTimeToTimestamp(time)

				result['timestamp'].append(ts)
				asks = list(map(float, i['bid'].values()))
				bids = list(map(float, i['ask'].values()))
				result['ask_open'].append(asks[0])
				result['ask_high'].append(asks[1])
				result['ask_low'].append(asks[2])
				result['ask_close'].append(asks[3])
				result['bid_open'].append(bids[0])
				result['bid_high'].append(bids[1])
				result['bid_low'].append(bids[2])
				result['bid_close'].append(bids[3])

			if count:
				if not self._oandaIsLastCandleFound(period, start_dt, end_dt, count):
					last_dt = datetime.datetime.strptime(candles[-1]['time'], '%Y-%m-%dT%H:%M:%S.000000000Z')
					return self._oandaDownload(product, period, start_dt=last_dt, end_dt=end_dt, count=5000, result=result)

			return pd.DataFrame(data=result).set_index('timestamp')
		if res.status_code == 400:
			print('({}) Bad Request: {}'.format(res.status_code, res.json()['errorMessage']))
			if 'Maximum' in res.json()['errorMessage'] or 'future' in res.json()['errorMessage']:
				return self._oandaDownload(product, period, start_dt=start_dt, end_dt=end_dt, count=5000, result={})
			else:
				return pd.DataFrame(data=result).set_index('timestamp')
		else:
			print('Error:\n{0}'.format(res.json()))
			return None

	def _oandaIsLastCandleFound(self, period, start_dt, end_dt, count):
		if period == Constants.ONE_MINUTE:
			return start_dt + datetime.timedelta(minutes=count) >= end_dt
		elif period == Constants.ONE_HOUR:
			return start_dt + datetime.timedelta(hours=count) >= end_dt
		elif period == Constants.FOUR_HOURS:
			return start_dt + datetime.timedelta(hours=count*4) >= end_dt
		elif period == Constants.DAILY:
			return start_dt + datetime.timedelta(hours=count*24) >= end_dt
		else:
			raise Exception('Period not found.')

	def convertTimezone(self, dt, tz):
		return dt.astimezone(pytz.timezone(tz))

	def setTimezone(self, dt, tz):
		return pytz.timezone(tz).localize(dt)

	def convertTimeToTimestamp(self, time):
		time = time.replace(tzinfo=None)
		return int((time - Constants.TS_CONVERT_DATE).total_seconds())

	def convertUTCTimeToTimestamp(self, time):
		time = self.setTimezone(time, 'UTC')
		time = self.convertTimezone(time, 'Australia/Melbourne').replace(tzinfo=None)
		return int((time - Constants.TS_CONVERT_DATE).total_seconds())

	def convertTimestampToTime(self, ts):
		return Constants.TS_CONVERT_DATE + datetime.timedelta(seconds=int(ts))
