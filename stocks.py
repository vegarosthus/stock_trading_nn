
import time
import csv
import urllib.request

import numpy as np
import cvxopt

from numpy import convolve

#from stock_market import StockMarket
from commodities import Commodity


class Company:
	def __init__(self, name):
		self.name = name

class Stock(Company):
	def __init__(self, name):
		Company.__init__(self, name)
		
		self.time = []
		self.dates = []
		self.prices = []
		self.opens = []
		self.highs = []
		self.lows = []
		self.closes = []
		self.volumes = []
		self.values = []
	
		self.prices_sma = []
		self.opens_sma = []
		self.highs_sma = []
		self.lows_sma = []
		self.closes_sma = []
		self.volumes_sma = []
		self.values_sma = []
		
		self.prices_var = []
		self.opens_var = []
		self.highs_var = []
		self.lows_var = []
		self.closes_var = []
		self.volumes_var = []
		self.values_var = []
		
		self.prices_std = []
		self.opens_std = []
		self.highs_std = []
		self.lows_std = []
		self.closes_std = []
		self.volumes_std = []
		self.values_std = []
	
		self.prices_upper_bb = []
		self.prices_lower_bb = []
		self.opens_upper_bb = []
		self.opens_lower_bb = []
		self.highs_upper_bb = []
		self.highs_lower_bb = []
		self.lows_upper_bb = []
		self.lows_lower_bb = []
		self.closes_upper_bb = []
		self.closes_lower_bb = []
		self.volumes_upper_bb = []
		self.volumes_lower_bb = []
		self.values_upper_bb = []
		self.values_lower_bb = []
	
		self.prev_no_of_lines = 0
		self.no_of_lines = 0

	def compute_bb(self, values, window):
#		try:
		weighs = np.repeat(1.0, window)/window
		sma = np.convolve(values, weighs, 'valid')
		var = np.convolve(((values[len(values)-len(sma):] - sma)**2), weighs, 'valid')
		std = np.sqrt(var)

		upper_bb = sma[len(sma)-len(std):] + 2*std
		lower_bb = sma[len(sma)-len(std):] - 2*std
		
		return (sma, var, std, upper_bb, lower_bb, )

#		except ValueError as e:
#			print(e)
#			print("No data for stock " + self.name + "...")
#			pass

	def get_returns(self, observed_days, interval):
		returns = list(map(lambda x, y: round(float(y/x-1), 4), self.opens[-1::-interval], self.closes[-interval::-interval]))

		#while len(returns) < (observed_days-1)/interval:
		#returns.append(0.0)

		return returns
	
	def get_sma_bb_ratio(self):
		try:
			sma_bb_ratio = round(self.closes_lower_bb[0]/self.closes[0]-1, 4)
		except (TypeError, IndexError) as e:
			print(7)
			print(self.name, e)
			return 0.0

		return sma_bb_ratio
	
	def get_volume_inc(self):
		try:
			volume_inc = round(self.volumes_sma[0]/self.volumes_sma[1] - 1, 4)
		except (TypeError, IndexError) as e:
			print(6)
			print(self.name, e)
			return 0.0

		return volume_inc
	
	def get_dd(self):
	
		try:
			d = round((self.closes_sma[0] - self.closes_sma[1]), 4)
			dd = round((self.closes_sma[0] - 2*self.closes_sma[1] + self.closes_sma[2]), 4)
		except (TypeError, IndexError) as e:
			print(5)
			print(self.name, e)
			return 0.0
		
		return dd
	
	def get_long_term_progress(self):
	
		try:
			progress = round((self.closes_sma[0] - self.closes_sma[2]), 4)
		except (TypeError, IndexError) as e:
			print(self.name, e)
			return 0.0
				
		return progress
	
	def hammer(self):
		try:
			if self.lows[0] < self.opens[0]*0.98 and self.closes[0] > self.opens[0]*0.99:
				return 1
			else:
				return 0
		except (TypeError, IndexError) as e:
			print(self.name, e)
			return 0


	def fetch_intraday_data(self, run_no):
		
		html_file = urllib.request.urlopen("https://www.netfonds.no/quotes/tradedump.php?paper=" + self.name + ".OSE&csv_format=txt")
		html_txt = html_file.read().html_txt.split(b"\n")
		

		if run_no == 0:
			try:
				for (i,l) in enumerate(html_txt):
					if i >= 1:
						
						line = html_txt[i].split(b"\t")
						
						#self.time.append(float(line[0].decode("UTF-8")))
						self.prices.append(float(line[1].decode("UTF-8")))
						self.volumes.append(float(line[2].decode("UTF-8")))
			except (ValueError, IndexError) as e:
				pass
			
			self.no_of_lines = i+1
			self.prev_no_of_lines = self.no_of_lines
		
		else:
			try:
				for (i, line) in enumerate(html_txt):
					pass
		
				self.no_of_lines = i+1
				lines_to_read = self.no_of_lines - self.prev_no_of_lines
				self.prev_no_of_lines = self.no_of_lines
			
				for i in range(1,lines_to_read):
					#self.time.insert(i, float(line[0].decode("UTF-8")))
					self.prices.insert(i, float(line[1].decode("UTF-8")))
					self.volumes.insert(i, float(line[2].decode("UTF-8")))
			
			except (ValueError, IndexError) as e:
				pass
			
		try:
			self.prices_sma, self.prices_var, self.prices_std, self.prices_upper_bb, self.prices_lower_bb = self.compute_bb(self.prices, 20)
			self.volumes_sma, self.volumes_var, self.volumes_std, self.volumes_upper_bb, self.volumes_lower_bb = self.compute_bb(self.volumes, 20)
		except TypeError as e:
			print(e)
			pass
		print(self.prices_var)
		print(self.prices_std)
	
	

	def fetch_historic_data(self, run_no, observed_days):
		
		try:
			history_html = urllib.request.urlopen("http://www.netfonds.no/quotes/paperhistory.php?paper=" + str(self.name) + ".OSE&csv_format=txt", None, 5)
			history_txt = history_html.read().split(b"\n")
			
			
			if run_no == 0:
				try:
					for i in range(1,1+observed_days):
						
						

						line = history_txt[i].split(b"\t")

	#self.dates.append(float(line[0].decode("UTF-8")))
						self.opens.append(float(line[3].decode("UTF-8")))
						self.highs.append(float(line[4].decode("UTF-8")))
						self.lows.append(float(line[5].decode("UTF-8")))
						self.closes.append(float(line[6].decode("UTF-8")))
						self.volumes.append(float(line[7].decode("UTF-8")))
						self.values.append(float(line[8].decode("UTF-8")))
			
				except IndexError as e:
					print(e)
					return e
					pass
				
			else:
				try:
					line = history_txt[1-run_no].split(b"\t")

	#				self.dates.pop()
	#				self.dates.insert(0, float(line[0].decode("UTF-8")))
					self.opens.pop()
					self.opens.insert(0, float(line[3].decode("UTF-8")))
					self.highs.pop()
					self.highs.insert(0, float(line[4].decode("UTF-8")))
					self.lows.pop()
					self.lows.insert(0, float(line[5].decode("UTF-8")))
					self.closes.pop()
					self.closes.insert(0, float(line[6].decode("UTF-8")))
					self.volumes.pop()
					self.volumes.insert(0, float(line[7].decode("UTF-8")))
					self.values.pop()
					self.values.insert(0, float(line[8].decode("UTF-8")))
				except (ValueError, IndexError):
					print(2)
					print(e)
					pass


			try:
				self.opens_sma, self.opens_var, self.opens_std, self.opens_upper_bb, self.opens_lower_bb = self.compute_bb(self.opens, 20)
				self.highs_sma, self.highs_var, self.highs_std, self.highs_upper_bb, self.highs_lower_bb = self.compute_bb(self.highs, 20)
				self.lows_sma, self.lows_var, self.lows_std, self.lows_upper_bb, self.lows_lower_bb = self.compute_bb(self.lows, 20)
				self.closes_sma, self.closes_var, self.closes_std, self.closes_upper_bb, self.closes_lower_bb = self.compute_bb(self.closes, 20)
				self.volumes_sma, self.volumes_var, self.volumes_std, self.volumes_upper_bb, self.volumes_lower_bb = self.compute_bb(self.volumes, 20)
				self.values_sma, self.values_var, self.values_std, self.values_upper_bb, self.values_lower_bb = self.compute_bb(self.values, 20)
			except TypeError as e:
				print(1)
				print(self.name, e)
				return e
				pass
		except:

			print("Timeout: could not access page")
			pass
#


		return None

