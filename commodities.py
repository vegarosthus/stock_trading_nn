import urllib.request
import numpy as np
import random

class Commodity:
	
	def __init__(self, name, type):
		
		self.name = name
		self.type = type
		
		self.prices_open = []
		self.prices_close = []
	
			
	def get_commodity_prices(self, run_no, observed_days):
		
		#		try:
		
		if self.type == 'energy':
			history_html = urllib.request.urlopen("http://www.netfonds.no/quotes/paperhistory.php?paper=" + self.name + "DBR-SP.GTIS&csv_format=txt", None, 5)
		elif self.type == 'metal':
			history_html = urllib.request.urlopen("http://www.netfonds.no/quotes/paperhistory.php?paper=" + self.name + "DOZ-SP.GTIS&csv_format=txt", None, 5)

		
		history_txt = history_html.read()
		history_txt = history_txt.split(b"\n")
		
		if run_no == 0:
			try:
				for i in range(1,observed_days):
					
					line = history_txt[i].split(b"\t")
					self.prices_open.append(float(line[3].decode("UTF-8")))
					self.prices_close.append(float(line[6].decode("UTF-8")))
		
			except (ValueError, IndexError) as e:
				#print(e)
				pass
		else:
			try:
				line = history_txt[50-run_no].split(b"\t")
				self.prices_open.pop()
				self.prices_open.insert(0, float(line[0].decode("UTF-8")))
				self.prices_close.pop()
				self.prices_close.insert(0, float(line[3].decode("UTF-8")))
			except (ValueError, IndexError):
				pass

		#except:
		#			print("Timeout: could not access page")
		#			pass

	def get_returns(self, observed_days, interval):
		
		returns = list(map(lambda x, y: round(float(y/x-1), 4), self.prices_open[-1::-interval], self.prices_close[-interval::-interval]))

		while len(returns) < (observed_days-1)/interval:
			returns.append(0.0)

		return returns