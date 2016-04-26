import urllib.request
import resource
import re
import numpy as np

from stocks import Stock
from commodities import Commodity

class StockMarket:
	def __init__(self, name, requested_stocks):
		
		self.name = name
		self.benchmark = []
		self.stocks = []
		self.commodities = []
		self.portfolio = []
		self.sold_stocks = []
		
		self.fetch_stock_data(requested_stocks, 0, 'historic', 100)
		self.fetch_commodity_data(0, 100)
	
	def fetch_benchmark(self, run_no):
		history_html = urllib.request.urlopen("http://www.netfonds.no/quotes/paperhistory.php?paper=OSEBX.OSE&csv_format=txt", None, 5).read()
		history_txt = history_html.split(b"\n")
		
		line = history_txt[50-run_no].split(b"\t")
		
		
		if run_no == 0:
			benchmark = float(line[6].decode("UTF-8"))
			self.benchmark_volume = 100000/benchmark
			share = benchmark*self.benchmark_volume
		
			self.benchmark.append(share)
		
		else:
			benchmark = float(line[6].decode("UTF-8"))
			
			share = benchmark*self.benchmark_volume
			
			self.benchmark.append(share)


	def get_returns(self, stocks, observed_days, interval):
		
		# Get historic stock returns
#		if stocks != 'all':
#			returns = []
#			for stock in stocks:
#				stock_return = self.stocks[[s.name for s in self.stocks].index(stock)].get_returns(observed_days, interval)
#				returns.append(stock_return)
#
#		else:
#			dds = [stock.get_dd() for stock in self.stocks]
#			returns = [stock.get_returns(observed_days, interval) for stock in self.stocks]# if stock.volumes_sma != [] and stock.volumes_sma[0] >= 200000]
#
#		commodity_returns = [commodity.get_returns(observed_days, interval) for commodity in self.commodities]
#		
#		for com in commodity_returns:
#			returns.append(com)

		returns = []
		
		for stock in stocks:
			
			stock_profits = stock.get_returns(observed_days, interval)
		
			returns.append(stock_profits)

		return returns
	
	def get_sma_bb_ratios(self, stocks):
	
		# Get historic stock returns
#		if stocks != 'all':
#			ratios = []
#			for stock in stocks:
#				stock_ratio = self.stocks[[s.name for s in self.stocks].index(stock)].get_sma_bb_ratio()
#				ratios.append(stock_ratio)
#
#		else:
#			dds = [stock.get_dd() for stock in self.stocks]
#			ratios = [stock.get_sma_bb_ratio() for stock in self.stocks]# if stock.volumes_sma != [] and stock.volumes_sma[0] >= 200000]

		ratios = []

		for stock in stocks:
		
			stock_ratio = stock.get_sma_bb_ratio()
		
			ratios.append(stock_ratio)

		return ratios

	def get_stock_covariances(self, returns):
		
#		if stocks == 'all':
#			dds = [stock.get_dd() for stock in self.stocks]
#			data = [stock.get_returns(observed_days, interval) for stock in self.stocks]# if stock.volumes_sma != [] and stock.volumes_sma[0] >= 200000]
#		else:
#			data = []
#			for stock in stocks:
#				stock_return = self.stocks[[s.name for s in self.stocks].index(stock)].get_returns(observed_days, interval)
#				data.append(stock_return)
#
#		commodity_data = [commodity.get_returns(observed_days, interval) for commodity in self.commodities]
#		
#		for com in commodity_data:
#			data.append(com)

		cov_matrix = np.cov(returns)

		return cov_matrix

	def get_volume_incs(self, stocks):

		if stocks == 'all':

			volume_incs = [stock.get_volume_inc() for stock in self.stocks]# if stock.volumes_sma != [] and stock.volumes_sma[0] >= 200000]

		else:
			volume_incs = []
			for stock in stocks:
				volume_incs.append(self.stocks[[s.name for s in self.stocks].index(stock)].get_volume_inc())

		return volume_incs
	
	def get_dds(self, stocks):
	
		if stocks == 'all':
			dds = [stock.get_dd() for stock in self.stocks]
		else:
			dds = []
			for stock in stocks:
				dds.append(self.stocks[[s.name for s in self.stocks].index(stock)].get_dd())

		return dds
	
	def filter(self, stocks, promising_stocks):
		dds = self.get_dds(stocks)
		
		filter = []
		print(promising_stocks)
		
		if stocks == 'all':
			for i, stock in zip(range(len(self.stocks)), self.stocks):
#				if stock.volumes_sma == [] or stock.volumes_sma[0] <= 100000 or stock.hammer() == 0:
				if stock not in promising_stocks or stock.volumes_sma == [] or stock.volumes_sma[0] <= 100000:
					filter.append(self.stocks.index(stock))
		else:
			for i, stock in zip(range(len(stocks)), stocks):
#				if self.stocks[[s.name for s in self.stocks].index(stock)].volumes_sma == [] or self.stocks[[s.name for s in self.stocks].index(stock)].volumes_sma[0] <= 200000 or self.stocks[[s.name for s in self.stocks].index(stock)].get_volume_inc() < 0.0 or not self.stocks[[s.name for s in self.stocks].index(stock)].hammer():
#					filter.append([s.name for s in self.stocks].index(stock))
				if self.stocks[[s.name for s in self.stocks].index(stock)] not in promising_stocks:
					filter.append(stocks.index(stock))
					
		return filter
	


	def fetch_commodity_data(self, run_no, observed_days):
#		try:


		html_file = urllib.request.urlopen("http://www.netfonds.no/quotes/kurs.php?exchange=GTIS", None, 5)
		html_text = html_file.read()
		
		pattern = re.compile(b'<td class="left">(.+?)...</td>')
		
		commodities = re.findall(pattern, html_text)

		commodities = [commodity.decode("UTF-8") for commodity in commodities[1::2]]
		
		for (i, com) in enumerate(commodities):
			if i == 0 or i == 5:
				commodity = Commodity(com, 'energy')
			else:
				commodity = Commodity(com, 'metal')
			commodity.get_commodity_prices(run_no, observed_days)
			self.commodities.append(commodity)

#		except:
#			print("Could not establish connection to commodities prices")
#			pass


	def fetch_stock_data(self, stocks, run_no, data_depth, observed_days):
		
		no_go_stocks = ['OBTEST0' + str(i) for i in range(10)]
		
#		try:

		html_text = urllib.request.urlopen("http://www.netfonds.no/quotes/kurs.php", None, 5).read()
		stock_market = re.findall(re.compile(b'<td name="ju.last.(.+?).OSE">(.+?)</td>'), html_text)
		
		if stocks != 'all':
			stock_market = [stock for stock in stock_market if stock[0].decode("UTF-8") in stocks]

		if run_no == 0:
			for (stock, price) in stock_market:
				if stock.decode("UTF-8") not in no_go_stocks:
					self.stocks.append(Stock(stock.decode("UTF-8")))
		
		else:
			pass

		print("Fetching data...")

		error_stocks = []

		for stock in self.stocks:
			if data_depth == 'day':
				stock.fetch_intraday_data(run_no)
			elif data_depth == 'historic':
				
				error = stock.fetch_historic_data(run_no, observed_days)
				if error != None:
					error_stocks.append(stock)

		self.stocks = [stock for stock in self.stocks if not stock in error_stocks]		

		print("Finished fetching data")
		
#		except:
#			print("URLError: could not establish connection")

