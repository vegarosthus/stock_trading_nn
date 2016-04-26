import quicksort
from math import floor
import optimization

def find_possible_purchases(stock_market, portfolio, data):
	
	possible_purchases = []
	
	if data == 'historic':
		for stock in stock_market.stocks:
			try:
				if (stock.volumes_sma[0] > 100000) and (stock.name not in [portfolio_stock[0].name for portfolio_stock in portfolio.stocks]) and (stock.closes[0] < stock.closes_lower_bb[0]): #if close price below lower Bollinger Band, put stock in portfolio and log "buying price"
					possible_purchases.append(stock.name)#(stock, ((stock.closes_sma[0] - stock.closes[0])/stock.closes_sma[0])*100))
			except IndexError as e:
				pass
	elif data == 'intraday':
		for stock in stock_market.stocks:
			try:
				if (stock.name not in [portfolio_stock[0].name for portfolio_stock in portfolio.stocks]) and (stock.prices[0] < stock.prices_lower_bb[0]): #if close price below lower Bollinger Band, put stock in portfolio and log "buying price"
					possible_purchases.append((stock, ((stock.prices_sma[0] - stock.prices[0])/stock.prices_sma[0])*100))
			except IndexError as e:
				pass
	

#	possible_purchases = quicksort.quicksort(possible_purchases, 0, len(possible_purchases)-1, 1)
#	possible_purchases.reverse()
#	
#	print("\nPossible purchases")
#	for (stock, weight) in possible_purchases:
#		print(stock.name, weight)

	return possible_purchases


def trade(stock_market, portfolio, reccomended_stocks, data, run_no):

	MAX_TRANSACTION = 25000
		
		#possible_purchases = find_possible_purchases(stock_market, portfolio, data)

	portfolio.sell(stock_market, run_no, data)

#	if run_no == 1 or run_no % 6 == 0 or run_no == 49:
	portfolio.invest(reccomended_stocks, run_no, data)

	portfolio.print_transaction_info(stock_market, data)


class Portfolio:
	def __init__(self):
		self.stocks = []
		self.transactions =[]
		self.earnings = []
		
		self.balance = 100000.0
		self.max_no_stocks = 5

	def	calculate_earnings(self, unrealized_profit):
		locked_capita = 0.0
		for (bought, buying_price, volume) in self.stocks:
			locked_capita += buying_price*volume
		earnings = locked_capita + self.balance + unrealized_profit

		return earnings


	def invest(self, possible_purchases, run_no, data):
		
		volumes = []
		
		for (stock, weight, mean, std) in possible_purchases:
			if len(self.stocks) < self.max_no_stocks:
				if data == 'historic':
					volumes.append((self.balance*weight)/stock.closes[0])
				elif data == 'intraday':
					volumes.append(self.balance*weight)/stock.prices[0]


		for volume, (stock, weight, mean, std) in zip(volumes, possible_purchases):
			if data == 'historic':
				#if volume*stock.closes[0] <= self.balance:
				if stock not in [investment[0] for investment in self.stocks]:
					self.stocks.append((stock, stock.closes[0], floor(volume)))
					self.trans(stock, 'buy', stock.closes[0], 0, floor(volume), run_no)

			elif data == 'intraday':
				#if volume*stock.prices[0] <= self.balance:
					
				self.stocks.append((stock, stock.prices[0], floor(volume)))
				self.trans(stock, 'buy', stock.prices[0], 0, floor(volume), run_no)


	def sell(self, stock_market, run_no, data):
		
		for (self_stock, buying_price, buying_volume) in self.stocks:
			try:
				if run_no % 6 == 0 or run_no == 49:
					#at the end of each 5 day period, sell portfolio
					self.trans(self_stock, 'sale', buying_price, stock_market.stocks[[stock for stock in stock_market.stocks].index(self_stock)].closes[0], buying_volume, run_no)
					self.stocks.pop([stock[0].name for stock in self.stocks].index(self_stock.name))
				else:
					if data == 'historic':
						if stock_market.stocks[[stock for stock in stock_market.stocks].index(self_stock)].closes[0] <= buying_price*0.99:#if stock price falls bellow 1% of buying price, sell stock and log the "selling price"
							self.trans(self_stock, 'sale', buying_price, stock_market.stocks[[stock for stock in stock_market.stocks].index(self_stock)].closes[0], buying_volume, run_no)
							self.stocks.pop([stock[0].name for stock in self.stocks].index(self_stock.name))
						
						elif stock_market.stocks[[stock for stock in stock_market.stocks].index(self_stock)].closes[0] >= stock_market.stocks[[stock for stock in stock_market.stocks].index(self_stock)].closes_sma[0]:#if stock's closing value rises above the moving average price for that stock, sell stock and log "selling price"
							self.trans(self_stock, 'sale', buying_price, stock_market.stocks[[stock for stock in stock_market.stocks].index(self_stock)].closes[0], buying_volume, run_no)
							self.stocks.pop([stock[0].name for stock in self.stocks].index(self_stock.name))
								
					if data == 'intraday':
						if stock_market.stocks[[stock for stock in stock_market.stocks].index(self_stock)].prices[0] <= buying_price*0.98:#if stock price falls bellow 2% of buying price, sell stock and log the "selling price"
							self.trans(self_stock, 'sale', buying_price, stock_market.stocks[[stock for stock in stock_market.stocks].index(self_stock)].prices[0], buying_volume, run_no)
							self.stocks.pop([stock[0].name for stock in self.stocks].index(self_stock.name))
						
						elif stock_market.stocks[[stock for stock in stock_market.stocks].index(self_stock)].prices[0] >= stock_market.stocks[[stock for stock in stock_market.stocks].index(self_stock)].prices_sma[0]:#if stock's closing value rises above the moving average price for that stock, sell stock and log "selling price"
							self.trans(self_stock, 'sale', buying_price, stock_market.stocks[[stock for stock in stock_market.stocks].index(self_stock)].prices[0], buying_volume, run_no)
							self.stocks.pop([stock[0].name for stock in self.stocks].index(self_stock.name))
		
			except (IndexError, ValueError) as e:
				print(e)
				pass

	def trans(self, stock, action, buying_price, sell_price, volume, run_no):
		if action == 'sale':
			profit = volume * (sell_price - buying_price)
			self.balance = self.balance + volume*sell_price
			price = sell_price
		else:
			profit = 0.0
			self.balance = self.balance - buying_price*volume
			price = buying_price
	
		self.transactions.append((stock, action, price, volume, profit, run_no))

	def print_transaction_info(self, stock_market, data):
		try:
			print("\nPortfolio")
			
			unrealized_profit = 0
			
			if data == 'day':
				for (bought, buying_price, buying_volume) in self.stocks:
					print(bought.name, buying_price, stock_market.stocks[[stock.name for stock in stock_market.stocks].index(bought.name)].prices[0])
					unrealized_profit += buying_volume*(stock_market.stocks[[stock.name for stock in stock_market.stocks].index(bought.name)].prices[0]-buying_price)
			elif data == 'historic':
				for (bought, buying_price, buying_volume) in self.stocks:
					print(bought.name, buying_price, stock_market.stocks[[stock.name for stock in stock_market.stocks].index(bought.name)].closes[0])
					unrealized_profit += buying_volume*(stock_market.stocks[[stock.name for stock in stock_market.stocks].index(bought.name)].closes[0]-buying_price)
			try:
				print("Unrealized profit: ", unrealized_profit)
			except UnboundLocalError:
				pass

			print("\nTransactions")
			for (stock, action, price, volume, profit, run_no) in self.transactions:
				print(stock.name, action,  price, volume, profit, run_no)
			
			print("Balance: ", self.balance)
			
			earnings = self.calculate_earnings(unrealized_profit)
			self.earnings.append(earnings)
			print("Total earnings: ", earnings)

		except ValueError as e:
			print(e)
			pass
