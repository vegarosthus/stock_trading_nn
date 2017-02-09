import numpy as np

class TrainingSet():
	
	def __init__(self, stock_market, NO_OF_FEATURES, EX_PER_STOCK, INTERVAL):
		
		self.min = [0]*NO_OF_FEATURES
		self.max = [0]*NO_OF_FEATURES
		self.mean = [0]*NO_OF_FEATURES
		self.var = [0]*NO_OF_FEATURES
		
		print("Preprocessing data...")
		self.X_train, self.y_train = self.generate_data_set(stock_market, NO_OF_FEATURES, EX_PER_STOCK, INTERVAL)
		
		self.X_cv, self.y_cv, self.X_test, self.y_test = self.cross_validation(stock_market, NO_OF_FEATURES*INTERVAL, EX_PER_STOCK)
		
		self.X_predict, self.stocks = self.generate_data_set(stock_market, NO_OF_FEATURES, 1, INTERVAL, type = "predict")
		
		print("Finished preprocessing data")
	
	def get_data_set(self):
		
		return self.X_train, self.y_train, self.X_cv, self.y_cv, self.X_test, self.y_test, self.X_predict, self.stocks
	
	def dim_reduct(self, X):
		
		sigma = np.cov(X, rowvar=0)
		U,S,V = np.linalg.svd(sigma)
		stds = np.diag(S)
		
		k = 0
		var_ret = 0.0
		
		while var_ret <= 0.99:
			k += 1
			var_ret = np.sum(stds[:k]) / np.sum(stds)
		
		Z = np.dot(X, U[:,0:k])
		
		return Z, k

	def feature_norm(self, array, feature, type = None):
		
		#Normalize array elements by subtracting the mean and dividing by standard deviation
		if type == "training":
			
			self.min[feature] = np.min(array)
			self.max[feature] = np.max(array)
				
			self.mean[feature] = np.mean(array)
			self.var[feature] = np.var(array)
			
		norm_array = (array-self.min[feature])/(self.max[feature]-self.min[feature])
		#norm_array = (array-self.mean[feature])/self.var[feature]
			
		return norm_array
	
	
	def generate_data_set(self, stock_market, NO_OF_FEATURES, EX_PER_STOCK, INTERVAL, type = "training"):
		
		X = np.zeros((len(stock_market.stocks)*EX_PER_STOCK, INTERVAL*NO_OF_FEATURES))
		y = np.zeros((len(stock_market.stocks)*EX_PER_STOCK, 1))
		
		stocks = []
		
		benchmark_open_incs_5 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock_market.benchmark['open_sma_5'], stock_market.benchmark['open_sma_5'][1:]))
		benchmark_close_incs_5 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock_market.benchmark['close_sma_5'], stock_market.benchmark['close_sma_5'][1:]))
		benchmark_high_incs_5 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock_market.benchmark['high_sma_5'], stock_market.benchmark['high_sma_5'][1:]))
		benchmark_low_incs_5 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock_market.benchmark['low_sma_5'], stock_market.benchmark['low_sma_5'][1:]))
		benchmark_value_incs_5 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock_market.benchmark['value_sma_5'], stock_market.benchmark['value_sma_5'][1:]))
		
		commodity_prices = {}
		
		for commodity in stock_market.commodities:
			name = commodity.name
			dict_entry_open = name + '_open_incs_5'
			dict_entry_close = name + '_close_incs_5'
			dict_entry_high = name + '_high_incs_5'
			dict_entry_low = name + '_low_incs_5'
			
			commodity_prices[dict_entry_open] = list(map(lambda x, y: round((y-x)/x * 100, 4), commodity.prices['open_sma_5'], commodity.prices['open_sma_5'][1:]))
			commodity_prices[dict_entry_close] = list(map(lambda x, y: round((y-x)/x * 100, 4), commodity.prices['close_sma_5'], commodity.prices['close_sma_5'][1:]))
			commodity_prices[dict_entry_high] = list(map(lambda x, y: round((y-x)/x * 100, 4), commodity.prices['high_sma_5'], commodity.prices['high_sma_5'][1:]))
			commodity_prices[dict_entry_low] = list(map(lambda x, y: round((y-x)/x * 100, 4), commodity.prices['low_sma_5'], commodity.prices['low_sma_5'][1:]))
		
		for stock, i in zip(stock_market.stocks, range(len(stock_market.stocks))):
			
			open_incs_5 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['open_sma_5'], stock.stock_data['open_sma_5'][1:]))
			close_incs_5 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['close_sma_5'], stock.stock_data['close_sma_5'][1:]))
			high_incs_5 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['high_sma_5'], stock.stock_data['high_sma_5'][1:]))
			low_incs_5 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['low_sma_5'], stock.stock_data['low_sma_5'][1:]))
			volume_incs_5 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['volume_sma_5'], stock.stock_data['volume_sma_5'][1:]))
			value_incs_5 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['value_sma_5'], stock.stock_data['value_sma_5'][1:]))
			
			open_incs_10 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['open_sma_10'], stock.stock_data['open_sma_10'][1:]))
			close_incs_10 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['close_sma_10'], stock.stock_data['close_sma_10'][1:]))
			high_incs_10 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['high_sma_10'], stock.stock_data['high_sma_10'][1:]))
			low_incs_10 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['low_sma_10'], stock.stock_data['low_sma_10'][1:]))
			volume_incs_10 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['volume_sma_10'], stock.stock_data['volume_sma_10'][1:]))
			value_incs_10 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['value_sma_10'], stock.stock_data['value_sma_10'][1:]))
			
			open_incs_20 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['open_sma_20'], stock.stock_data['open_sma_20'][1:]))
			close_incs_20 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['close_sma_20'], stock.stock_data['close_sma_20'][1:]))
			high_incs_20 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['high_sma_20'], stock.stock_data['high_sma_20'][1:]))
			low_incs_20 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['low_sma_20'], stock.stock_data['low_sma_20'][1:]))
			volume_incs_20 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['volume_sma_20'], stock.stock_data['volume_sma_20'][1:]))
			value_incs_20 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['value_sma_20'], stock.stock_data['value_sma_20'][1:]))
			
			open_incs_50 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['open_sma_50'], stock.stock_data['open_sma_50'][1:]))
			close_incs_50 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['close_sma_50'], stock.stock_data['close_sma_50'][1:]))
			high_incs_50 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['high_sma_50'], stock.stock_data['high_sma_50'][1:]))
			low_incs_50 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['low_sma_50'], stock.stock_data['low_sma_50'][1:]))
			volume_incs_50 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['volume_sma_50'], stock.stock_data['volume_sma_50'][1:]))
			value_incs_50 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['value_sma_50'], stock.stock_data['value_sma_50'][1:]))
			
			open_incs_100 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['open_sma_100'], stock.stock_data['open_sma_100'][1:]))
			close_incs_100 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['close_sma_100'], stock.stock_data['close_sma_100'][1:]))
			high_incs_100 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['high_sma_100'], stock.stock_data['high_sma_100'][1:]))
			low_incs_100 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['low_sma_100'], stock.stock_data['low_sma_100'][1:]))
			volume_incs_100 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['volume_sma_100'], stock.stock_data['volume_sma_100'][1:]))
			value_incs_100 = list(map(lambda x, y: round((y-x)/x * 100, 4), stock.stock_data['value_sma_100'], stock.stock_data['value_sma_100'][1:]))
			
			
			if type == "training":
				for j in range(EX_PER_STOCK):

					X[EX_PER_STOCK*i+j,:] = np.concatenate((open_incs_5[INTERVAL*j:(j+1)*INTERVAL], close_incs_5[INTERVAL*j:(j+1)*INTERVAL], high_incs_5[INTERVAL*j:(j+1)*											INTERVAL], low_incs_5[INTERVAL*j:(j+1)*INTERVAL], volume_incs_5[INTERVAL*j:(j+1)*INTERVAL],															value_incs_5[INTERVAL*j:(j+1)*INTERVAL],
															open_incs_10[INTERVAL*j:(j+1)*INTERVAL], close_incs_10[INTERVAL*j:(j+1)*INTERVAL], high_incs_10[INTERVAL*j:(j+1)*INTERVAL], low_incs_10[INTERVAL*j:(j+1)*INTERVAL], volume_incs_10[INTERVAL*j:(j+1)*INTERVAL],value_incs_10[INTERVAL*j:(j+1)*INTERVAL],
															open_incs_20[INTERVAL*j:(j+1)*INTERVAL], close_incs_20[INTERVAL*j:(j+1)*INTERVAL], high_incs_20[INTERVAL*j:(j+1)*INTERVAL], low_incs_20[INTERVAL*j:(j+1)*INTERVAL], volume_incs_20[INTERVAL*j:(j+1)*INTERVAL], value_incs_20[INTERVAL*j:(j+1)*INTERVAL],
															open_incs_50[INTERVAL*j:(j+1)*INTERVAL], close_incs_50[INTERVAL*j:(j+1)*INTERVAL], high_incs_50[INTERVAL*j:(j+1)*INTERVAL], low_incs_50[INTERVAL*j:(j+1)*INTERVAL], volume_incs_50[INTERVAL*j:(j+1)*INTERVAL], value_incs_50[INTERVAL*j:(j+1)*INTERVAL],
															open_incs_100[INTERVAL*j:(j+1)*INTERVAL], close_incs_100[INTERVAL*j:(j+1)*INTERVAL], high_incs_100[INTERVAL*j:(j+1)*INTERVAL], low_incs_100[INTERVAL*j:(j+1)*INTERVAL], volume_incs_100[INTERVAL*j:(j+1)*INTERVAL], value_incs_100[INTERVAL*j:(j+1)*INTERVAL],
															benchmark_open_incs_5[INTERVAL*j:(j+1)*INTERVAL], benchmark_close_incs_5[INTERVAL*j:(j+1)*INTERVAL], benchmark_high_incs_5[INTERVAL*j:(j+1)*INTERVAL], benchmark_low_incs_5[INTERVAL*j:(j+1)*INTERVAL], benchmark_value_incs_5[INTERVAL*j:(j+1)*INTERVAL],
															commodity_prices['C-EBROUS_open_incs_5'][INTERVAL*j:(j+1)*INTERVAL], commodity_prices['C-EBROUS_close_incs_5'][INTERVAL*j:(j+1)*INTERVAL], commodity_prices['C-EBROUS_high_incs_5'][INTERVAL*j:(j+1)*INTERVAL], commodity_prices['C-EBROUS_low_incs_5'][INTERVAL*j:(j+1)*INTERVAL],
															commodity_prices['C-SXAUUS_open_incs_5'][INTERVAL*j:(j+1)*INTERVAL], commodity_prices['C-SXAUUS_close_incs_5'][INTERVAL*j:(j+1)*INTERVAL], commodity_prices['C-SXAUUS_high_incs_5'][INTERVAL*j:(j+1)*INTERVAL], commodity_prices['C-SXAUUS_low_incs_5'][INTERVAL*j:(j+1)*INTERVAL],
															commodity_prices['C-SXPDUS_open_incs_5'][INTERVAL*j:(j+1)*INTERVAL], commodity_prices['C-SXPDUS_close_incs_5'][INTERVAL*j:	(j+1)*INTERVAL], commodity_prices['C-SXPDUS_high_incs_5'][INTERVAL*j:(j+1)*INTERVAL], commodity_prices['C-SXPDUS_low_incs_5'][INTERVAL*j:(j+1)*INTERVAL],
															commodity_prices['C-SXPTUS_open_incs_5'][INTERVAL*j:(j+1)*INTERVAL], commodity_prices['C-SXPTUS_close_incs_5'][j:INTERVAL+j], commodity_prices['C-SXPTUS_high_incs_5'][INTERVAL*j:(j+1)*INTERVAL], commodity_prices['C-SXPTUS_low_incs_5'][INTERVAL*j:(j+1)*INTERVAL],
															commodity_prices['C-SXAGUS_open_incs_5'][INTERVAL*j:(j+1)*INTERVAL], commodity_prices['C-SXAGUS_close_incs_5'][INTERVAL*j:(j+1)*INTERVAL], commodity_prices['C-SXAGUS_high_incs_5'][INTERVAL*j:(j+1)*INTERVAL], commodity_prices['C-SXAGUS_low_incs_5'][INTERVAL*j:(j+1)*INTERVAL],
															commodity_prices['C-EWTIUS_open_incs_5'][INTERVAL*j:(j+1)*INTERVAL], commodity_prices['C-EWTIUS_close_incs_5'][INTERVAL*j:(j+1)*INTERVAL], commodity_prices['C-EWTIUS_high_incs_5'][INTERVAL*j:(j+1)*INTERVAL], commodity_prices['C-EWTIUS_low_incs_5'][INTERVAL*j:(j+1)*INTERVAL]
															))
						
															
					if ((stock.stock_data['close_sma_5'][(j+1)*INTERVAL] - stock.stock_data['open_sma_5'][(j+1)*INTERVAL])/stock.stock_data['open_sma_5'][(j+1)*INTERVAL]) > 0:
						y[EX_PER_STOCK*i+j] = 1
					else:
						y[EX_PER_STOCK*i+j] = 0
			
			elif type == "predict":
				
				X[i,:] = np.concatenate((open_incs_5[-INTERVAL:], close_incs_5[-INTERVAL:], high_incs_5[-INTERVAL:], low_incs_5[-INTERVAL:], volume_incs_5[-INTERVAL:],								 value_incs_5[-INTERVAL:],
										 open_incs_10[-INTERVAL:], close_incs_10[-INTERVAL:], high_incs_10[-INTERVAL:], low_incs_10[-INTERVAL:], volume_incs_10[-INTERVAL:], value_incs_10[-INTERVAL:],
										 open_incs_20[-INTERVAL:], close_incs_20[-INTERVAL:], high_incs_20[-INTERVAL:], low_incs_20[-INTERVAL:], volume_incs_20[-INTERVAL:], value_incs_20[-INTERVAL:],
										 open_incs_50[-INTERVAL:], close_incs_50[-INTERVAL:], high_incs_50[-INTERVAL:], low_incs_50[-INTERVAL:], volume_incs_50[-INTERVAL:], value_incs_50[-INTERVAL:],
										 open_incs_100[-INTERVAL:], close_incs_100[-INTERVAL:], high_incs_100[-INTERVAL:], low_incs_100[-INTERVAL:], volume_incs_100[-INTERVAL:], value_incs_100[-INTERVAL:],
										 benchmark_open_incs_5[-INTERVAL:], benchmark_close_incs_5[-INTERVAL:], benchmark_high_incs_5[-INTERVAL:], benchmark_low_incs_5[-INTERVAL:], benchmark_value_incs_5[-INTERVAL:],
										 commodity_prices['C-EBROUS_open_incs_5'][-INTERVAL:], commodity_prices['C-EBROUS_close_incs_5'][-INTERVAL:], commodity_prices['C-EBROUS_high_incs_5'][-INTERVAL:], commodity_prices['C-EBROUS_low_incs_5'][-INTERVAL:],
										 commodity_prices['C-SXAUUS_open_incs_5'][-INTERVAL:], commodity_prices['C-SXAUUS_close_incs_5'][-INTERVAL:], commodity_prices['C-SXAUUS_high_incs_5'][-INTERVAL:], commodity_prices['C-SXAUUS_low_incs_5'][-INTERVAL:],
										 commodity_prices['C-SXPDUS_open_incs_5'][-INTERVAL:], commodity_prices['C-SXPDUS_close_incs_5'][-INTERVAL:], commodity_prices['C-SXPDUS_high_incs_5'][-INTERVAL:], commodity_prices['C-SXPDUS_low_incs_5'][-INTERVAL:],
										 commodity_prices['C-SXPTUS_open_incs_5'][-INTERVAL:], commodity_prices['C-SXPTUS_close_incs_5'][-INTERVAL:], commodity_prices['C-SXPTUS_high_incs_5'][-INTERVAL:], commodity_prices['C-SXPTUS_low_incs_5'][-INTERVAL:],
										 commodity_prices['C-SXAGUS_open_incs_5'][-INTERVAL:], commodity_prices['C-SXAGUS_close_incs_5'][-INTERVAL:], commodity_prices['C-SXAGUS_high_incs_5'][-INTERVAL:], commodity_prices['C-SXAGUS_low_incs_5'][-INTERVAL:],
										 commodity_prices['C-EWTIUS_open_incs_5'][-INTERVAL:], commodity_prices['C-EWTIUS_close_incs_5'][-INTERVAL:], commodity_prices['C-EWTIUS_high_incs_5'][-INTERVAL:], commodity_prices['C-EWTIUS_low_incs_5'][-INTERVAL:]
										 ))
										 
				stocks.append(stock)

			#Normalize data
			#		for i in range(NO_OF_FEATURES):
			#			X[:,i*INTERVAL:(i+1)*INTERVAL] = self.feature_norm(X[:,i*INTERVAL:(i+1)*INTERVAL], i, type)
			
		if type == "predict":
			return X, stocks
						
		return X, y

	def cross_validation(self, stock_market, K, EX_PER_STOCK):
	
		number_of_test_items = np.floor(len(stock_market.stocks)*EX_PER_STOCK*0.4)
		
		random_indices = np.random.randint(len(stock_market.stocks)*EX_PER_STOCK-1, None, number_of_test_items)
		
		cv_indices = random_indices[ : np.floor(np.size(random_indices)/2)]
		test_indices = random_indices[np.floor(np.size(random_indices)/2) : ]
		
		X_cv = np.zeros((np.size(cv_indices), K))
		y_cv = np.zeros((np.size(cv_indices),1))
		
		X_test = np.zeros((np.size(test_indices), K))
		y_test = np.zeros((np.size(test_indices),1))
		
		for cv_index, test_index, i in zip(cv_indices, test_indices, range(np.size(cv_indices))):
			X_cv[i,:] = self.X_train[cv_index,:]
			y_cv[i] = self.y_train[cv_index]
			
			X_test[i,:] = self.X_train[test_index,:]
			y_test[i] = self.y_train[test_index]
		
		self.X_train = np.delete(self.X_train, random_indices, 0)
		self.y_train = np.delete(self.y_train, random_indices, 0)
		
		return X_cv, y_cv, X_test, y_test
