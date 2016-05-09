import optimization as opt
from stock_market import StockMarket
from plot_stockdata import plot, plot_learning_curve, plot_markowitz
import nn
import numpy as np
import random

import stock_algorithms


if __name__ == '__main__':
	
	N = 100			# Number of days observed
	N_c = 6			# Number of commodities
	interval = 5	# Number of days over at which returns are being calculated
	mu = 1.0   		# Weigth factor risk tolerance
	rho = 0.0		# Weight factor on price to bollinger band ratio
	phi = 0.0		# Weight factor on volume increase
	kappa = 1.0		# Weight factor on returns
	
	lambda_reg =1
	
	FEATURES = 6
	INTERVAL = 5
	EX_PER_STOCK = 60
	
	requested_stocks = 'all'#['STL', 'REC', 'SDRL', 'SAS-NOK', 'ATLA-NOK', 'BIONOR', 'HAVI', 'SBX', 'FUNCOM', 'TEL', 'ARCHER', 'NAVA', 'EMGS', 'MHG', 'FRO', 'PRS', 'ODFB', 'AKVA', 'SONG', 'NAS']
	
	OSEBX = StockMarket('OSEBX', requested_stocks)
	my_portfolio = stock_algorithms.Portfolio()
	
	TS = nn.TrainingSet(OSEBX, FEATURES, EX_PER_STOCK, INTERVAL)

	X, y, X_cv, y_cv, X_test, y_test, X_p, stocks = TS.get_data_set()
	
	NN = nn.NeuralNetwork(FEATURES*INTERVAL, 12, 12)

	NN.input_data(X, y, X_cv, y_cv, X_test, y_test)
	NN.minimize_cost(X, y, lambda_reg)

	
	NN.test_nn(X_test, y_test, lambda_reg, theta = None, type = "test")
	run_no = 0

	while run_no < 24:

	
		X_p, stocks, y_p, prediction = NN.predict(X_p, stocks)
		
		promising_stocks = [stock for stock, prob in prediction]

		costs_train, costs_cv, costs_test, accuracies_train, f1s_train, accuracies_cv, f1s_cv, accuracies_test, f1s_test = NN.get_mod_ver_data()
#	
		plot_learning_curve(costs_train, costs_cv, costs_test, accuracies_train, f1s_train, accuracies_cv, f1s_cv, accuracies_test, f1s_test, run_no)
#	
		for stock, prob in prediction:
			print(stock.name, prob)

		print("Day no.: ", run_no)

		# Get historic stock and commidity returns
		returns = OSEBX.get_returns(promising_stocks, N, interval)

		# Get volume movements over last two days
#		volume_movement = np.array(OSEBX.get_volume_incs(requested_stocks))#np.concatenate((OSEBX.get_volume_incs(requested_stocks), np.zeros((1,6))[0]))#
		# Get recent second derivatives
#		dds = np.array(OSEBX.get_dds(requested_stocks))#np.concatenate((OSEBX.get_dds(requested_stocks), np.zeros((1,6))[0]))#

		# Get ratio of moving average and lower bollinger band for stocks
		sma_bb_ratio = np.array(OSEBX.get_sma_bb_ratios(promising_stocks))#np.concatenate((OSEBX.get_sma_bb_ratios(requested_stocks), np.zeros((1,6))[0]))#

		# Get historic stock covariances
		cov_matrix = OSEBX.get_stock_covariances(returns)

#		filter = OSEBX.filter(requested_stocks, [stock for stock, prob in promising_stocks])

		# Generate optimization problem
		P, q, G, h, A, b, means, stds = opt.generate_ProblemMatrices(returns, cov_matrix, sma_bb_ratio, rho, phi, kappa)

		# Solve optimization problem
		optimized_portfolio = opt.optimize_portfolio(OSEBX, promising_stocks, P, q, G, h, A, b, means, stds, sma_bb_ratio, mu)

		stock_algorithms.trade(OSEBX, my_portfolio, optimized_portfolio, 'historic', run_no)
#
		OSEBX.fetch_benchmark(run_no)
#
		if run_no == 10 or run_no == 20 or run_no == 30 or run_no == 40:
			plot(my_portfolio.earnings, OSEBX.benchmark, run_no)
#
#
		run_no+=1
#
		OSEBX.fetch_stock_data(requested_stocks, run_no, 'historic', N)
#
		X_p, stocks = TS.generate_data_set(OSEBX, FEATURES, 1, INTERVAL, type = "predict")


#	plot(my_portfolio.earnings, OSEBX.benchmark, run_no)
#
	opt_means = [stock[2] for stock in optimized_portfolio]
	opt_stds = [stock[3] for stock in optimized_portfolio]
#
	plot_markowitz(optimized_portfolio, means, stds, opt_means, opt_stds)
	plot_markowitz(optimized_portfolio, means[0:-N_c], stds[0:-N_c], opt_means, opt_stds)

