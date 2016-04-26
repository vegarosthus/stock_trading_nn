import cvxopt as opt
import numpy as np


def generate_ProblemMatrices(returns, covariances, ratios, rho, phi,kappa):
	
	N_c = 6
	N = len(returns)

	# Transform to numpy arrays
	r = np.array(returns)
	sigma = np.array(covariances)

	# Average return over time period
	r_avg = np.mean(r, 1)

	# Standard deviations
	if len(returns) > 1:
		stds = np.diag(sigma)
	else:
		stds = np.array(sigma)

	# Create CVXOPT matrices
	p = opt.matrix(r_avg+ratios)
	sigma = opt.matrix(np.matrix(sigma))

	# Format optimization problem
	P = sigma
	#q = opt.matrix(np.zeros((N, 1)))
	q = p

	# Minimum expected return
	r_min = 0.0

	# Inequality constraints capturing w'x >= 0 w'x <= 0.25
#	G = opt.matrix(np.concatenate((-np.eye(N), np.eye(N) ), 0))
#	h = opt.matrix(np.concatenate((np.zeros((N,1)), np.ones((N,1))*0.25), 0))

	G = -opt.matrix((np.eye(N)))  # negative n x n identity matrix
	h = opt.matrix((np.zeros((N,1))))
	
	
	# Equality constraints capturing sum(stock_weights) = 1 and excluding commodities for trade
	
#	A_sub1 = np.ones((1,N))
#	A_sub2 = []
#
#	b_sub1 = [1]
#	b_sub2 = []
#
#	for i in range(len(returns)):
#		if i in filter:
#			
#			b_sub2.append(0.0)
#			row = [0.0]*N
#			row[i] = 1.0
#			A_sub2.append(row)
#
#	if A_sub2 == []:
#
#		A = opt.matrix(A_sub1)
#		b = opt.matrix(b_sub1)
#	else:
#
#		A = opt.matrix( np.vstack(( A_sub1, A_sub2 )))
#		b = opt.matrix( np.concatenate(( b_sub1, b_sub2 )))
#
#	print(A)


	A = opt.matrix(np.ones((1,N)))
	b = opt.matrix(np.ones((1,1)))


	#A = opt.matrix(np.vstack((np.concatenate((np.transpose(np.ones(N-N_c)), np.transpose(np.zeros(N_c)))), np.delete(np.eye(N), (range(N-N_c)), 0))))
	#b = opt.matrix(np.concatenate((np.ones((1,1)), np.zeros((N_c, 1)))))

	return P, q, G, h, A, b, r_avg, stds


def optimize_portfolio(stock_market, requested_stocks, P, q, G, h, A, b, means, stds, ratios, mu):

	optimized_portfolio = []
	
	# Find optimal weights
	
	w = opt.solvers.qp(mu*P, -q, G, h, A, b)

#	indices = np.where(np.array(w['x']) > 0.01)[0]
#	#indices = [index for index in indices if index < len(means)-6]
#	weights = [np.array(w['x'])[i] for i in indices]

	sol = list(w['x'])
	indices = [index for index, weight in enumerate(sol) if weight > 0.001]
	weights = [weight for weight in sol if weight > 0.001]

	means = np.array(means)
	stds = np.array(stds)


	for i, weight in zip(indices, weights):
		if requested_stocks == 'all':
			print(requested_stocks[i].name, weight, means[i], stds[i], ratios[i])
			optimized_portfolio.append((requested_stocks[i], weight, means[i], stds[i]))
		
		else:
			print(requested_stocks[i].name, weight, means[i], stds[i], ratios[i])
#optimized_portfolio.append((stock_market.stocks[[s.name for s in stock_market.stocks].index(requested_stocks[i])], weight, means[i], stds[i]))
			optimized_portfolio.append((requested_stocks[i], weight, means[i], stds[i]))


	return optimized_portfolio
