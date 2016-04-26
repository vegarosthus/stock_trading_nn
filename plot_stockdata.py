import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator,\
	DayLocator, MONDAY
from matplotlib.finance import candlestick2_ohlc


def plot(earnings, benchmark, run_no = None):
	#candlestick_tuple = (stock.opens, stock.closes, stock.highs, stock.lows)
	
	x = range(len(earnings))
	
	plt.figure(1)
	plt.plot(x, earnings)
	plt.plot(x, benchmark)
	#plt.plot(x[len(stock.dates)-len(stock.opens_sma):], stock.opens_sma)
	
	#plt.plot(x[len(stock.opens)-len(stock.opens_upper_bb):], stock.opens_upper_bb)
	#plt.plot(x[len(stock.opens)-len(stock.opens_lower_bb):], stock.opens_lower_bb)
	
	#fig, ax = plt.subplots()
	#candlestick2_ohlc(ax, stock.opens, stock.closes, stock.highs, stock.lows, width=0.6)
	#plt.plot(x[len(stock.opens)-len(stock.opens_upper_bb):], stock.opens_upper_bb)
	#plt.plot(x[len(stock.opens)-len(stock.opens_lower_bb):], stock.opens_lower_bb)
	

	path = '/Users/vegarosthus/Dropbox/portfolio' + str(run_no) + '.png'
	plt.savefig(path)

def plot_learning_curve(costs_train, costs_cv, costs_test, accuracies_train, f1s_train, accuracies_cv, f1s_cv, accuracies_test, f1s_test):

	x = range(len(costs_train))
	
	fig = plt.figure()
	ax = plt.subplot(111)

	ax.plot(x, costs_train, label = 'cost, training')
	ax.plot(x, costs_cv, label = 'cost, CV')
	ax.plot(x, costs_test, label = 'cost, test')
	ax.plot(x, accuracies_train, label = 'accuracy, train')
	ax.plot(x, f1s_train, label = 'f1 score, train')
	ax.plot(x, accuracies_cv, label = 'accuracy, CV')
	ax.plot(x, f1s_cv, label = 'f1 score, CV')
	ax.plot(x, accuracies_test, label = 'accuracy, test')
	ax.plot(x, f1s_test, label = 'f1 score, test')

	plt.xlabel('iterations')
	plt.ylabel('cost')
	plt.title('Cost as a function of iterations')

	ax.legend()

	plt.ylim([0,1])

	path = '/Users/vegarosthus/Dropbox/costs.png'
	plt.savefig(path)

def plot_weight_dev(thetas, time):

	figs = []
	axs = []
	
	for theta, i in zip(thetas, range(len(thetas))):
		figs.append(plt.figure(figsize=(12, 9)))
		axs.append(figs[i].add_subplot(111))
		axs[i].set_title('colorMap')

		plt.imshow(theta)
	
		axs[i].set_aspect('equal')

		cax = figs[i].add_axes([0.25, 0.1, 0.78, 0.8])

		cax.get_xaxis().set_visible(False)
		cax.get_yaxis().set_visible(False)
		cax.patch.set_alpha(1)
		cax.set_frame_on(False)

		plt.colorbar(orientation='vertical')

		path = '/Users/vegarosthus/Dropbox/Theta' + str(i+1) + time + '.png'
		plt.savefig(path)
#		plt.show()


def plot_markowitz(requested_stocks, means, stds, opt_means, opt_stds):

	fig, ax = plt.subplots()
	ax.plot(stds, means, 'o', markersize=5)
	plt.xlabel('std')
	plt.ylabel('mean')
	plt.title('Mean and standard deviation of returns stock market stocks')
	
	ax.plot(opt_stds, opt_means, 'y-o')
	
	for i, txt in enumerate([asset[0].name for asset in requested_stocks]):
		ax.annotate(txt, (opt_stds[i], opt_means[i]))
	
	
	plt.show()