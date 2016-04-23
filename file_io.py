def read_from_file(name):
	with open('stock_data/' + name + '.txt', 'r') as data_file:
		next(data_file)
			first_line = data_file.readline()
		data_file.close()
		
		first_line = first_line.split("\t")
		date = int(float(first_line[0]))
		
		return date

def write_to_file(name, data_depth):
	if data_depth == 'day':
		
		with open('stock_data/' + name + '.txt', 'w+') as data_file:
			data_file.write("opening prices\t high prices\t low prices\t closing prices\t volumes\t values\t opens SMA\t closes SMA\t highs SMA\t lows SMA\t volumes SMA\t values SMA opens upper BB\t opens lower BB\t closes upper BB\t closes lower BB\t highs upper BB\t highs lower BB\t lows upper BB\t lows lower BB\t volumes upper BB\t volumes lower BB\t values upper BB\t values lower BB\n")
				
				for (la, o, h, l, vol, val, la_sma, o_sma, h_sma, l_sma, vol_sma, val_sma, la_u_bb, la_l_bb, o_u_bb, o_l_bb, h_u_bb, h_l_bb, l_u_bb, l_l_bb, vol_u_bb, vol_l_bb, val_u_bb, val_l_bb) in zip(self.lasts, self.opens, self.highs, self.lows, self.volumes, self.values, self.lasts_sma, self.opens_sma, self.highs_sma, self.lows_sma, self.volumes_sma, self.values_sma, self.lasts_upper_bb, self.lasts_lower_bb, self.opens_upper_bb, self.opens_lower_bb, self.highs_upper_bb ,self.highs_lower_bb, self.lows_upper_bb, self.lows_lower_bb, self.volumes_upper_bb, self.volumes_lower_bb, self.values_upper_bb, self.values_lower_bb):
					
					data_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(la, o, h, l, vol, val, la_sma, o_sma, h_sma,l_sma, vol_sma, val_sma, la_u_bb, la_l_bb, o_u_bb, o_l_bb, h_u_bb, h_l_bb, l_u_bb, l_l_bb, vol_u_bb, vol_l_bb, val_u_bb, val_l_bb))
	
		elif data_depth == 'historic':
			
			with open('stock_data/' + name + '.txt', 'w+') as data_file:
				data_file.write("opening prices\t high prices\t low prices\t closing prices\t volumes\t values\t opens SMA\t closes SMA\t highs SMA\t lows SMA\t volumes SMA\t values SMA opens upper BB\t opens lower BB\t closes upper BB\t closes lower BB\t highs upper BB\t highs lower BB\t lows upper BB\t lows lower BB\t volumes upper BB\t volumes lower BB\t values upper BB\t values lower BB\n")
				
				for (d, o, h, l, c, vol, val, o_sma, h_sma, l_sma, c_sma, vol_sma, val_sma, o_u_bb, o_l_bb, h_u_bb, h_l_bb, l_u_bb, l_l_bb, c_u_bb, c_l_bb, vol_u_bb, vol_l_bb, val_u_bb, val_l_bb) in zip(self.dates, self.opens, self.highs, self.lows, self.closes, self.volumes, self.values, self.opens_sma, self.highs_sma, self.lows_sma, self.closes_sma, self.volumes_sma, self.values_sma, self.opens_upper_bb, self.opens_lower_bb, self.highs_upper_bb ,self.highs_lower_bb, self.lows_upper_bb, self.lows_lower_bb,self.closes_upper_bb, self.closes_lower_bb, self.volumes_upper_bb, self.volumes_lower_bb, self.values_upper_bb, self.values_lower_bb):
					
					data_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(d, o, h, l, c, vol, val, o_sma, h_sma,l_sma, c_sma, vol_sma, val_sma, o_u_bb, o_l_bb, h_u_bb, h_l_bb, l_u_bb, l_l_bb, c_u_bb, c_l_bb, vol_u_bb, vol_l_bb, val_u_bb, val_l_bb))


		data_file.close()