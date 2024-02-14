import sys

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: {} <your log file.txt>".format(sys.argv[0]))

	with open(sys.argv[1], 'r') as f:
		ilpTime = None
		dpTime = None
		ilpEnergy = None
		dpEnergy = None

		for line in f:
			if "Total (root+branch&cut)" in line:
				ilpTime = float(line.split()[3])
			elif "Done Tracking in" in line:
				dpTime = float(line[line.find('Done Tracking in')+16:line.find('secs')].strip())
			elif "ILP model eval:" in line:
				ilpEnergy = float(line.split()[3])
			elif "DP model eval:" in line:
				dpEnergy = float(line.split()[3])

		#print("ILP time\tDP time\tILP Energy\tDP Energy")
		print("{},{},{},{}".format(ilpTime, dpTime, ilpEnergy, dpEnergy))