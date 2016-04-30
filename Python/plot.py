import matplotlib.pyplot as plt

fileName = ['a9a','real-sim.svml','news20.binary','rcv1_test.binary']

interval = 2000
start = 0
last = 2000
index = 0
x = []
y = []

for line in open("chart1.txt"):
	listWords = line.split(" ")
	listWords[1] = listWords[1].rstrip('\n')
	if float(listWords[0]) < last:
		x.append(float(listWords[0]) - start)
		y.append(float(listWords[1]))
	else:
		plt.xlabel('Time')
		plt.ylabel('Function Value')
		plt.title(fileName[index])
		plt.plot(x, y, '-o')
		plt.savefig('chart1_'+fileName[index]+'.jpg')
		plt.close()
		x = []
		y = []
		start += interval
		last += interval
		index += 1
		x.append(float(listWords[0]) - start)
		y.append(float(listWords[1]))

plt.xlabel('Time')
plt.ylabel('Function Value')
plt.title(fileName[index])
plt.plot(x, y, '-o')
plt.savefig('chart1_'+fileName[index]+'.jpg')
plt.close()

interval = 2000
start = 0
last = 2000
index = 0
x = []
y = []

for line in open("chart2.txt"):
	listWords = line.split(" ")
	listWords[1] = listWords[1].rstrip('\n')
	if float(listWords[0]) < last:
		x.append(float(listWords[0]) - start)
		y.append(float(listWords[1]))
	else:
		plt.xlabel('Time')
		plt.ylabel('|g|_inf')
		plt.title(fileName[index])
		plt.plot(x, y, '-o')
		plt.savefig('chart2_'+fileName[index]+'.jpg')
		plt.close()
		x = []
		y = []
		start += interval
		last += interval
		index += 1
		x.append(float(listWords[0]) - start)
		y.append(float(listWords[1]))

plt.xlabel('Time')
plt.ylabel('|g|_inf')
plt.title(fileName[index])
plt.plot(x, y, '-o')
plt.savefig('chart2_'+fileName[index]+'.jpg')
plt.close()