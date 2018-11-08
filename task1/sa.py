import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
##read the given data sheet in csv and place the columns in separate lists

class Statistic_Analysis:
	def __init__(self):
		self.x1 = []
		self.x2 = []
		self.x3 = []
		self.x4 = []
		self.x5 = []
		self.var_y = []
		self.df = []

	def read_csv(self):
		reader = csv.reader(open('./rraman2.csv','r'),delimiter=',') ##change file name here
		columns = list(zip(*reader))
		self.x1 = [float(i) for i in columns[0]]
		self.x2 = [float(i) for i in columns[1]]
		self.x3 = [float(i) for i in columns[2]]
		self.x4 = [float(i) for i in columns[3]]
		self.x5 = [float(i) for i in columns[4]]
		self.var_y = [float(i) for i in columns[5]]
		self.df = pd.read_csv("./rraman2.csv", sep=',', header=None)
		

	def histogram(self,arr,i):
		plt.style.use('fivethirtyeight')
		binwidth = 1
		n, bins, patches = plt.hist(arr,bins= range(int(min(arr)), int(max(arr)) + binwidth, binwidth), histtype='bar',
		align='mid', color='#DC143C',edgecolor='white')
		plt.xlabel('x')
		plt.title('Histogram-x'+str(i))
		plt.savefig('./histo_x'+str(i))
		plt.clf()
	
	def mean(self):
		mean = []
		mean.append(sum(self.x1)/len(self.x1))
		mean.append(sum(self.x2)/len(self.x2))	
		mean.append(sum(self.x3)/len(self.x3))
		mean.append(sum(self.x4)/len(self.x4))
		mean.append(sum(self.x5)/len(self.x5))
		return mean
	
	def variance(self):	
		var_x = []
		var_x.append(np.var(self.x1))
		var_x.append(np.var(self.x2))
		var_x.append(np.var(self.x3))
		var_x.append(np.var(self.x4))
		var_x.append(np.var(self.x5))
		return var_x

	def boxplot(self,arr,i):
		plt.style.use('fivethirtyeight')
		res = plt.boxplot(arr,0)
		plt.xlabel('x')
		plt.title('Boxplot-x'+str(i))
		plt.savefig('./boxplot_x'+str(i))
		plt.clf()
		

	def z_score(self):
		z = np.abs(stats.zscore(self.df))
		df1 = self.df[(z < 2).all(axis=1)]
		return df1

	def correlation(self):
		arr = [self.df[5],self.df[0],self.df[1],self.df[2],self.df[3],self.df[4]]
		co_matrix = np.corrcoef(arr)
		return co_matrix

		
			

st = Statistic_Analysis()
st.read_csv()

##calculate histogram for all x values
st.histogram(st.x1,1)
st.histogram(st.x2,2)
st.histogram(st.x3,3)
st.histogram(st.x4,4)
st.histogram(st.x5,5)

##calculate mean for all x values
mean_x = st.mean()
print("mean of x1 to x5:", mean_x)

##variance for all x values
variance_x = st.variance() 
print("variance of x values:", variance_x)

##calculate boxplots for all x values, detect outliers and return arrays without outliers
st.boxplot(st.x1,1)
st.boxplot(st.x2,2)
st.boxplot(st.x3,3)
st.boxplot(st.x4,4)
st.boxplot(st.x5,5)

st.df = st.z_score()

st.boxplot(st.df[0],6)
st.boxplot(st.df[1],7)
st.boxplot(st.df[2],8)
st.boxplot(st.df[3],9)
st.boxplot(st.df[4],10)

##co-relation matrices
co_matrix = st.correlation()
print(co_matrix)
st.df.columns = ['x1', 'x2', 'x3','x4','x5','y']
st.df.to_csv('./rraman2_1.csv',index=False)
