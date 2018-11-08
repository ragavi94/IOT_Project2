import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import chisquare

##read the given data sheet in csv and place the columns in separate lists

class Linear_multivar:
	def __init__(self):
		self.df = []
		self.model = None
		#plt.style.use('fivethirtyeight')

	def read_csv(self):
		data = pd.read_csv("./rraman2_1.csv") ##change file name here
		#data = pd.read_csv("./rraman2.csv")
		#data.columns = ['x1', 'x2', 'x3','x4','x5','y']
		#print(data)
		self.df = data

	def create(self):
		x = self.df.iloc[:,0:5] ##take x1 to x5
		x = sm.add_constant(x) ##needs to be used later to find intercept/a0
		y = self.df.iloc[:,5:6] ##take y volumn
		return x,y

	def create1(self,j):
		cols = [3,4]
		x = self.df.iloc[:,cols] ##take x1 to x5 without the mentioned column
		x = sm.add_constant(x) ##needs to be used later to find intercept/a0
		y = self.df.iloc[:,5:6] ##take y volumn
		return x,y
		
	def runlinear(self, y, x):
		self.model = sm.OLS(y, x).fit() ##method for Ordinary Least Squares method
		predictions = self.model.predict(x) ##using model to make predictions
		##printing all statistics got out of the model
		print(self.model.summary())
		print("Intercept/a0:",self.model.params[0])
		print("MSE/variance value:",self.model.mse_total)
		print("MSE of residuals:",self.model.mse_resid)
		print("R-Square:",self.model.rsquared)
		print("F-Value:",self.model.fvalue)

	def correlation(self):
		arr = [self.df['y'],self.df['x1'],self.df['x2'],self.df['x3'],self.df['x4'],self.df['x5']]
		co_matrix = np.corrcoef(arr)
		return co_matrix

	def regplot(self,x,y,i):
		sns.regplot(x='x1', y='y', data=self.df)
		plt.savefig('./linear_reg_x'+str(i))
		plt.clf()
	
	def get_residuals(self):
		res = self.model.resid #fetching residuals from the model
		return res

	def qqplot(self,res,i):
		qq = sm.qqplot(res, stats.t, fit=True, line='45')
		plt.savefig('./qqplot_x'+str(i))
		plt.clf()

	def histogram(self,res,i):
		(mu, sigma) = norm.fit(res)
		n, bins, patches = plt.hist(res,normed=1, bins='auto', align='mid', color='navy',edgecolor='white')
		y = mlab.normpdf( bins, mu, sigma)
		l = plt.plot(bins, y, 'r--', linewidth=2)
		plt.savefig('./histo_x'+str(i))
		plt.clf()
	
	def s_plot(self,x,res,i):
		s_plot = sns.residplot(x[["x"+str(i)]],res)
		plt.xlabel('x'+str(i))
		plt.ylabel('y')
		plt.savefig('./scatter_x'+str(i))
		plt.clf()

	def chisquare(self,res):
		chi_s = chisquare(res)
		#chi_s = stats.chi2.cdf(res, 1)
		print("Chi-square Results:",chi_s)
		
		
s_linear = Linear_multivar()
s_linear.read_csv()
x,y = s_linear.create()
s_linear.runlinear(y, x)
s_linear.regplot(x,y,1)
res = s_linear.get_residuals()
s_linear.qqplot(res,1)
s_linear.histogram(res,1)
s_linear.chisquare(res)
s_linear.s_plot(x,res,1) ##scatter plots for all columns from x1 to x5
s_linear.s_plot(x,res,2)
s_linear.s_plot(x,res,3)
s_linear.s_plot(x,res,4)
s_linear.s_plot(x,res,5)
##co-relation matrix
co_matrix = s_linear.correlation()
print(co_matrix)

##dropping x2
x,y = s_linear.create1(1)
s_linear.runlinear(y, x)
s_linear.regplot(x,y,2)
res = s_linear.get_residuals()
s_linear.qqplot(res,2)
s_linear.histogram(res,2)
s_linear.chisquare(res)
#s_linear.s_plot(x,res,1) ##scatter plots for all columns from x1 to x5







