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

class Simple_Linear:
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
		x = self.df.iloc[:,0:1] ##take x1 alone
		x = sm.add_constant(x) ##needs to be used later to find intercept/a0
		y = self.df.iloc[:,5:6] ##take y volumn
		return x,y

	def hocreate(self):
		x1 = self.df.iloc[:,0:1] ##take x1
		x2 = self.df.iloc[:,1:2] ##take x2
		#x3 = self.df.iloc[:,2:3] ##take x3
		X = np.column_stack((x1, x1**2))
		X = sm.add_constant(X) ##needs to be used later to find intercept/a0
		y = self.df.iloc[:,5:6] ##take y volumn
		return X,y
		
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
		

	def regplot(self,x,y,i):
		sns.regplot(x='x1', y='y', data=self.df)
		plt.savefig('./linear_reg_x1'+str(i))
		plt.clf()
	
	def get_residuals(self):
		res = self.model.resid #fetching residuals from the model
		return res

	def qqplot(self,res,i):
		qq = sm.qqplot(res, stats.t, fit=True, line='45')
		plt.savefig('./qqplot_x1'+str(i))
		plt.clf()

	def histogram(self,res,i):
		(mu, sigma) = norm.fit(res)
		n, bins, patches = plt.hist(res,normed=1, bins='auto', align='mid', color='navy',edgecolor='white')
		y = mlab.normpdf( bins, mu, sigma)
		l = plt.plot(bins, y, 'r--', linewidth=2)
		plt.savefig('./histo_x1'+str(i))
		plt.clf()
	
	def s_plot(self,x,res,i):
		s_plot = sns.residplot(x[["x1"]],res)
		plt.xlabel('x1')
		plt.ylabel('y')
		plt.savefig('./scatter_x1'+str(i))
		plt.clf()

	def chisquare(self,res):
		chi_s = chisquare(res)
		#chi_s = stats.chi2.cdf(res, 1)
		print("Chi-square Results:",chi_s)
		
		
s_linear = Simple_Linear()
s_linear.read_csv()
x,y = s_linear.create()
s_linear.runlinear(y, x)
s_linear.regplot(x,y,1)
res = s_linear.get_residuals()
s_linear.qqplot(res,1)
s_linear.histogram(res,1)
s_linear.chisquare(res)
s_linear.s_plot(x,res,1)

##Repeat same for higher order


X,y = s_linear.hocreate()
s_linear.runlinear(y,X)
s_linear.regplot(X,y,2)
res = s_linear.get_residuals()
s_linear.qqplot(res,2)
s_linear.histogram(res,2)
s_linear.chisquare(res)
#s_linear.s_plot(X,res,2)


