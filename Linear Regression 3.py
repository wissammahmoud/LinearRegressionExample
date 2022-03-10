import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats                  # to implement linear regression
import os
from colorama import Fore, Style

# init
scriptFolder = os.path.dirname(os.path.abspath(__file__))
csvPath = scriptFolder + "/data.csv"

# create a data frame df from the csv file
df = pd.read_csv(csvPath)

# init x & y vectors
x = df.Duration
y = df.Maxpulse

# perform the linear regression
result = stats.linregress(x, y)
slope = result.slope
yIntercept = result.intercept
correlationCoefficient = result.rvalue
slopeStandardError = result.stderr
yInterceptStandardError = result.intercept_stderr

# 
# regressionFunction
# 
# straight line: y = ax + b
# a: slope
# b: y-intercept
# x: input data
# y: output data 
# 
def regressionFunction(x):
    return slope * x + yIntercept

# predict the vector y using the regression function and the input vector x
predictedY = list(map(regressionFunction, x))

# output the result
print("\n Training Data Set Size:", len(x), "Records")
print("\n Regression Line: Y = " + str(round(slope, 2)) + "X + " + str(round(yIntercept, 2)))
print(Fore.RED + "\n Correlation Coefficient:", round(correlationCoefficient, 2),
        "==> There is no relationship between Duration & Max Pulse!", Style.RESET_ALL)     
print("\n Slope Standard Error:", round(slopeStandardError, 2))
print("\n Y-Intercept Standard Error:", round(yInterceptStandardError, 2))
print("\n")

# plot the original data along with the fitted line
font1 = {'family':'serif','color':'blue','size':25}
font2 = {'family':'serif','color':'darkred','size':15}
plt.plot(x, y, "o", label="Original Data")
plt.plot(x, predictedY, color="red", label="Fitted Line")
plt.title("Training", fontdict=font1)
plt.xlabel("Duration", fontdict=font2)
plt.ylabel("Max Pulse", fontdict=font2)
plt.legend()
plt.grid(color="green", linestyle="--", linewidth=0.5)
plt.show() 
