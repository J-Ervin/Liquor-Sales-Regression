import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Part a) Code - I manually created a date column to plot the graph

df = pd.read_csv("DataLiquor.csv")
df['Date'] = pd.date_range(start="1987-01", periods=len(df), freq="M")
df.set_index('Date', inplace=True)
df.rename(columns={'liquor': 'Sales'}, inplace=True)

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index, y=df['Sales'], color='blue', linewidth=2)
plt.title("Monthly Liquor Sales in the U.S. (1987-2014)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# Part b)

# Time index
df['Time'] = np.arange(1, len(df) + 1)
# Quadratic term
df['Time^2'] = df['Time'] ** 2

# Linear Model
X_linear = sm.add_constant(df['Time']) 
model_linear = sm.OLS(df['Sales'], X_linear).fit()

# Quadratic Model
X_quadratic = sm.add_constant(df[['Time', 'Time^2']])
model_quadratic = sm.OLS(df['Sales'], X_quadratic).fit()

# Exponential Model
df['Log_Sales'] = np.log(df['Sales'])
X_exp = sm.add_constant(df['Time'])
model_exponential = sm.OLS(df['Log_Sales'], X_exp).fit()

# Results
print("Linear Model Summary:")
print(model_linear.summary())

print("\nQuadratic Model Summary:")
print(model_quadratic.summary())

print("\nExponential Model Summary:")
print(model_exponential.summary())

#Linear Model Results:
#==============================================================================
#Dep. Variable:                  Sales   R-squared:                       0.819
#Model:                            OLS   Adj. R-squared:                  0.818
#Method:                 Least Squares   F-statistic:                     1511.
#Date:                Mon, 17 Feb 2025   Prob (F-statistic):          5.58e-126
#Time:                        14:39:23   Log-Likelihood:                -2261.0
#No. Observations:                 336   AIC:                             4526.
#Df Residuals:                     334   BIC:                             4534.
#Df Model:                           1
#Covariance Type:            nonrobust
#==============================================================================
#                 coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------
#const        553.8861     22.197     24.953      0.000     510.223     597.550
#Time           4.4372      0.114     38.865      0.000       4.213       4.662
#==============================================================================
#Omnibus:                      135.694   Durbin-Watson:                   1.626
#Prob(Omnibus):                  0.000   Jarque-Bera (JB):              536.268
#Skew:                           1.743   Prob(JB):                    3.56e-117
#Kurtosis:                       8.114   Cond. No.                         390.
#==============================================================================

# Quadratic Model Results
#==============================================================================
#Dep. Variable:                  Sales   R-squared:                       0.833
#Model:                            OLS   Adj. R-squared:                  0.832
#Method:                 Least Squares   F-statistic:                     829.9
#Date:                Mon, 17 Feb 2025   Prob (F-statistic):          4.25e-130
#Time:                        14:39:23   Log-Likelihood:                -2247.5
#No. Observations:                 336   AIC:                             4501.
#Df Residuals:                     333   BIC:                             4512.
#Df Model:                           2
#Covariance Type:            nonrobust
#==============================================================================
#                 coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------
#const        427.0565     32.153     13.282      0.000     363.808     490.305
#Time           6.6886      0.441     15.181      0.000       5.822       7.555
#Time^2        -0.0067      0.001     -5.277      0.000      -0.009      -0.004
#==============================================================================
#Omnibus:                      149.678   Durbin-Watson:                   1.762
#Prob(Omnibus):                  0.000   Jarque-Bera (JB):              614.650
#Skew:                           1.939   Prob(JB):                    3.39e-134
#Kurtosis:                       8.372   Cond. No.                     1.53e+05
#==============================================================================

# Exponential Model Results
#==============================================================================
#Dep. Variable:              Log_Sales   R-squared:                       0.843
#Model:                            OLS   Adj. R-squared:                  0.843
#Method:                 Least Squares   F-statistic:                     1798.
#Date:                Mon, 17 Feb 2025   Prob (F-statistic):          1.76e-136
#Time:                        14:39:23   Log-Likelihood:                 140.53
#No. Observations:                 336   AIC:                            -277.1
#Df Residuals:                     334   BIC:                            -269.4
#Df Model:                           1
#Covariance Type:            nonrobust
#==============================================================================
#                 coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------
#const          6.4543      0.017    369.483      0.000       6.420       6.489
#Time           0.0038   8.98e-05     42.399      0.000       0.004       0.004
#==============================================================================
#Omnibus:                       29.927   Durbin-Watson:                   1.079
#Prob(Omnibus):                  0.000   Jarque-Bera (JB):               40.277
#Skew:                           0.644   Prob(JB):                     1.79e-09
#Kurtosis:                       4.103   Cond. No.                         390.
#==============================================================================

# Part c) - All of the trend's coefficients are significant as they are all < 0.05.
# The linear model fit the worst, while the exponential model had the best fit as it had the highest R^2 and the lowest AIC/BIC

# Part d) -
# Linear - AIC - 4526, BIC - 4534.
# Quadratic - AIC - 4501, BIC - 4512.
# Exponential - AIC - (-277.1), BIC - (-269.4)
# The Exponential Model is superior as it has by far the lowest AIC/BIC

# Part e) Code
df['Predicted_Linear'] = model_linear.fittedvalues
df['Residuals_Linear'] = df['Sales'] - df['Predicted_Linear']
plt.figure(figsize=(12, 6))

# Actual Sales and Predicted Sales
plt.subplot(2, 1, 1)
sns.lineplot(data=df, x=df.index, y='Sales', label='Actual Sales', color='blue', linewidth=2)
sns.lineplot(data=df, x=df.index, y='Predicted_Linear', label='Predicted Sales', color='red', linestyle='--', linewidth=2)
plt.title("Actual vs Predicted Liquor Sales (Linear Trend)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)

# Plot 
plt.subplot(2, 1, 2)
sns.lineplot(data=df, x=df.index, y='Residuals_Linear', color='green', linewidth=2)
plt.axhline(0, color='black', linestyle='--', linewidth=1)  # Add a line at y=0
plt.title("Residuals from Linear Model", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Residuals")
plt.grid(True)

plt.tight_layout()
plt.show()

# Part e) - The residuals seem to be somewhat randomly distributed which means that the linear model fits well but also means that a more complex model could fit better.

# Part f) Code

df['Month'] = df.index.month

# Create seasonal dummy
seasonal_dummies = pd.get_dummies(df['Month'], prefix='Month', drop_first=True)
seasonal_dummies = seasonal_dummies.apply(pd.to_numeric, errors='coerce')
df = pd.concat([df, seasonal_dummies], axis=1)

# Making sure data is formatted properly
df['Time'] = np.arange(1, len(df) + 1)
df['Time^2'] = df['Time'] ** 2
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
df[['Time', 'Time^2']] = df[['Time', 'Time^2']].apply(pd.to_numeric, errors='coerce')
df[seasonal_dummies.columns] = df[seasonal_dummies.columns].apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Linear Model
X_linear_seasonal = df[['Time'] + seasonal_dummies.columns.tolist()]
X_linear_seasonal = sm.add_constant(X_linear_seasonal)
X_linear_seasonal = np.asarray(X_linear_seasonal, dtype=float)
y = np.asarray(df['Sales'], dtype=float)
model_linear_seasonal = sm.OLS(y, X_linear_seasonal).fit()

# Quadratic Model
X_quadratic_seasonal = df[['Time', 'Time^2'] + seasonal_dummies.columns.tolist()]
X_quadratic_seasonal = sm.add_constant(X_quadratic_seasonal)
X_quadratic_seasonal = np.asarray(X_quadratic_seasonal, dtype=float)
model_quadratic_seasonal = sm.OLS(y, X_quadratic_seasonal).fit()

print("Linear Model with Seasonal Dummies Summary:")
print(model_linear_seasonal.summary())

print("\nQuadratic Model with Seasonal Dummies Summary:")
print(model_quadratic_seasonal.summary())

# Part f) Results - The Quadratic model has a higher R^2 and a lower AIC/BIC than the Linear, meaning it has a better fit
# Multicollinearity might be an issue with a large condition number

#Linear Model with Seasonal Dummies Summary:
#                            OLS Regression Results
#==============================================================================
#Dep. Variable:                      y   R-squared:                       0.945
#Model:                            OLS   Adj. R-squared:                  0.943
#Method:                 Least Squares   F-statistic:                     465.0
#Date:                Mon, 17 Feb 2025   Prob (F-statistic):          1.21e-195
#Time:                        21:03:30   Log-Likelihood:                -2059.9
#No. Observations:                 336   AIC:                             4146.
#Df Residuals:                     323   BIC:                             4195.
#Df Model:                          12
#Covariance Type:            nonrobust
#==============================================================================
#                 coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------
#const        430.4644     23.835     18.060      0.000     383.573     477.356
#x1             4.3946      0.064     68.819      0.000       4.269       4.520
#x2           -61.8946     30.324     -2.041      0.042    -121.552      -2.237
#x3            36.7465     30.324      1.212      0.226     -22.912      96.405
#x4            40.4947     30.325      1.335      0.183     -19.164     100.154
#x5           123.2787     30.325      4.065      0.000      63.619     182.938
#x6           131.2412     30.326      4.328      0.000      71.580     190.902
#x7           192.2038     30.326      6.338      0.000     132.541     251.866
#x8           152.6306     30.327      5.033      0.000      92.966     212.295
#x9            73.4503     30.328      2.422      0.016      13.784     133.116
#x10           98.5557     30.330      3.249      0.001      38.887     158.224
#x11          136.5539     30.331      4.502      0.000      76.883     196.225
#x12          643.9093     30.332     21.229      0.000     584.236     703.583
#==============================================================================
#Omnibus:                       18.077   Durbin-Watson:                   0.674
#Prob(Omnibus):                  0.000   Jarque-Bera (JB):               33.310
#Skew:                          -0.309   Prob(JB):                     5.85e-08
#Kurtosis:                       4.413   Cond. No.                     2.42e+03
#==============================================================================

#Notes:
#[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#[2] The condition number is large, 2.42e+03. This might indicate that there are
#strong multicollinearity or other numerical problems.

#Quadratic Model with Seasonal Dummies Summary:
#                            OLS Regression Results
#==============================================================================
#Dep. Variable:                      y   R-squared:                       0.959
#Model:                            OLS   Adj. R-squared:                  0.958
#Method:                 Least Squares   F-statistic:                     583.6
#Date:                Mon, 17 Feb 2025   Prob (F-statistic):          6.30e-215
#Time:                        21:03:30   Log-Likelihood:                -2010.3
#No. Observations:                 336   AIC:                             4049.
#Df Residuals:                     322   BIC:                             4102.
#Df Model:                          13
#Covariance Type:            nonrobust
#==============================================================================
#                 coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------
#const        303.6368     23.862     12.725      0.000     256.692     350.581
#x1             6.6482      0.221     30.058      0.000       6.213       7.083
#x2            -0.0067      0.001    -10.522      0.000      -0.008      -0.005
#x3           -61.9615     26.200     -2.365      0.019    -113.505     -10.418
#x4            36.6261     26.200      1.398      0.163     -14.918      88.170
#x5            40.3343     26.200      1.539      0.125     -11.211      91.879
#x6           123.0915     26.200      4.698      0.000      71.546     174.637
#x7           131.0406     26.201      5.001      0.000      79.494     182.587
#x8           192.0032     26.202      7.328      0.000     140.455     243.551
#x9           152.4434     26.202      5.818      0.000     100.894     203.993
#x10           73.2898     26.203      2.797      0.005      21.739     124.841
#x11           98.4353     26.204      3.756      0.000      46.882     149.988
#x12          136.4870     26.205      5.208      0.000      84.932     188.042
#x13          643.9093     26.207     24.571      0.000     592.352     695.467
#==============================================================================
#Omnibus:                       14.766   Durbin-Watson:                   0.906
#Prob(Omnibus):                  0.001   Jarque-Bera (JB):               28.082
#Skew:                           0.221   Prob(JB):                     7.98e-07
#Kurtosis:                       4.346   Cond. No.                     6.32e+05
#==============================================================================

# Part g) - No. That would make the model have perfect multicollinearity.

# Part h) The seasonal dummies improved the fit of both models.
# The AIC and BIC for the Linear model dropped roughly 500
# The AIC and BIC for the Quadratic model dropped roughly 100
# The models with the dummies are still less fit than the exponential model without dummies.

# Linear - 
#AIC:                             4526.
#BIC:                             4534.
# Quad
#AIC:                             4501.
#BIC:                             4512.
# Expo
#AIC:                            -277.1
#BIC:                            -269.4
# Linear Dummy
#AIC:                             4146.
#BIC:                             4195.
# Quad Dummy
#AIC:                             4049.
#BIC:                             4102.

# Part i) Code

# Predicted values from the linear model with seasonal dummies
df['Predicted_Linear_Seasonal'] = model_linear_seasonal.predict(X_linear_seasonal)

# Residuals from the linear model with seasonal dummies
df['Residuals_Linear_Seasonal'] = df['Sales'] - df['Predicted_Linear_Seasonal']

plt.figure(figsize=(14, 6))


sns.lineplot(data=df, x=df.index, y=df['Sales'], label='Actual Sales', color='blue', linewidth=2)
sns.lineplot(data=df, x=df.index, y=df['Predicted_Linear_Seasonal'], label='Predicted Sales (Linear with Seasonal Dummies)', color='orange', linestyle='--', linewidth=2)

# Predicted vs Actual Plot
plt.title('Actual vs Predicted Liquor Sales (Linear with Seasonal Dummies)', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual Plot
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x=df.index, y=df['Residuals_Linear_Seasonal'], label='Residuals (Linear with Seasonal Dummies)', color='green', linestyle=':', linewidth=2)
plt.title('Residuals from Linear Trend Model with Seasonal Dummies', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Residuals')
plt.grid(True)
plt.tight_layout()
plt.show()

# Part i) - The predicted sales fit far better on this plot comapred to the one in part e)

# Part j) Code
from statsmodels.stats.stattools import durbin_watson

residuals_linear_seasonal = model_linear_seasonal.resid
dw_statistic = durbin_watson(residuals_linear_seasonal)
print(f"Durbin-Watson Statistic: {dw_statistic}")

# Part j) - Durbin-Watson Statistic: 0.6740069382293244
# The statistic is less than 2 which means that the Residuals are not independent of each other and that there is likely serial correlation