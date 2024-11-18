import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from scipy.stats import pearsonr
from pandas import read_csv
from sklearn.linear_model import LinearRegression


df=pd.read_csv(r"C:\Users\yuvik\Desktop\archive\credit_record.csv")


print(df.head(10)) 
print(df.tail(10)) 

colors = ['blue', 'green'] #color of the dots
plt.scatter(df['MONTHS_BALANCE'], df['STATUS'])

# Add labels and title
plt.xlabel('MONTHS_BALANCE')
plt.ylabel('STATUS')
plt.title('scatter plot')
plt.show()

#histogram 
plt.hist(df['STATUS'])
plt.title('Histogram')

#box plot
df.boxplot(by='MONTHS_BALANCE', column="STATUS", grid=False)
plt.title('box plot')
# Show the plot
plt.show()

#person correlation
list1=df["MONTHS_BALANCE"]
list2=df["STATUS"]

corr,_=pearsonr(list1,list2)
print("person correlation is:",corr)

#independent variable=year
#dependant variable=rank


#linear relationship
X = df[['MONTHS_BALANCE']].values.reshape(-1,1) # Independent variable (year)
y = df['STATUS'].values # Dependent variable (rank)

# Create a linear regression model
model = LinearRegression()

# Train the model on the data
model.fit(X, y)

# Predict ratings based on the year
predicted_ratings = model.predict(X)

# Plot the original data
plt.scatter(df['MONTHS_BALANCE'], df['STATUS'], color='blue', label='Actual Ratings')

# Plot the regression line (Predicted ratings)
plt.plot(df['MONTHS_BALANCE'], predicted_ratings, color='red', label='Predicted Ratings')

plt.xlabel('MONTHS_BALANCE')
plt.ylabel('STATUS')
plt.title('monthbalance vs status with Regression Line')
plt.legend()
plt.show()

# Print the model's slope and intercept
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

