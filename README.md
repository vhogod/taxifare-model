#Train a regression model
 
Now it's time build a regression model and train it with the data prepared in the previous exercise. We'll try three different regression algorithms to determine which one produces the most accurate results, and use cross-validation to meaasure accuracy. Then we'll train the best model. Start with a linear-regression model.

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

x = df.drop(['fare_amount'], axis=1)
y = df['fare_amount']
 
model = LinearRegression()
cross_val_score(model, x, y, cv=5).mean()
0.7258845061910318
Now try a RandomForestRegressor with the same dataset and see how its accuracy compares.

from sklearn.ensemble import RandomForestRegressor
 
model = RandomForestRegressor(random_state=0)
cross_val_score(model, x, y, cv=5).mean()
0.706157807448991
Assess a third model that uses GradientBoostingRegressor.

from sklearn.ensemble import GradientBoostingRegressor
 
model = GradientBoostingRegressor(random_state=0)
cross_val_score(model, x, y, cv=5).mean()
0.750496262408626
The GradientBoostingRegressor produced the highest cross-validated coefficient of determination. Train it using the entire dataset.

model.fit(x, y)
GradientBoostingRegressor(random_state=0)
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
The model is now trained and ready to make predictions.

Use the model to predict fare amounts
Finish up by using the trained model to make a pair of predictions. First, estimate what it will cost to hire a taxi for a 2-mile trip at 5:00 p.m. on Friday afternoon.

model.predict(pd.DataFrame({ 'day_of_week': [4], 'pickup_time': [17], 'distance': [2.0] }))
array([11.49105571])
Now predict the fare amount for a 2-mile trip taken at 5:00 p.m. one day later (on Saturday).

model.predict(pd.DataFrame({ 'day_of_week': [5], 'pickup_time': [17], 'distance': [2.0] }))
array([10.95309995])
Does the model predict a higher or lower fare amount for the same trip on Saturday afternoon? Does this make sense given that the data comes from a New York City cab company?

