import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

df=pd.read_csv("exp1/student_scores.csv")

# analyse the graph it linear or not
x=df[['Hours']]
y=df['Scores']
plt.figure(figsize=(7,8))
plt.scatter(x,y)
plt.show()

model=LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    test_size=0.2,
    random_state=42
)

print("Training Data:")
print(pd.concat([x_train, y_train], axis=1).head())
print("\nTesting Data:")
print(pd.concat([x_test, y_test], axis=1).head())

model.fit(x_train, y_train) # .fit do Train the model using the given data.
print("model train successfully")

plt.figure(figsize=(7,8))
plt.scatter(x,y,color="Blue",label="actual data")
plt.plot(x,model.predict(x),color="Red",label="predicted") #model.predict(x) represent y axis
plt.legend()
plt.show()

y_pred = model.predict(x_test)

table = pd.concat([
    x_test,
    y_test,
    pd.DataFrame(y_pred.round(2), columns=["Predicted Marks"], index=x_test.index)
], axis=1) #axis 1 Combine them side-by-side → add columns. #axis=0 means top-to-bottom → add rows
table = table.sort_values(by='Hours').reset_index(drop=True)
print(table)


# Actual y          Predicted y
#    ↓                  ↓
# y_test    → compare ← y_pred
#               ↓
#              MAE

mae = mean_absolute_error(y_test, y_pred) # it only take dependent value that is 
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score : {r2:.4f}")

#conclusion
evaluation = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R² Score"],
    "Value": [
        round(mae,2),
        round(mse,2),
        round(rmse,2),
        round(r2,4)
    ]
})
print(evaluation)


