# For mathematical calculations
import numpy as np

# For handling datasets
import pandas as pd

# For plotting graphs
from matplotlib import pyplot as plt

# For linear regression
from sklearn.linear_model import LinearRegression

# This is the URL I got the dataset from, before scrubbing it
#     and converting it into a CSV. Might be able to do it on
#     the fly, but I didn't want to waste time.
url = "http://www.randomservices.org/random/data/Galton.txt"

# Import the dataset, explicitly labeling the headers and index column
df = pd.read_csv('galtondata.csv',
                 index_col='Family', skiprows=1,
                 names=['Family','Father','Mother',
                        'Gender','Height','Kids'],)

# Prepare the training set
x_train = df['Father'].values[:,np.newaxis]
y_train = df['Height'].values


lm = LinearRegression()

# Train the model
lm.fit(x_train, y_train)

x_test = [[72.8],[61.1],[67.4],[70.2],[75.6],[60.2],[65.3],[59.1],
          [71.1],[71.0],[68.2],[60.2],[72.5],[61.2],[66.3],[61.5],
          [70.2],[64.4],[67.7],[62.6],[71.7],[69.3],[70.3],[59.0]]

# Get the predictions from our LM
predictions = lm.predict(x_test)

# Print the results before plotting them
newD = {'Father': x_test, 'Height': predictions}
outputframe = pd.DataFrame(data=newD)
print(outputframe)

# Plot the training data
plt.scatter(x_train, y_train,color='blue')

# Plot the best fit line using predicted value
plt.plot(x_test,predictions,color='black',linewidth=2)

plt.xlabel('Father height in inches')
plt.ylabel("Child height in inches")
plt.show()
