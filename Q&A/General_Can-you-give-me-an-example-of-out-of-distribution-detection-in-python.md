# Can you give me an example of out of distribution in python?

Here is a simple example of out-of-distribution detection in python using the
Keras library.

```python 
# Import the necessary libraries
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import to_categorical

# Define the inputs and outputs for the model
inputs = Input(shape=(10,))
outputs = Dense(5, activation='softmax')(inputs)

# Create the model and compile it
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Generate some dummy data for training
X = np.random.rand(100, 10)
y = to_categorical(np.random.randint(0, 5, 100))

# Train the model on the dummy data
model.fit(X, y, epochs=10)

# Generate some out-of-distribution data
X_ood = np.random.rand(100, 10) + 5

# Get the model's predictions on the out-of-distribution data
predictions = model.predict(X_ood)

# Calculate the probability that each prediction is the true label
probabilities = np.max(predictions, axis=1)

# Set a threshold for the minimum probability required to consider a prediction reliable
threshold = 0.9

# Identify the predictions that are less than the threshold
out_of_distribution_indices = np.where(probabilities < threshold)[0]
```

In this example, we first create and train a simple model on some dummy data.
Then, we generate some out-of-distribution data and use the model to make
predictions on it. Finally, we calculate the probability that each prediction is
the true label, and identify the predictions that have a probability below a
certain threshold. These are the predictions that are likely to be unreliable
due to the model being out of distribution.