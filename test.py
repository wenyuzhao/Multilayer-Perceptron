import nn
from data.arrow import train_data, test_data



# define model
model = nn.NeuralNetwork([
    nn.InputLayer(inputs=25),
    nn.LeakyReLULayer(inputs=25, outputs=25, alpha=0.01),
    nn.LeakyReLULayer(inputs=25, outputs=4, alpha=0.01),
    nn.SigmoidLayer(inputs=4, outputs=4),
])

# train data
model.fit(train_data, learning_rate=0.01, threshold=1e-5, epochs=200000)

# test
for x, y in test_data:
    print(x, y, model.predict(x))

