import nn
from data.arrow import display, train_data, test_data

#display(train_data[0][0])

# define model
input_layer = [ nn.InputLayer(inputs=25) ]
encoder = [
    nn.LeakyReLULayer(inputs=25, outputs=16, alpha=0.01),
    nn.LeakyReLULayer(inputs=16, outputs=4, alpha=0.01),
    nn.LeakyReLULayer(inputs=4, outputs=2, alpha=0.01),
]
decoder = [
    nn.LeakyReLULayer(inputs=2, outputs=4, alpha=0.01),
    nn.LeakyReLULayer(inputs=4, outputs=16, alpha=0.01),
    nn.LeakyReLULayer(inputs=16, outputs=25, alpha=0.01),
]
model = nn.NeuralNetwork(input_layer + encoder + decoder)



# train data
_train_data = [ (x, x) for x, _ in train_data ]
model.fit(_train_data, learning_rate=0.03, threshold=1e-5, epochs=300000)



# test
for x, y in test_data:
    print('---------\ninput:')
    display(x)
    print('\nprediction:')
    display(model.predict(x))
    print('')
