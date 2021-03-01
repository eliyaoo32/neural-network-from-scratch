import matplotlib.pyplot as plt
from GameOfLife.dataSet import classified_training_data, classified_test_data
from GameOfLife.game import BOARD_WIDTH, BOARD_HEIGHT
from NeuralNetwork.NeuralNetwork import NeuralNetwork
from NeuralNetwork.dataStructure.ActivationFunction import tanh_by_beta, SIGMOID, ReLU, square_error


# Creating Neural Network
total_cells = BOARD_WIDTH * BOARD_HEIGHT
layers_size = (total_cells, total_cells, 1)

neural_network = NeuralNetwork(
    layers_size=layers_size,
    activation=tanh_by_beta(0.002),
    error_function=square_error
)
performance = neural_network.train(
    rate=1000, epoch=50,
    training_data=classified_training_data,
    testing_data=classified_test_data
)

print("====== Training Data ======")
print("Expected \t Got")
for data in classified_training_data:
    print("{} \t {}".format(str(data.expected), str(neural_network.apply(data.value))))

print("====== Testing Data ======")
print("Expected \t Got")
for data in classified_test_data:
    print("{} \t {}".format(str(data.expected), str(neural_network.apply(data.value))))

plt.plot(range(1, len(performance.training_errors) + 1), performance.training_errors)
plt.xlabel("Epoch")
plt.ylabel("Error Rate")
plt.title('Training Data Performance')
plt.show()

plt.plot(range(1, len(performance.test_errors) + 1), performance.test_errors)
plt.xlabel("Epoch")
plt.ylabel("Error Rate")
plt.title('Testing Data Performance')
plt.show()
