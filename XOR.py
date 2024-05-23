import numpy as np

class NeuralNetworkXOR:
    def _init_(self, learning_rate=0.1, epochs=20000, hidden_units=4):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_units = hidden_units
        self.weights_input_hidden = None
        self.weights_hidden_output = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, outputs):
        n_samples, n_features = inputs.shape
        self.weights_input_hidden = np.random.rand(n_features + 1, self.hidden_units) * 2 - 1
        self.weights_hidden_output = np.random.rand(self.hidden_units + 1, 1) * 2 - 1

        for _ in range(self.epochs):
            # Feedforward
            input_with_bias = np.insert(inputs, 0, 1, axis=1)
            net_hidden = np.dot(input_with_bias, self.weights_input_hidden)
            output_hidden = self.sigmoid(net_hidden)

            hidden_with_bias = np.insert(output_hidden, 0, 1, axis=1)
            net_output = np.dot(hidden_with_bias, self.weights_hidden_output)
            output = self.sigmoid(net_output)

            # Backpropagation
            error_output = outputs - output
            delta_output = error_output * self.sigmoid_derivative(output)

            error_hidden = delta_output.dot(self.weights_hidden_output[1:, :].T)
            delta_hidden = error_hidden * self.sigmoid_derivative(output_hidden)

            # Update weights
            self.weights_hidden_output += hidden_with_bias.T.dot(delta_output) * self.learning_rate
            self.weights_input_hidden += input_with_bias.T.dot(delta_hidden) * self.learning_rate

    def predict(self, inputs):
        input_with_bias = np.insert(inputs, 0, 1, axis=1)
        net_hidden = np.dot(input_with_bias, self.weights_input_hidden)
        output_hidden = self.sigmoid(net_hidden)

        hidden_with_bias = np.insert(output_hidden, 0, 1, axis=1)
        net_output = np.dot(hidden_with_bias, self.weights_hidden_output)
        output = self.sigmoid(net_output)

        return np.round(output)

    def print_weights(self):
        print("Pesos de la capa de entrada a la capa oculta:")
        print(self.weights_input_hidden)
        print("Pesos de la capa oculta a la capa de salida:")
        print(self.weights_hidden_output)

# Datos de entrenamiento para la operación XOR
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

# Crear y entrenar la red neuronal MLP
mlp_xor = NeuralNetworkXOR(learning_rate=0.5, epochs=50000)
mlp_xor.train(inputs, outputs)

# Mostrar los pesos generados
mlp_xor.print_weights()

# Probar la red neuronal MLP
for x in inputs:
    result = mlp_xor.predict(x.reshape(1, -1))[0][0]
    print(f"{x[0]} XOR {x[1]} = {int(result)}")
