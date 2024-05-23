import numpy as np

class PerceptronAND:
    def _init_(self, learning_rate=0.1, epochs=1008):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
    
    def activation_function(self, x):
        return 1 if x >= 0 else 0
    
    def train(self, inputs, outputs):
        n_samples, n_features = inputs.shape
        self.weights = np.zeros(n_features + 1)
        
        for _ in range(self.epochs):
            for idx, x_i in enumerate(inputs):
                x_i = np.insert(x_i, 0, 1)  # Insertar el término de sesgo
                net_input = np.dot(x_i, self.weights)
                prediction = self.activation_function(net_input)
                error = outputs[idx] - prediction
                self.weights += self.learning_rate * error * x_i
    
    def predict(self, inputs):
        inputs = np.insert(inputs, 0, 1)  # Insertar el término de sesgo
        net_input = np.dot(inputs, self.weights)
        return self.activation_function(net_input)

# Datos de entrenamiento para la operación AND
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([0, 0, 0, 1])

# Crear y entrenar el perceptrón
perceptron = PerceptronAND()
perceptron.train(inputs, outputs)

# Probar el perceptrón
for x in inputs:
    prediction = perceptron.predict(x)
    print(f"{x[0]} AND {x[1]} = {prediction}")