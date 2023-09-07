import numpy as np

class neural_network():
    def __init__(self, inputs, outputs, hiddenlayers, learning_factor=1):
        self.learning_factor = learning_factor
        self.inputs=inputs
        self.outputs = outputs 
        self.layers = [inputs] + hiddenlayers + [len(outputs)]
        self.weights = [np.random.rand(inputs, hiddenlayers[0])]
        for i in range(1,len(hiddenlayers)):
            self.weights.append(np.random.rand(hiddenlayers[i-1], hiddenlayers[i]))
        
        self.weights.append(np.random.rand(hiddenlayers[-1], len(outputs)))


        self.biasis = []
        for layer in hiddenlayers:
            self.biasis.append(np.random.rand(layer))
        
        self.biasis.append(np.random.rand(len(outputs)))


    def run(self, data, all_neurons=False):
        if isinstance(data, list):
            if len(data) == self.inputs:
                input = np.array(data)
                layers = [input]
                for i in range(len(self.layers)-1):
                    result = np.dot(layers[i], self.weights[i]) + self.biasis[i]
                    layer = self.sigmoid(result)
                    # Calculate the sum of each row
                    layers.append(layer)
                if not all_neurons: return layers[-1]
                else: return layers
            else:
                raise Exception(f"Inputs must be equal to the number of input neurons: {self.inputs}")
        else:
            raise Exception("Input data must be a list")

    def test(self, data, desired_outputs):
        for i in range(len(data)):
            output = self.run(data[i])
            print("output:", output)
            print("desired output", desired_outputs[i])

    def backprop(self, layers, desired_ouputs):
        change_to_neurons = []
        for layer in self.layers:
            change_to_neurons.append(np.array([0] * layer))

        changes_to_weights = []
        for weights in self.weights:
            changes_to_weights.append(weights*0)

        changes_to_biasis = []
        for bias in self.biasis:
            changes_to_biasis.append(bias*0)

        for i in range(len(layers)-1,0):
            if i == len(layers)-1:
                change_to_neurons[i] = 2*(layers[i]-desired_ouputs)
            else:
                change_to_neurons[i] = np.dot(self.dif_sigmoid(np.dot(layers[i], self.weights[i+1]) + self.biasis[i+1]) * change_to_neurons[i+1], weights[i+1])

            #[3,4,5,6]            [3,4,5,8]            
            #[5,6,7,9] * sigmoid'([5,6,7,3] * [0.5, 0.3, 0.7, 0.3] + [3,6,2,9]) * [3,6,1]
            #[2,3,6,1]            [2,3,6,5]
            
            changes_to_weights[i] = np.outer(self.dif_sigmoid(np.dot(layers[i], self.weights[i+1]) + self.biasis[i+1]) * change_to_neurons[i+1], weights[i+1])
            changes_to_biasis[i] = np.dot(self.dif_sigmoid(np.dot(layers[i-1], self.weights[i]) + self.biasis[i]), change_to_neurons[i])

        return changes_to_weights, changes_to_biasis

    def train(self, data, desired_outputs):
        if len(data) != len(desired_outputs): raise Exception("length of data must be equal to the length of desired_outputs")
        
        changes_to_weights = []
        for weights in self.weights:
            changes_to_weights.append(weights*0)

        changes_to_biasis = []
        for bias in self.biasis:
            changes_to_biasis.append(bias*0)

        for i in range(len(data)):
            layers = self.run(data[i], True)
            weights, biasis = self.backprop(layers, desired_outputs[i])

            for i in range(len(weights)):
                changes_to_weights[i] += weights[i]
            
            for i in range(len(biasis)):
                changes_to_biasis[i] += biasis[i]

        print("change_to_weights", changes_to_weights)
        print("change_to_biasis", changes_to_biasis)

        for change in changes_to_weights:
            change = (change * self.learning_factor)/len(data)

        for change in changes_to_biasis:
            change = (change * self.learning_factor)/len(data)

        
        for i in range(len(self.weights)):
            self.weights[i] += changes_to_weights[i]
        
        for i in range(len(self.biasis)):
            self.biasis[i] += changes_to_biasis[i]

        return changes_to_biasis, changes_to_weights

    def dif_sigmoid(self, array):
        return np.exp(-array)/np.square(1 + np.exp(-array))

    def sigmoid(self, array):
        return 1/(1+np.exp(-array))


nn = neural_network(1,["0","1"],[3,3], 100000)

data = []
desired_outputs = []

for i in range(2):
    data.append([i%2])
    desired_outputs.append([(i+1)%2,i%2])


nn.train(data, desired_outputs)
nn.test([[0],[1],[0],[1]], [[1,0],[0,1],[1,0],[0,1]])




