import numpy as np

class neural_network():
    def __init__(self, inputs, outputs, hiddenlayers):
        self.inputs=inputs
        self.outputs = outputs 
        self.layers = len(hiddenlayers) + 1
        self.weights = [np.random.rand(inputs, hiddenlayers[0])]
        for i in range(1,len(hiddenlayers)):
            self.weights.append(np.random.rand(hiddenlayers[i-1], hiddenlayers[i]))
        
        self.weights.append(np.random.rand(hiddenlayers[-1], len(outputs)))


        self.biasis = [np.random.rand(inputs)]
        for layer in hiddenlayers:
            self.biasis.append(np.random.rand(layer))
        
        self.biasis.append(np.random.rand(len(outputs)))


    def run(self, data, all_neurons=False):
        if isinstance(data, list):
            if len(data) == self.inputs:
                input = np.array(data)
                layers = [input]
                for i in range(self.layers):
                    result = np.dot(layers[i], self.weights[i]) + self.biasis[i]
                    layer = self.sigmoid(result)
                    # Calculate the sum of each row
                    layers.append(layer)
                if all_neurons: return layers[-1]
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
        

    def train(self, data, desired_outputs):
        if len(data) != len(desired_outputs): raise Exception("length of data must be equal to the length of desired_outputs")
        
        changes_to_weights = []
        for weights in self.weights:
            changes_to_weights.append(weights*0)
        changes_to_biasis = np.array([0]*len(self.biasis))
        for i in range(len(data)):
            layers = self.test(data[i], True)
            weights, biasis = self.backprop(layers, desired_outputs[i])

            for i in range(len(weights)):
                changes_to_weights[i] += weights[i]

            changes_to_biasis += biasis
        
        for i in range(len(self.weights)):
            self.weights[i] += changes_to_weights[i]

        self.biasis += changes_to_biasis

        return changes_to_biasis, changes_to_weights

    def dif_sigmoid(self, array):
        return np.exp(-array)/np.square(1 + np.exp(-array))

    def sigmoid(self, array):
        return 1/(1+np.exp(-array))

nn = neural_network(1,["1","2"],[2,2])
nn.test([0.5])




