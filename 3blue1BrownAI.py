import numpy as np

class neural_network():
    def __init__(self, inputs, outputs, hiddenlayers):
        self.inputs=inputs
        self.outputs = outputs 
        self.layers = len(hiddenlayers) + 2
        self.weights = [np.random.rand(inputs, hiddenlayers[0])]
        for i in range(1,len(hiddenlayers)):
            self.weights.append(np.random.rand(hiddenlayers[i-1],hiddenlayers[i]))
        
        self.weights.append(np.random.rand(hiddenlayers[-1], len(outputs)))


        self.biasis = [np.random.rand(inputs)]
        for layer in hiddenlayers:
            self.biasis.append(np.random.rand(layer))
        
        self.biasis.append(np.random.rand(len(outputs)))


    def test(self, data):
        if isinstance(data) == list:
            if len(data) == self.inputs:
                input = np.array(data)
                layers = [input]
                for i in range(self.layers):
                    result = np.dot(self.weights[i], layers[i]) + self.biasis[i]

                    # Calculate the sum of each row
                    layers.append(self.sigmoid(result))
                
                print(layers[-1])
                    
            
    def sigmoid(self, array):
        return 1/(1+array)







