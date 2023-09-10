import numpy as np

class neural_network():
    def __init__(self, inputs, outputs, hiddenlayers, learning_factor=1, debug=False):
        self.debug = debug
        self.learning_factor = learning_factor
        self.inputs = inputs
        self.outputs = outputs 
        self.layers = [inputs] + hiddenlayers + [len(outputs)]
        self.weights = [np.random.randint(-1, 1, size = (hiddenlayers[0], inputs)) + np.random.rand(hiddenlayers[0], inputs)]
        for i in range(1,len(hiddenlayers)):
            self.weights.append(np.random.randint(-1, 1, size = (hiddenlayers[i], hiddenlayers[i-1])) + np.random.rand(hiddenlayers[i], hiddenlayers[i-1]))
        
        self.weights.append(np.random.randint(-1, 1, size = (len(outputs), hiddenlayers[-1])) + np.random.rand(len(outputs), hiddenlayers[-1]))

        np.random.randn
        self.biases = []
        for layer in hiddenlayers:
            self.biases.append(np.zeros((layer)))
        self.biases.append(np.zeros((len(outputs))))
        """  for layer in hiddenlayers:
            self.biases.append(np.random.randint(-1, 1, size = (layer)) + np.random.rand(layer))
        self.biases.append(np.random.randint(-1, 1, size = (len(outputs))) + np.random.rand(len(outputs))) """

    def run(self, data, all_neurons=False):
        if isinstance(data, list):
            if len(data) == self.inputs:
                input = np.array(data)
                layers = [input]
                for i in range(len(self.layers)-1):
                    result = np.dot(self.weights[i], layers[i]) + self.biases[i]
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
        cost = 0
        for i in range(len(data)):
            output = self.run(data[i])
            print("output:", output)
            print("desired output", desired_outputs[i])
            cost += self.cost_function(output, desired_outputs[i])
        print("cost:", cost/len(data))

    def cost_function(self, output, desired_output):
        return np.sum(np.square(output-desired_output))

    def backprop(self, layers, desired_ouputs):
        change_to_neurons = []
        for layer in self.layers:
            change_to_neurons.append(np.array([0] * layer))

        changes_to_weights = []
        for weights in self.weights:
            changes_to_weights.append(weights*0)

        changes_to_biases = []
        for bias in self.biases:
            changes_to_biases.append(bias*0)

        for i in range(len(layers)-1, 0, -1):
            if i == len(layers)-1:
                #print(layers[i], desired_ouputs)
                #print(2*(layers[i]-desired_ouputs))
                change_to_neurons[i] = 2*(layers[i]-desired_ouputs)
                #print("layer:", i, "\n",change_to_neurons[i], "\n")
                if self.debug:
                    print(i)
                    print(layers[i], desired_ouputs)
            else:
                if self.debug:
                    print(i)
                    print(self.weights[i], layers[i], self.biases[i], self.dif_sigmoid(np.dot(self.weights[i], layers[i]) + self.biases[i]), self.biases[i], change_to_neurons[i+1])

                change_to_neurons[i] = np.dot(self.dif_sigmoid(np.dot(self.weights[i], layers[i]) + self.biases[i]) * change_to_neurons[i+1], self.weights[i])
            if self.debug: 
                print("cn", change_to_neurons[i])
                print()
                #print("layer:", i, "\n",change_to_neurons[i], "\n")
            #[3,4,5,8]            [3,4,5,8]            
            #[5,6,7,3] * sigmoid'([5,6,7,3] * [0.5, 0.3, 0.7] + [3,6,2,9]) * [3,6,1,6]
            #[2,3,6,5]            [2,3,6,5]
            
            #print(change_to_neurons[i-1])
            changes_to_weights[i-1] = np.outer(self.dif_sigmoid(np.dot(self.weights[i-1], layers[i-1]) + self.biases[i-1]) * change_to_neurons[i], layers[i-1])
            if self.debug: 
                print(self.weights[i-1], layers[i-1], self.biases[i-1], self.dif_sigmoid(np.dot(self.weights[i-1], layers[i-1]) + self.biases[i-1]), change_to_neurons[i])
                print("cw", changes_to_weights[i-1])
                print()

            changes_to_biases[i-1] = self.dif_sigmoid(np.dot(self.weights[i-1], layers[i-1]) + self.biases[i-1]) * change_to_neurons[i]
            if self.debug: 
                print(self.weights[i-1], layers[i-1], self.biases[i-1], self.dif_sigmoid(np.dot(self.weights[i-1], layers[i-1]) + self.biases[i-1]), change_to_neurons[i])
                print("cb", changes_to_biases[i-1])
                print()

        #print("weights:", changes_to_weights)
        return changes_to_weights, changes_to_biases

    def train(self, training_data):
        changes_to_weights = []
        for weights in self.weights:
            changes_to_weights.append(weights*0)

        changes_to_biases = []
        for bias in self.biases:
            changes_to_biases.append(bias*0)
        cost = 0
        for i in range(len(training_data)):
            if self.debug: print("sec:", i)
            layers = self.run(training_data[i][0], True)
            _cost = self.cost_function(layers[-1], training_data[i][1])
            cost += _cost
            weights, biases = self.backprop(layers, training_data[i][1])

            for i in range(len(weights)):
                changes_to_weights[i] -= weights[i]
            
            for i in range(len(biases)):
                changes_to_biases[i] -= biases[i]

        if self.debug: print("change_to_weights", changes_to_weights)
        if self.debug: print("change_to_biases", changes_to_biases)
        print("cost:", cost/len(training_data))
   
        for i in range(len(self.weights)):
            self.weights[i] -= (changes_to_weights[i] * self.learning_factor)/len(training_data)
        
        for i in range(len(self.biases)):
            self.biases[i] -= (changes_to_biases[i] * self.learning_factor)/len(training_data)

        return changes_to_biases, changes_to_weights, cost

    def dif_sigmoid(self, array):
        return self.sigmoid(-array)*(1 - self.sigmoid(-array))

    def sigmoid(self, array):
        return 1/(1+np.exp(-array))





nn = neural_network(1, ["0", "1"], [3,3], 1, debug=False)

training_data = []


for i in range(10000):
    training_data.append(([i%2],[(i+1)%2, i%2]))


nn.test([[0],[1]], [[0,1],[1,0]])
nn.train(training_data)
nn.test([[0],[1]], [[0,1],[1,0]])

