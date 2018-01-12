from numpy import exp,array,random,dot
class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2*random.random((3,1))-1
    def __sigmoid(self,x):
        return 1/(1+exp(-x))
    def __sigmoid_derivation(self,x):
        return x*(1-x)
    def train(self, training_set_inputs,training_set_output,number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.think(training_set_inputs)
            error = training_set_output-output
            adjustment = dot(training_set_inputs.T,error*self.__sigmoid_derivation(output))
            self.synaptic_weights += adjustment
    def think(self, inputs):
        dot_product = dot(inputs, self.synaptic_weights)
        return self.__sigmoid(dot_product)
if __name__ == "__main__":
    neural_network = NeuralNetwork()
    print("\n\nRandom starting synaptic weights: ")
    print(neural_network.synaptic_weights)
    training_set_input = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T
    neural_network.train(training_set_input,training_set_outputs,10000)
    print ("\nNew synaptic weights after trainig: ")
    print(neural_network.synaptic_weights)
    print(neural_network.think(array([1,0,0])))






























    
    
































































    
