require "rubygems"
require "ai4r"


#Creating Neural Network
#Input layer: 1
#First Hidden layer: 5
#Second Hidden layer: 10
#Third Hidden layer: 5
#Output layer: 6

net = Ai4r::NeuralNetwork::Backpropagation.new([1, 10, 10, 10, 10, 10, 6])

net.set_parameters( 
    :learning_rate => 0.15,
    :propagation_function => lambda { |x| Math.tanh(x) },
    :derivative_propagation_function => lambda { |y| 1.0 - y**2 }
)



puts "Training the network"
20000.times do |i|

    net.train([1], [0.02, 0.03, 0.05, 0.08, 0.13, 0.21])
    net.train([2], [0.03, 0.05, 0.08, 0.13, 0.21, 0.34])
    net.train([3], [0.05, 0.08, 0.13, 0.21, 0.34, 0.55])
    error = net.train([5], [0.08, 0.13, 0.21, 0.34, 0.55, 0.89])

    puts "Error after iteration #{i}:\t#{error}\n" if i % 2000 == 0
end




puts "Testing data"
fibonacci_series = [1, 2, 3, 5]

fibonacci_series.each do |number|
    puts "Input: #{number}  =>  Output: #{net.eval([number]).map { |element| (element.to_f * 100).round }.inspect}"
end