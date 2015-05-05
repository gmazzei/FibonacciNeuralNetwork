require "rubygems"
require "ai4r"

net = Ai4r::NeuralNetwork::Backpropagation.new([2, 3, 4, 3, 1])

net.set_parameters( 
    :momentum => 0.15, 
    :learning_rate => 0.1,
    :propagation_function => lambda { |x| Math.tanh(x) },
    :derivative_propagation_function => lambda { |y| 1.0 - y**2 }
    )

# Train the network 
1000000.times do |i|
	net.train([0,0], [0])
    net.train([0,1], [1])
    net.train([1,0], [1])
    error = net.train([1,1], [0])

    puts "Error after iteration #{i}:\t#{error}" if i%20000 == 0
end


puts "Test data"
puts "[0,0] = > #{net.eval([0,0]).inspect}"
puts "[0,1] = > #{net.eval([0,1]).inspect}"
puts "[1,0] = > #{net.eval([1,0]).inspect}"
puts "[1,1] = > #{net.eval([1,1]).inspect}"


