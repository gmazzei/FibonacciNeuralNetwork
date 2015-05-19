require "rubygems"
require "ai4r"

=begin

1) Creacion de Red Neuronal Artificial

Utiliza el metodo Backpropagation.
Las capas tienen diferentes cantidades de neuronas, a continuacion se detalla:
1ra capa: 1 neurona (igual a la cantidad de inputs)
2da capa: 10 neuronas
3ra capa: 10 neuronas
4ta capa: 10 neuronas
5ta capa: 10 neuronas
6ta capa: 10 neuronas
7ma capa: 6 neuronas (igual a la cantidad de outputs)

=end


net = Ai4r::NeuralNetwork::Backpropagation.new([1, 10, 10, 10, 10, 10, 6])





=begin

2) Se setean los parametros de la red.
Se elige un coeficiente de aprendizaje de 0.15
Se cambia la funcion sigmoidal (seteada por defecto) por la funcion tanh (tangente hiperbolica),
ya que esta ultima alcanza los resultados deseados mas rapidamente.
La ultima funcion en los parametros es la derivada de la funcion tanh.

=end

net.set_parameters( 
    :learning_rate => 0.15,
    :propagation_function => lambda { |x| Math.tanh(x) },
    :derivative_propagation_function => lambda { |y| 1.0 - y**2 }
)


=begin

3) Se itera para entrenar la RNA
Los valores de salida se dividen por 100 a comparacion con los reales, 
ya que la funcion tanh tiene imagen entre -1 y 1.

Ejemplo:
Para Input = 1, el output real deberia ser 2, 3, 5, 8, 13 y 21, pero se divide por 100 ya que la red no puede
devolver valores mayores a 1.  


Luego, cada 2000 iteraciones se imprime por consola el error aproximado de los resultados.

=end

puts "Entrenando la RNA"
20000.times do |i|

    net.train([1], [0.02, 0.03, 0.05, 0.08, 0.13, 0.21])
    net.train([2], [0.03, 0.05, 0.08, 0.13, 0.21, 0.34])
    net.train([3], [0.05, 0.08, 0.13, 0.21, 0.34, 0.55])
    error = net.train([5], [0.08, 0.13, 0.21, 0.34, 0.55, 0.89])

    puts "Error luego de la iteracion #{i}:\t#{error}\n" if i % 2000 == 0
end



=begin

4) Se prueba la red con los valores de la serie.

Para cada valor de la serie, se obtiene el output de la RNA, luego se multiplica cada valor resultado por 100 y 
se redondean al entero mas cercano.

=end

puts "Probando la Red"
fibonacci_series = [1, 2, 3, 5]

fibonacci_series.each do |number|

	result = net.eval([number])
	roundedResult = result.map { |element| (element.to_f * 100).round }

    puts "Input: #{number}  =>  Output: #{roundedResult.inspect}"
end
