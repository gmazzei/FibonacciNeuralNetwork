# FibonacciNeuralNetwork
Red Neuronal para Inteligencia Artificial

Pasos para correrlo en Linux:

1) Instalar dependencias anteriores a Ruby

sudo apt-get update

sudo apt-get install git-core curl zlib1g-dev build-essential libssl-dev libreadline-dev libyaml-dev libsqlite3-dev sqlite3 libxml2-dev libxslt1-dev libcurl4-openssl-dev python-software-properties libffi-dev


2) Instalar Ruby

cd

git clone git://github.com/sstephenson/rbenv.git .rbenv

echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.bashrc

echo 'eval "$(rbenv init -)"' >> ~/.bashrc

exec $SHELL


git clone git://github.com/sstephenson/ruby-build.git ~/.rbenv/plugins/ruby-build

echo 'export PATH="$HOME/.rbenv/plugins/ruby-build/bin:$PATH"' >> ~/.bashrc

exec $SHELL

git clone https://github.com/sstephenson/rbenv-gem-rehash.git ~/.rbenv/plugins/rbenv-gem-rehash


rbenv install 2.2.2

rbenv global 2.2.2

ruby -v


echo "gem: --no-ri --no-rdoc" > ~/.gemrc

gem install bundler

3) Instalar AI4R

gem install ai4r


4) Bajar el script, ir al directorio y ejecutarlo.

ruby fibonacci.rb


Link al framework AI4R (Inteligencia Artificial):
http://ai4r.org/
