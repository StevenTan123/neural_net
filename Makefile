GPP = g++
FLAGS = -Wall -Wextra -pedantic

main: main.o neural_net.o layer.o loss.o MNIST_loader.o
	$(GPP) -o main main.o neural_net.o layer.o loss.o MNIST_loader.o -g

main.o: main.cpp neural_net.h layer.h loss.h
	$(GPP) -c main.cpp $(FLAGS)

neural_net.o: neural_net.cpp neural_net.h layer.h
	$(GPP) -c neural_net.cpp $(FLAGS)

layer.o: layer.cpp layer.h
	$(GPP) -c layer.cpp $(FLAGS)

loss.o: loss.cpp loss.h neural_net.h layer.h
	$(GPP) -c loss.cpp $(FLAGS)

MNIST_loader.o: MNIST_loader.cpp MNIST_loader.h
	$(GPP) -c MNIST_loader.cpp $(FLAGS)

clean:
	rm -f *.o main *~
