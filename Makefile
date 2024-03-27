GPP = g++
FLAGS = -Wall -Wextra -pedantic

main: main.o neural_net.o layer.o loss.o
	$(GPP) -o main main.o neural_net.o layer.o loss.o -g

main.o: main.cpp neural_net.h layer.h loss.h
	$(GPP) -c main.cpp $(FLAGS)

neural_net.o: neural_net.cpp neural_net.h layer.h
	$(GPP) -c neural_net.cpp $(FLAGS)

layer.o: layer.cpp layer.h
	$(GPP) -c layer.cpp $(FLAGS)

loss.o: loss.cpp loss.h neural_net.h layer.h
	$(GPP) -c loss.cpp $(FLAGS)

clean:
	rm -f *.o main *~
