CXX=g++
CXXFLAGS=-std=c++17 -O3 -DNDEBUG

all: conll2vec

conll2vec : src/conll2vec.cpp
	$(CXX) src/conll2vec.cpp -o conll2vec $(CXXFLAGS) -pthread -licuuc

clean:
	rm -rf conll2vec
