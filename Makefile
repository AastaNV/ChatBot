all:
	swig -c++ -python src/tensorNet.i 
	nvcc -std=c++11 -O3 --compiler-options '-fPIC' -c src/tensorNet.cpp -Isrc/
	nvcc -std=c++11 -O3 --compiler-options '-fPIC' -c src/tensorNet_wrap.cxx -I/usr/include/python2.7/ -Isrc/
	nvcc -shared tensorNet.o tensorNet_wrap.o -lnvinfer -lnvparsers -lnvinfer_plugin -o src/_tensorNet.so
	rm *.o
