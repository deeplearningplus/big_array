all:
	# Compile as shared library
	gcc -O3 -fopenmp -mavx2 -shared -o libbig_array.so -fPIC big_array.c -O3 -march=native -mfma -flto -fopenmp
