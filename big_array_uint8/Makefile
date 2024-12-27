All:
	#gcc -o big_array big_array.c -Wall -O3 -mavx2 -O3 -march=native -mfma -flto -fopenmp
	gcc -o big_array big_array.c -Wall -O3 -mavx2 -msse4 -O3 -march=native -mfma -flto -fopenmp

	# Compile as shared library
	gcc -O3 -fopenmp -mavx2 -shared -o libbig_array.so -fPIC big_array.c -O3 -march=native -mfma -flto -fopenmp
