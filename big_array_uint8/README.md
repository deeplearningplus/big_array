

[big_array_data.py](./big_array_data.py): generate a big array in python.

[big_array.c](./big_array.c): C functions to read big array.

[example_usage.py](./example_usage.py): Example usage of using C functions in python.

[Makefile](./Makefile): Compile `big_array.c` as a shared library.

Compare the running speed of each functions:
```bash
make
./big_array

python example_usage.py

```

