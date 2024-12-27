import ctypes
import numpy as np
import os
import time

os.environ["OMP_NUM_THREADS"] = "2"

# Load the shared library
lib = ctypes.CDLL(os.path.abspath("libbig_array.so"))

# Define the argument and return types for the C functions
lib.get_packed_array_size.argtypes = [ctypes.c_char_p]
lib.get_packed_array_size.restype = ctypes.c_size_t

lib.write_big_array.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint16), ctypes.c_size_t]
lib.write_big_array.restype = ctypes.c_int

lib.read_big_array.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint16), ctypes.c_size_t]
lib.read_big_array.restype = ctypes.c_int

class MappedArray(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_uint16)),
                ("size", ctypes.c_size_t),
                ("fd", ctypes.c_int)]

lib.mmap_big_array.argtypes = [ctypes.c_char_p]
lib.mmap_big_array.restype = ctypes.POINTER(MappedArray)

lib.close_mmap_array.argtypes = [ctypes.POINTER(MappedArray)]
lib.close_mmap_array.restype = None

lib.read_at_index.argtypes = [ctypes.POINTER(MappedArray), ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint16)]
lib.read_at_index.restype = ctypes.c_int

lib.read_range.argtypes = [ctypes.POINTER(MappedArray), ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint16)]
lib.read_range.restype = ctypes.c_int

lib.read_range_simd.argtypes = [ctypes.POINTER(MappedArray), ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint16)]
lib.read_range_simd.restype = ctypes.c_int

lib.read_range_simd_parallel.argtypes = [ctypes.POINTER(MappedArray), ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint16)]
lib.read_range_simd_parallel.restype = ctypes.c_int

# Example usage
def example_usage():
    filename = "big_array.bin"

    # Create a numpy array and write it to a file
    #rng = np.random.RandomState(42)
    #array_size = int(1e9)
    #big_array = rng.randint(0, 256, array_size, dtype=np.uint16)
    #test_filename = "big_array_c.bin"
    #if lib.write_big_array(test_filename.encode('utf-8'), big_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), array_size) != 0:
    #    print("Error writing big array to file")
    #    return

    # Read the array size from the file
    num_elements = lib.get_packed_array_size(filename.encode('utf-8'))
    if num_elements == -1:
        print("Error getting packed array size")
        return
    print(f"Number of elements: {num_elements}")

    array_size = num_elements

    # Read the array from the file
    #read_array = np.zeros(array_size, dtype=np.uint16)
    #if lib.read_big_array(filename.encode('utf-8'), read_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), array_size) != 0:
    #    print("Error reading big array from file")
    #    return

    # Memory map the file
    marray = lib.mmap_big_array(filename.encode('utf-8'))
    if not marray:
        print("Error memory-mapping the file")
        return
    print(f"Memory-mapped array size: {marray.contents.size}")

    # Read a value at a specific index
    index = 10
    value = ctypes.c_uint16()
    if lib.read_at_index(marray, index, ctypes.byref(value)) != 0:
        print("Error reading value at index")
        lib.close_mmap_array(marray)
        return
    print(f"Value at index {index}: {value.value}")

    start = 100
    length = int(1e8)

    # Read with np.memmap
    tic = time.time()
    a = np.memmap(filename, dtype=np.uint16, mode="r")
    true_value = np.array(a[start: start + length])
    print(f"np.memmap: {time.time() - tic:.6f} secs")
    print(f"Value at index (np.memmap) 10: {a[10]}")

    # Read a range of values
    tic = time.time()
    buffer = np.zeros(length, dtype=np.uint16)
    if lib.read_range(marray, start, length, buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))) != 0:
        print("Error reading range of values")
        lib.close_mmap_array(marray)
        return
    #print(f"Values from {start} to {start + length}: {buffer}")
    print(f"read_range: {time.time() - tic:.6f} secs, identifical: {(true_value == buffer).all()}")


    # Use SIMD read range
    tic = time.time()
    if lib.read_range_simd(marray, start, length, buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))) != 0:
        print("Error reading range of values with SIMD parallel")
        lib.close_mmap_array(marray)
        return
    #print(f"Values from {start} to {start + length} (SIMD parallel): {buffer}")
    print(f"read_range_simd: {time.time() - tic:.6f} secs, identifical: {(true_value == buffer).all()}")


    # Use SIMD and parallel read range
    tic = time.time()
    if lib.read_range_simd_parallel(marray, start, length, buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))) != 0:
        print("Error reading range of values with SIMD parallel")
        lib.close_mmap_array(marray)
        return
    #print(f"Values from {start} to {start + length} (SIMD parallel): {buffer}")
    print(f"read_range_simd_parallel: {time.time() - tic:.6f} secs, identifical: {(true_value == buffer).all()}")


    # Close the memory-mapped array
    lib.close_mmap_array(marray)

if __name__ == "__main__":
    example_usage()
