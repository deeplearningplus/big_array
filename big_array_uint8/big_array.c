#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <immintrin.h> // For AVX2, SSE
#include <cpuid.h>
#include <time.h> // For timespec
#include <sys/time.h> // For gettimeofday
#include <omp.h>

/**
 * Get the number of elements in the packed array file
 * @param filename The path to the binary file
 * @return The number of elements (size in bytes since each element is uint8)
 *         or -1 on error
 */
size_t get_packed_array_size(const char *filename) {
    if (filename == NULL) {
        return -1;
    }

    // Open the file
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "Error opening file '%s': %s\n", 
                filename, strerror(errno));
        return -1;
    }

    // Get file size using fstat
    struct stat st;
    if (fstat(fd, &st) == -1) {
        fprintf(stderr, "Error getting file size: %s\n", strerror(errno));
        close(fd);
        return -1;
    }

    close(fd);
    return st.st_size;  // For uint8, number of elements equals file size in bytes
}

/**
 * Writes a big array to a binary file.
 * 
 * @param filename The name of the file to write to.
 * @param arr The array to write.
 * @param size The number of elements in the array.
 * @return 0 on success, -1 on failure.
 */
int write_big_array(const char *filename, const uint8_t *arr, size_t size) {
    if (filename == NULL || arr == NULL || size == 0) {
        fprintf(stderr, "Invalid arguments\n");
        return -1;
    }

    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Error opening file");
        return -1;
    }

    size_t written = fwrite(arr, sizeof(uint8_t), size, file);
    if (written != size) {
        perror("Error writing to file");
        fclose(file);
        return -1;
    }

    fclose(file);
    return 0;
}

/**
 * Reads binary data from a file into a provided array
 * 
 * @param filename The path to the binary file to read
 * @param arr Pointer to the pre-allocated array where data will be stored
 * @param size The number of elements to read
 * @return 0 on success, -1 on failure
 */
int read_big_array(const char *filename, uint8_t *arr, size_t size) {
    if (filename == NULL || arr == NULL || size == 0) {
        fprintf(stderr, "Invalid arguments\n");
        return -1;
    }

    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file '%s': %s\n", 
                filename, strerror(errno));
        return -1;
    }

    // Get file size to verify it matches the expected size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    if (file_size != size) {
        fprintf(stderr, "File size (%ld) does not match expected size (%zu)\n", 
                file_size, size);
        fclose(file);
        return -1;
    }

    // Read the entire file into the provided array
    size_t read_size = fread(arr, 1, size, file);
    if (read_size != size) {
        fprintf(stderr, "Read error: expected %zu bytes, got %zu bytes\n", 
                size, read_size);
        fclose(file);
        return -1;
    }

    fclose(file);
    return 0;
}

/**
 * Utility function to print a preview of the array contents
 * 
 * @param arr The array to print
 * @param size Total size of the array
 * @param preview_size Number of elements to print
 */
void print_array_preview(const uint8_t *arr, size_t size, size_t preview_size) {
    if (arr == NULL || size == 0) {
        printf("Array is invalid\n");
        return;
    }

    size_t n = (preview_size < size) ? preview_size : size;
    printf("First %zu elements:\n", n);
    for (size_t i = 0; i < n; i++) {
        printf("%3u ", arr[i]);
        if ((i + 1) % 10 == 0) printf("\n");
    }
    printf("\n");
}



/**
 * Structure to hold memory-mapped array information
 */
typedef struct {
    uint8_t *data;      // Pointer to the memory-mapped data
    size_t size;        // Size of the array in bytes
    int fd;             // File descriptor
} MappedArray;

/**
 * Opens and memory-maps a binary file for random access
 *
 * @param filename The path to the binary file
 * @return MappedArray structure or NULL on failure
 */
MappedArray* mmap_big_array(const char *filename) {
    MappedArray *marray = (MappedArray*)malloc(sizeof(MappedArray));
    if (marray == NULL) {
        fprintf(stderr, "Failed to allocate MappedArray structure\n");
        return NULL;
    }

    // Open the file
    marray->fd = open(filename, O_RDONLY);
    if (marray->fd == -1) {
        fprintf(stderr, "Error opening file '%s': %s\n",
                filename, strerror(errno));
        free(marray);
        return NULL;
    }

    // Get file size
    struct stat sb;
    if (fstat(marray->fd, &sb) == -1) {
        fprintf(stderr, "Error getting file size: %s\n", strerror(errno));
        close(marray->fd);
        free(marray);
        return NULL;
    }
    marray->size = sb.st_size;

    // Memory map the file
    marray->data = (uint8_t*)mmap(NULL, marray->size,
                                 PROT_READ, MAP_SHARED,
                                 marray->fd, 0);
    if (marray->data == MAP_FAILED) {
        fprintf(stderr, "Error memory-mapping file: %s\n", strerror(errno));
        close(marray->fd);
        free(marray);
        return NULL;
    }

    return marray;
}

/**
 * Closes the memory-mapped array and frees resources
 *
 * @param marray Pointer to MappedArray structure
 */
void close_mmap_array(MappedArray *marray) {
    if (marray != NULL) {
        if (marray->data != NULL && marray->data != MAP_FAILED) {
            munmap(marray->data, marray->size);
        }
        if (marray->fd != -1) {
            close(marray->fd);
        }
        free(marray);
    }
}

/**
 * Reads a value at a specific index from the memory-mapped array
 *
 * @param marray Pointer to MappedArray structure
 * @param index Index to read from
 * @param value Pointer to store the read value
 * @return 0 on success, -1 on failure
 */
int read_at_index(const MappedArray *marray, size_t index, uint8_t *value) {
    if (marray == NULL || marray->data == NULL || value == NULL) {
        return -1;
    }
    if (index >= marray->size) {
        fprintf(stderr, "Index %zu out of bounds (size: %zu)\n",
                index, marray->size);
        return -1;
    }
    *value = marray->data[index];
    return 0;
}

/**
 * Reads a range of values from the memory-mapped array
 *
 * @param marray Pointer to MappedArray structure
 * @param start Starting index
 * @param length Number of elements to read
 * @param buffer Pre-allocated buffer to store the values
 * @return 0 on success, -1 on failure
 */
int read_range(const MappedArray *marray, size_t start,
               size_t length, uint8_t *buffer) {
    if (marray == NULL || marray->data == NULL || buffer == NULL) {
        return -1;
    }
    if (start + length > marray->size) {
        fprintf(stderr, "Range [%zu, %zu) out of bounds (size: %zu)\n",
                start, start + length, marray->size);
        return -1;
    }
    memcpy(buffer, marray->data + start, length);
    return 0;
}

// SIMD-enabled methods
// CPU feature detection
static int has_avx2 = 0;
static int has_sse4_1 = 0;

void init_cpu_features(void) {
    unsigned int eax, ebx, ecx, edx;
    
    // Check SSE4.1 support
    __cpuid(1, eax, ebx, ecx, edx);
    has_sse4_1 = (ecx & bit_SSE4_1) != 0;
    
    // Check AVX2 support
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    has_avx2 = (ebx & bit_AVX2) != 0;
}

/**
 * SIMD-optimized read at index - primarily for demonstration
 * Note: For single value reads, SIMD might not provide benefits
 */
int read_at_index_simd(const MappedArray *marray, size_t index, uint8_t *value) {
    if (marray == NULL || marray->data == NULL || value == NULL) {
        return -1;
    }
    if (index >= marray->size) {
        fprintf(stderr, "Index %zu out of bounds (size: %zu)\n",
                index, marray->size);
        return -1;
    }

    // For single value reads, direct access is typically most efficient
    *value = marray->data[index];
    return 0;
}


/**
 * Simplified SIMD implementation
**/
int read_range_simd(const MappedArray *marray, size_t start,
                   size_t length, uint8_t *buffer) {
    if (marray == NULL || marray->data == NULL || buffer == NULL ||
        start + length > marray->size) {
        return -1;
    }

    const uint8_t *src = marray->data + start;
    uint8_t *dst = buffer;
    size_t i = 0;

    // Process 32 bytes at a time using AVX2
    for (; i + 32 <= length; i += 32) {
        __m256i data = _mm256_loadu_si256((const __m256i*)(src + i));
        _mm256_storeu_si256((__m256i*)(dst + i), data);
    }

    // Handle remaining bytes
    if (i < length) {
        memcpy(dst + i, src + i, length - i);
    }

    return 0;
}

/**
 * OpenMP + SIMD optimized implementation
 *
 * Remember to set the number of threads, e.g.,
 * export OMP_NUM_THREADS=2
 *
 * Set different values of OMP_NUM_THREADS and 
 * +see the benchmarked results from benchmark_read_range.
 *
 * OMP_NUM_THREADS=2 often gives good performance.
 * Set it to higher number if the range to fetch is super big.
 *
 */
int read_range_simd_parallel(const MappedArray *marray, size_t start,
                   size_t length, uint8_t *buffer) {
    if (marray == NULL || marray->data == NULL || buffer == NULL ||
        start + length > marray->size) {
        return -1;
    }

    const uint8_t *src = marray->data + start;
    uint8_t *dst = buffer;

    // Minimum chunk size per thread (32KB)
    const size_t min_chunk_size = 32 * 1024;

    // Calculate chunk size based on length
    size_t chunk_size = length / omp_get_max_threads();
    chunk_size = (chunk_size + 31) & ~31; // Round up to multiple of 32 bytes
    chunk_size = chunk_size < min_chunk_size ? min_chunk_size : chunk_size;

    #pragma omp parallel
    {
        // Get thread-specific bounds
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        size_t thread_start = thread_id * chunk_size;
        size_t thread_end = (thread_id == num_threads - 1) ? length :
                           (thread_id + 1) * chunk_size;

        // Ensure thread_start doesn't exceed length
        if (thread_start < length) {
            // Align thread_end to 32-byte boundary for AVX2
            thread_end = (thread_end > length) ? length : thread_end;
            size_t thread_i = thread_start;

            // Process 128 bytes at a time using AVX2
            for (; thread_i + 128 <= thread_end; thread_i += 128) {
                // Load and store first 32 bytes
                __m256i data1 = _mm256_loadu_si256((const __m256i*)(src + thread_i));
                _mm256_storeu_si256((__m256i*)(dst + thread_i), data1);

                // Load and store next 32 bytes
                __m256i data2 = _mm256_loadu_si256((const __m256i*)(src + thread_i + 32));
                _mm256_storeu_si256((__m256i*)(dst + thread_i + 32), data2);

                // Load and store next 32 bytes
                __m256i data3 = _mm256_loadu_si256((const __m256i*)(src + thread_i + 64));
                _mm256_storeu_si256((__m256i*)(dst + thread_i + 64), data3);

                // Load and store final 32 bytes
                __m256i data4 = _mm256_loadu_si256((const __m256i*)(src + thread_i + 96));
                _mm256_storeu_si256((__m256i*)(dst + thread_i + 96), data4);
            }

            // Process remaining 32-byte chunks
            for (; thread_i + 32 <= thread_end; thread_i += 32) {
                __m256i data = _mm256_loadu_si256((const __m256i*)(src + thread_i));
                _mm256_storeu_si256((__m256i*)(dst + thread_i), data);
            }

            // Handle remaining bytes in this thread's chunk
            if (thread_i < thread_end) {
                memcpy(dst + thread_i, src + thread_i, thread_end - thread_i);
            }
        }
    }

    return 0;
}

// Helper function to get current time in microseconds
static uint64_t get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}


/**
 * Simple benchmark function
 */
void benchmark_read_range(const MappedArray *marray, size_t start, 
                        size_t length, int iterations) {
    uint8_t *buffer1 = (uint8_t*)malloc(length);
    uint8_t *buffer2 = (uint8_t*)malloc(length);
    
    if (buffer1 == NULL || buffer2 == NULL) {
        fprintf(stderr, "Failed to allocate benchmark buffers\n");
        return;
    }

    // Warm up
    read_range(marray, start, length, buffer1);
    read_range_simd(marray, start, length, buffer2);

    // Benchmark standard version
    uint64_t start_time = get_time_us();
    for (int i = 0; i < iterations; i++) {
        read_range(marray, start, length, buffer1);
    }
    uint64_t end_time = get_time_us();
    double standard_time = (end_time - start_time) / 1000000.0;

    // Benchmark SIMD version
    start_time = get_time_us();
    for (int i = 0; i < iterations; i++) {
        read_range_simd(marray, start, length, buffer2);
    }
    end_time = get_time_us();
    double simd_time = (end_time - start_time) / 1000000.0;

    // Verify results match
    int mismatch = memcmp(buffer1, buffer2, length) != 0;

    printf("\nBenchmark Results (iterations: %d, length: %zu):\n", 
           iterations, length);
    printf("Standard Implementation: %.6f seconds\n", standard_time);
    printf("SIMD Implementation:     %.6f seconds\n", simd_time);
    printf("Speedup:                %.2fx\n", standard_time / simd_time);
    printf("Results match:          %s\n", mismatch ? "No" : "Yes");
    printf("Throughput (Standard):  %.2f MB/s\n", 
           (length * iterations) / (standard_time * 1024 * 1024));
    printf("Throughput (SIMD):      %.2f MB/s\n", 
           (length * iterations) / (simd_time * 1024 * 1024));

    free(buffer1);
    free(buffer2);
}


// Example usage
int main() {
	// Initialize CPU feature detection
    init_cpu_features();
    printf("CPU Features:\n");
    printf("AVX2:    %s\n", has_avx2 ? "Yes" : "No");
    printf("SSE4.1:  %s\n", has_sse4_1 ? "Yes" : "No");

    const char *filename = "big_array.bin";
    const size_t array_size = get_packed_array_size(filename);
    printf("Array size: %ld, %f MB\n", array_size, array_size/1e6);
//    const size_t array_size = 1000000000;  // 1e6 elements

/*
    // Allocate memory for the array
    uint8_t *array = (uint8_t *)malloc(array_size);
    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    // Read the file
    if (read_big_array(filename, array, array_size) != 0) {
        free(array);
        return 1;
    }
    // Print first 20 elements as a preview
    print_array_preview(array, array_size, 10);
    // Clean up
    free(array);
*/


	// Open memory-mapped array
    MappedArray *marray = mmap_big_array(filename);
    if (marray == NULL) {
        return 1;
    }
    printf("Successfully mapped file of size: %zu bytes\n", marray->size);

    // Example 1: Read individual values
    uint8_t value;
    printf("\nReading individual values:\n");
    for (size_t i = 0; i < 10; i++) {
        if (read_at_index(marray, i, &value) == 0) {
            printf("%u ", value);
        }
    }
    printf("\nReading individual values (SIMD):\n");
    for (size_t i = 0; i < 10; i++) {
        if (read_at_index_simd(marray, i, &value) == 0) {
            printf("%u ", value);
        }
    }

	// Example 2: Read a range of values
    printf("\nReading a range of values:\n");
    size_t range_size = 10;
    uint8_t buffer[range_size];
    if (read_range(marray, 0, range_size, buffer) == 0) {
        for (size_t i = 0; i < range_size; i++) {
            printf("%u ", buffer[i]);
        }
		printf("\n");
    }
    printf("Reading a range of values (SIMD):\n");
    if (read_range(marray, 0, range_size, buffer) == 0) {
        for (size_t i = 0; i < range_size; i++) {
            printf("%u ", buffer[i]);
        }
		printf("\n");
    }

	// Test different buffer sizes
    size_t test_sizes[] = {
        4096,      // 4 KB
        16384,     // 16 KB
        65536,     // 64 KB
        262144,    // 256 KB
        1048576    // 1 MB
    };

    for (size_t i = 0; i < sizeof(test_sizes)/sizeof(test_sizes[0]); i++) {
		size_t test_size = test_sizes[i];
        if (test_size <= marray->size) {
            benchmark_read_range(marray, 0, test_size, 1000);
        }
    }


    close_mmap_array(marray);


    return 0;
}




