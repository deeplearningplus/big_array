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
 * @return The number of elements (size in bytes divided by 4 since each element is uint32)
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
    return st.st_size / sizeof(uint32_t);  // For uint32, number of elements equals file size in bytes divided by 4
}

/**
 * Writes a big array to a binary file.
 * 
 * @param filename The name of the file to write to.
 * @param arr The array to write.
 * @param size The number of elements in the array.
 * @return 0 on success, -1 on failure.
 */
int write_big_array(const char *filename, const uint32_t *arr, size_t size) {
    if (filename == NULL || arr == NULL || size == 0) {
        fprintf(stderr, "Invalid arguments\n");
        return -1;
    }

    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Error opening file");
        return -1;
    }

    size_t written = fwrite(arr, sizeof(uint32_t), size, file);
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
int read_big_array(const char *filename, uint32_t *arr, size_t size) {
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

    if (file_size != size * sizeof(uint32_t)) {
        fprintf(stderr, "File size (%ld) does not match expected size (%zu)\n",
                file_size, size * sizeof(uint32_t));
        fclose(file);
        return -1;
    }

    // Read the entire file into the provided array
    size_t read_size = fread(arr, sizeof(uint32_t), size, file);
    if (read_size != size) {
        fprintf(stderr, "Read error: expected %zu elements, got %zu elements\n",
                size, read_size);
        fclose(file);
        return -1;
    }

    fclose(file);
    return 0;
}

/**
 * Structure to hold memory-mapped array information
 */
typedef struct {
    uint32_t *data;     // Pointer to the memory-mapped data
    size_t size;        // Size of the array in elements
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
    marray->size = sb.st_size / sizeof(uint32_t);

    // Memory map the file
    marray->data = (uint32_t*)mmap(NULL, sb.st_size,
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
            munmap(marray->data, marray->size * sizeof(uint32_t));
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
int read_at_index(const MappedArray *marray, size_t index, uint32_t *value) {
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
               size_t length, uint32_t *buffer) {
    if (marray == NULL || marray->data == NULL || buffer == NULL) {
        return -1;
    }
    if (start + length > marray->size) {
        fprintf(stderr, "Range [%zu, %zu) out of bounds (size: %zu)\n",
                start, start + length, marray->size);
        return -1;
    }
    memcpy(buffer, marray->data + start, length * sizeof(uint32_t));
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
int read_at_index_simd(const MappedArray *marray, size_t index, uint32_t *value) {
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
                   size_t length, uint32_t *buffer) {
    if (marray == NULL || marray->data == NULL || buffer == NULL ||
        start + length > marray->size) {
        return -1;
    }

    const uint32_t *src = marray->data + start;
    uint32_t *dst = buffer;
    size_t i = 0;

    // Process 8 elements (32 bytes) at a time using AVX2
    for (; i + 8 <= length; i += 8) {
        __m256i data = _mm256_loadu_si256((const __m256i*)(src + i));
        _mm256_storeu_si256((__m256i*)(dst + i), data);
    }

    // Handle remaining elements
    if (i < length) {
        memcpy(dst + i, src + i, (length - i) * sizeof(uint32_t));
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
 * see the benchmarked results from benchmark_read_range.
 *
 * OMP_NUM_THREADS=2 often gives good performance.
 * Set it to higher number if the range to fetch is super big.
 *
 */
int read_range_simd_parallel(const MappedArray *marray, size_t start,
                   size_t length, uint32_t *buffer) {
    if (marray == NULL || marray->data == NULL || buffer == NULL ||
        start + length > marray->size) {
        return -1;
    }

    const uint32_t *src = marray->data + start;
    uint32_t *dst = buffer;

    // Minimum chunk size per thread (32KB)
    const size_t min_chunk_size = 32 * 1024 / sizeof(uint32_t);

    // Calculate chunk size based on length
    size_t chunk_size = length / omp_get_max_threads();
    chunk_size = (chunk_size + 7) & ~7; // Round up to multiple of 8 elements
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
            // Align thread_end to 8-element boundary for AVX2
            thread_end = (thread_end > length) ? length : thread_end;
            size_t thread_i = thread_start;

            // Process 128 bytes (32 elements) at a time using AVX2
            for (; thread_i + 32 <= thread_end; thread_i += 32) {
                // Load and store first 16 elements
                __m256i data1 = _mm256_loadu_si256((const __m256i*)(src + thread_i));
                _mm256_storeu_si256((__m256i*)(dst + thread_i), data1);

                // Load and store next 16 elements
                __m256i data2 = _mm256_loadu_si256((const __m256i*)(src + thread_i + 16));
                _mm256_storeu_si256((__m256i*)(dst + thread_i + 16), data2);
            }

            // Process remaining 8-element chunks
            for (; thread_i + 8 <= thread_end; thread_i += 8) {
                __m256i data = _mm256_loadu_si256((const __m256i*)(src + thread_i));
                _mm256_storeu_si256((__m256i*)(dst + thread_i), data);
            }

            // Handle remaining elements in this thread's chunk
            if (thread_i < thread_end) {
                memcpy(dst + thread_i, src + thread_i, (thread_end - thread_i) * sizeof(uint32_t));
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
