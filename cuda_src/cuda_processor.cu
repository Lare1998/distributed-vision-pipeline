/*
 * cuda_processor.cu
 * Implements CUDA kernels for high-performance image processing within the distributed vision pipeline.
 * This file contains functions for grayscale conversion and basic image filtering using GPU acceleration.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// CUDA kernel for grayscale conversion
__global__ void grayscaleKernel(unsigned char* inputImage, unsigned char* outputImage, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = (y * width + x) * 3; // Assuming 3 channels (RGB)
        unsigned char r = inputImage[index];
        unsigned char g = inputImage[index + 1];
        unsigned char b = inputImage[index + 2];

        // Simple grayscale conversion: Y = 0.299R + 0.587G + 0.114B
        unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

        outputImage[y * width + x] = gray;
    }
}

// CUDA kernel for a simple box blur filter
__global__ void boxBlurKernel(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int sum = 0;
        int count = 0;

        for (int ky = -radius; ky <= radius; ++ky)
        {
            for (int kx = -radius; kx <= radius; ++kx)
            {
                int nx = x + kx;
                int ny = y + ky;

                // Check bounds
                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                {
                    sum += inputImage[ny * width + nx];
                    count++;
                }
            }
        }
        outputImage[y * width + x] = static_cast<unsigned char>(sum / count);
    }
}

// Host function to perform grayscale conversion
extern "C" void processImageGrayscale(unsigned char* h_inputImage, unsigned char* h_outputImage, int width, int height)
{
    unsigned char* d_inputImage;
    unsigned char* d_outputImage;
    size_t inputSize = width * height * 3 * sizeof(unsigned char);
    size_t outputSize = width * height * sizeof(unsigned char);

    // Allocate GPU memory
    cudaMalloc(&d_inputImage, inputSize);
    cudaMalloc(&d_outputImage, outputSize);

    // Copy input image from host to device
    cudaMemcpy(d_inputImage, h_inputImage, inputSize, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch grayscale kernel
    grayscaleKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);
    cudaDeviceSynchronize(); // Wait for kernel to complete

    // Copy output image from device to host
    cudaMemcpy(h_outputImage, d_outputImage, outputSize, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

// Host function to perform box blur
extern "C" void processImageBoxBlur(unsigned char* h_inputImage, unsigned char* h_outputImage, int width, int height, int radius)
{
    unsigned char* d_inputImage;
    unsigned char* d_outputImage;
    size_t imageSize = width * height * sizeof(unsigned char); // Assuming grayscale for blur

    // Allocate GPU memory
    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);

    // Copy input image from host to device
    cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch box blur kernel
    boxBlurKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height, radius);
    cudaDeviceSynchronize(); // Wait for kernel to complete

    // Copy output image from device to host
    cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

// Example usage (can be compiled with nvcc and linked with a C++ host program)
// int main() {
//     int width = 1920, height = 1080;
//     unsigned char* h_input = (unsigned char*)malloc(width * height * 3 * sizeof(unsigned char));
//     unsigned char* h_gray_output = (unsigned char*)malloc(width * height * sizeof(unsigned char));
//     unsigned char* h_blur_output = (unsigned char*)malloc(width * height * sizeof(unsigned char));

//     // Populate h_input with dummy RGB data (e.g., from an image file)
//     for (int i = 0; i < width * height * 3; ++i) {
//         h_input[i] = rand() % 256;
//     }

//     processImageGrayscale(h_input, h_gray_output, width, height);
//     std::cout << "Grayscale conversion complete." << std::endl;

//     processImageBoxBlur(h_gray_output, h_blur_output, width, height, 3);
//     std::cout << "Box blur complete." << std::endl;

//     free(h_input);
//     free(h_gray_output);
//     free(h_blur_output);
//     return 0;
// }
