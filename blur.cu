// Implements image box blur using CPU.
// Compile with nvcc blur.cu -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -std=c++11

#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// Blur input image
__global__ void blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int step) {
    //2D Index of current thread
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if((xIndex<width) && (yIndex<height)) {
        //Location of pixel in input and output
        const int image_tid = yIndex * step + (3 * xIndex);
        int blue = 0;
        int green = 0;
        int red = 0;
        int count = 0;

        // Apply filter
        for(int filterX=-2; filterX<3; filterX++) {
            for(int filterY=-2; filterY<3; filterY++) {
                int tid = (yIndex+filterY) * step + (3 * (xIndex+filterX));

                // Manage borders
                if((xIndex+filterX)%width>1 && (yIndex+filterY)%height>1) {
                    blue += input[tid];
                    green += input[tid+1];
                    red += input[tid+2];
                    count++;
                }
            }
        }
        output[image_tid] = static_cast<unsigned char>(blue/count);
        output[image_tid+1] = static_cast<unsigned char>(green/count);
        output[image_tid+2] = static_cast<unsigned char>(red/count);
    }
}


void blur(const cv::Mat& input, cv::Mat& output) {
    std::cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << std::endl;

    const int bytes = input.step * input.rows;
    unsigned char *d_input, *d_output;

    //Allocate device memory
    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, bytes);

    //Copy data from OpenCV input image to device memory
    cudaMemcpy(d_input,input.ptr(), bytes, cudaMemcpyHostToDevice);

    //Specify block size
    const dim3 block(32, 32);

    //Calculate grid size to cover the whole image
    const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));
    printf("grid.x %d grid.y %d block.x %d block.y %d\n", grid.x, grid.y, block.x, block.y);

    // Blur image
    auto start_cpu =  std::chrono::high_resolution_clock::now();
    blur_kernel<<<grid, block>>>(d_input, d_output, input.cols, input.rows, input.step);
    auto end_cpu =  std::chrono::high_resolution_clock::now();

    // Measure total time
    std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    printf("blur_kernel elapsed %f ms\n", duration_ms.count());

    // Sync
    cudaDeviceSynchronize();

    // Copy memory from device to host
    cudaMemcpy(output.ptr(),d_output,bytes,cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char *argv[]) {
    // Read input image
    std::string imagePath = "image.jpg";
    cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);
    cv::Mat output(input.rows, input.cols, CV_8UC3);

    // Blur image
    blur(input, output);

    namedWindow("Input", cv::WINDOW_NORMAL);
    namedWindow("Output", cv::WINDOW_NORMAL);

    imshow("Input", input);
    imshow("Output", output);
    
    cv::waitKey();
    
    return 0;
}