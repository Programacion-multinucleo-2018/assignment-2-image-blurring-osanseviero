// Implements image box blur using CPU with threads with OpenMP.
// Compile with g++ blur_threads.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -std=c++11

#include <iostream>
#include <cstdio>
#include <cmath>
#include <omp.h>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// Blur input image
void blur(const cv::Mat& input, cv::Mat& output) {
    std::cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << std::endl;

    int i, j, filterX, filterY;

    // Iterate over image Matrix
    #pragma omp parallel for private(i, j, filterX, filterY) shared(input, output)
    for(i = 0; i < input.rows; i++) {
        for(j = 0; j < input.cols; j++) {
            int blue = 0;
            int green = 0;
            int red = 0;
            int count = 0;

            // Check neighboring pixels
            for(filterX=-2; filterX<3; filterX++) {
                for(filterY=-2; filterY<3; filterY++) {
                    int xidx = i+filterX;
                    int yidx = j+filterY;

                    // Manage borders
                    if((yidx >0 && yidx < input.cols) && (xidx>0 && xidx < input.rows)) {
                        blue += input.at<cv::Vec3b>(xidx, yidx)[0];
                        green += input.at<cv::Vec3b>(xidx, yidx)[1];
                        red += input.at<cv::Vec3b>(xidx, yidx)[2];
                        count++;
                    }
                }
            }
            // Calculate average
            output.at<cv::Vec3b>(i, j)[0] = blue/count;
            output.at<cv::Vec3b>(i, j)[1] = green/count;
            output.at<cv::Vec3b>(i, j)[2] = red/count;
        }
    }
}

int main(int argc, char *argv[]) {
    // Read input image
    std::string imagePath = "image.jpg";
    cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);
    cv::Mat output(input.rows, input.cols, CV_8UC3);

    // Blur image
    auto start_cpu =  std::chrono::high_resolution_clock::now();
    blur(input, output);
    auto end_cpu =  std::chrono::high_resolution_clock::now();

    // Measure total time
    std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    printf("blur elapsed %f ms\n", duration_ms.count());

    namedWindow("Input", cv::WINDOW_NORMAL);
    namedWindow("Output", cv::WINDOW_NORMAL);

    imshow("Input", input);
    imshow("Output", output);
    cv::waitKey();

    return 0;
}