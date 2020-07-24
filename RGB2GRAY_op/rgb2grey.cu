#include<iostream>
#include<string>
#include<cassert>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>

#include<cuda_runtime.h>
#include<cuda.h>
#include<cuda_runtime_api.h>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4 *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows(){ return imageRGBA.rows; }
size_t numCols(){ return imageRGBA.cols; }

template<typename T>
void check(T err, const char* const func, const char* const file, const int line){
    if (err != cudaSuccess){
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const std::string &filename)
{
    checkCudaErrors(cudaFree(0));

    cv::Mat image;
    image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
    if (image.empty()){
        std::cerr << "Couldn't open file:" << filename << std::endl;
        exit(1); 
    }

    cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

    imageGrey.create(image.rows, image.cols, CV_8UC1);

    //报告矩阵是否连续
    if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()){
        std::cerr << "Image aren't continuous!! Exiting." << std::endl;
        exit(1);
    }

    //设置一个指向imageRGBA第一行的指针
    *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
    *greyImage = imageGrey.ptr<unsigned char>(0);

    const size_t numPixels = numRows() * numCols();

    checkCudaErrors(cudaMalloc(d_rgbaImage, numPixels * sizeof(uchar4)));
    checkCudaErrors(cudaMalloc(d_greyImage, numPixels * sizeof(unsigned char)));
    checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

    d_rgbaImage__ = *d_rgbaImage;
    d_greyImage__ = *d_greyImage;  
}

__global__ void rgba_to_greyscale(const uchar4* const rgbaImage, unsigned char* const greyImage, int numRows, int numCols){
    int idx = blockDim.x * blockIdx.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    if (idx < numRows * numCols){
        const unsigned char R = rgbaImage[idx].x;
        const unsigned char G = rgbaImage[idx].y;
        const unsigned char B = rgbaImage[idx].z;
        greyImage[idx] = .299f * R + .587f * G + .114f * B;
    }
}

void postProcess(const std::string& output_file, unsigned char*  data_ptr){
    cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)data_ptr);
    cv::imwrite(output_file.c_str(), output);
}

void cleanup(){
    cudaFree(d_rgbaImage__);
    cudaFree(d_greyImage__);
}

int main(int argc, char* argv[]){
    std::string input_file = argv[1];
    std::string output_file = argv[2];

    uchar4 *h_rgbaImage, *d_rgbaImage;
    unsigned char *h_greyImage, *d_greyImage;

    preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

    int thread = 16;
    int grid = (numRows() * numCols() + thread -1) / (thread * thread);
    const dim3 blockSize(thread, thread);
    const dim3 gridSize(grid);
    rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows(), numCols());

    cudaDeviceSynchronize();

    size_t numPixels = numRows() * numCols();
    checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    postProcess(output_file, h_greyImage);

    cleanup();
    return 0;
}
































