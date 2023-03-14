#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <dirent.h>

using namespace std;

void GetFileNames(string path, vector<string> &filenames) {
    DIR *pDir;
    struct dirent *ptr;
    if (!(pDir = opendir(path.c_str())))
        return;
    while ((ptr = readdir(pDir)) != 0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
            // filenames.push_back(path + "/" + ptr->d_name);
            filenames.push_back(ptr->d_name);
    }
    closedir(pDir);
}

__global__ void
cudaBuildMap(float *pCamK, float *pDistort, float *pInvNewCamK, float *pMapx, float *pMapy, int outImgW, int outImgH) {
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    if (tidx < outImgW && tidy < outImgH) {
        //图像坐标系->摄像机坐标系
        float _x = tidx * pInvNewCamK[0] + tidy * pInvNewCamK[1] + pInvNewCamK[2];
        float _y = tidx * pInvNewCamK[3] + tidy * pInvNewCamK[4] + pInvNewCamK[5];
        float _w = tidx * pInvNewCamK[6] + tidy * pInvNewCamK[7] + pInvNewCamK[8];
        //归一化
        float x = _x / _w;
        float y = _y / _w;
        //畸变校正
        float r = sqrt(x * x + y * y);
        float theta = atan(r);

        float theta2 = theta * theta;
        float theta4 = theta2 * theta2;
        float theta6 = theta4 * theta2;
        float theta8 = theta4 * theta4;
        float theta_d =
                theta * (1 + pDistort[0] * theta2 + pDistort[1] * theta4 + pDistort[2] * theta6 + pDistort[3] * theta8);
        //重投影到图像坐标系
        float scale = (r == 0) ? 1.0 : theta_d / r;
        float u = pCamK[0] * x * scale + pCamK[2];
        float v = pCamK[4] * y * scale + pCamK[5];
        //保存Map
        int mapIdx = tidy * outImgW + tidx;
        pMapx[mapIdx] = (float) u;
        pMapy[mapIdx] = (float) v;
    }
}

__global__ void
cudaRemap(uchar *pSrcImg, uchar *pDstImg, float *pMapx, float *pMapy, int inWidth, int inHeight, int outWidth,
          int outHeight, int channels) {
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    if (tidx < outWidth && tidy < outHeight) {
        int mapIdx = tidy * outWidth + tidx;
        float u = pMapx[mapIdx];
        float v = pMapy[mapIdx];
        //双线性插值
        int u1 = floor(u);
        int v1 = floor(v);
        int u2 = u1 + 1;
        int v2 = v1 + 1;
        if (u1 >= 0 && v1 >= 0 && u2 < inWidth && v2 < inHeight) {
            float dx = u - u1;
            float dy = v - v1;
            float weight1 = (1 - dx) * (1 - dy);
            float weight2 = dx * (1 - dy);
            float weight3 = (1 - dx) * dy;
            float weight4 = dx * dy;

            int resultIdx = mapIdx * 3;
            for (int chan = 0; chan < channels; chan++) {
                pDstImg[resultIdx + chan] = uchar(weight1 * pSrcImg[(v1 * inWidth + u1) * 3 + chan]
                                                  + weight2 * pSrcImg[(v1 * inWidth + u2) * 3 + chan]
                                                  + weight3 * pSrcImg[(v2 * inWidth + u1) * 3 + chan]
                                                  + weight4 * pSrcImg[(v2 * inWidth + u2) * 3 + chan]);
            }
        }
    }
}

int main() {
    const string old_path = "/media/ros/A666B94D66B91F4D/ros/test_port/camera/car2_train/test/dis/";  // 老的路径
    string new_path = "/media/ros/A666B94D66B91F4D/ros/test_port/camera/car2_train/test/undis/";  // 新的路径
    vector<string> file_name;
    GetFileNames(old_path, file_name);
    // 最好sort一下
    std::sort(file_name.begin(),file_name.end());
    cv::Mat srcImg = cv::imread("/media/ros/A666B94D66B91F4D/ros/test_port/camera/car1_line/frame0763.jpg");
    cv::Mat camK = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat newCamK = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat invCamK = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat D = cv::Mat::zeros(4, 1, CV_32F);
    cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
    int imgHeight = srcImg.rows;
    int imgWidth = srcImg.cols;
    int channels = srcImg.channels();
    int outImgHeight = imgHeight;
    int outImgWidth = imgWidth;
    cv::Mat undistortImg = cv::Mat(outImgHeight, outImgWidth, CV_8UC3);
    cv::Mat mapx = cv::Mat(outImgHeight, outImgWidth, CV_32F);
    cv::Mat mapy = cv::Mat(outImgHeight, outImgWidth, CV_32F);

    //内参矩阵
    // 1
    // camK.at<float>(0, 0) = 1003.9989013289942;
    // camK.at<float>(1, 1) = 1004.1132782586517;
    // camK.at<float>(0, 2) = 926.3763250309561;
    // camK.at<float>(1, 2) = 546.1004237610695;

    //2
    camK.at<float>(0, 0) = 1004.374739582285;
    camK.at<float>(1, 1) = 1003.9191500026428;
    camK.at<float>(0, 2) = 941.5232440525655;
    camK.at<float>(1, 2) = 592.1157654586079;

    //畸变系数矩阵
    // float disCoeff[4] = {-0.0526858350541784, -0.01873269061565343, 0.0060846931831152, -0.0016727061237763216};
    float disCoeff[4] = {-0.054613280720461926, -0.014843292079092893, 0.004670322686634465, -0.0014125235895859126};
    D = cv::Mat(4, 1, CV_32F, disCoeff);

    //内参矩阵求逆
    // newCamK = camK.clone();
    // 1
    // newCamK.at<float>(0, 0) = 627.1724853515625;
    // newCamK.at<float>(1, 1) = 1003.18359375;
    // newCamK.at<float>(0, 2) = 925.8938436648168;
    // newCamK.at<float>(1, 2) = 545.5947960948106;

    // 2
    newCamK.at<float>(0, 0) = 627.4072265625;
    newCamK.at<float>(1, 1) = 1002.989562988281;
    newCamK.at<float>(0, 2) = 941.0328309849137;
    newCamK.at<float>(1, 2) = 591.567489023193;

    invCamK = newCamK.inv(cv::DECOMP_SVD);  // 注意这个会不会出现问题，导致结果不对齐
    // cv::Matx33d iR = (PP * RR).inv(cv::DECOMP_SVD); OPENCV源码
    std:: cout << camK << std::endl;
    std:: cout << invCamK << std::endl;
    std:: cout << D << std::endl;


    //分配GPU内存并上传数据至GPU
    auto malloc_time_start = std::chrono::system_clock::now();
    cudaError err;
    float *pCamKData = NULL;
    float *pInvNewCamKData = NULL;
    float *pDistortData = NULL;
    uchar *pSrcImgData = NULL;
    uchar *pDstImgData = NULL;
    float *pMapxData = NULL;
    float *pMapyData = NULL;
    {
        err = cudaMalloc(&pCamKData, 9 * sizeof(float));
        err = cudaMalloc(&pInvNewCamKData, 9 * sizeof(float));
        err = cudaMalloc(&pDistortData, 4 * sizeof(float));
        err = cudaMalloc(&pSrcImgData, imgHeight * imgWidth * sizeof(uchar) * channels);
        err = cudaMalloc(&pMapxData, outImgHeight * outImgWidth * sizeof(float));
        err = cudaMalloc(&pMapyData, outImgHeight * outImgWidth * sizeof(float));
        err = cudaMalloc(&pDstImgData, outImgHeight * outImgWidth * sizeof(uchar) * channels);
    }
    auto malloc_time_end = std::chrono::system_clock::now();
    std::cout << "malloc time is " // 只统计模型预测时间, 不包含图像预处理后处理
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                      malloc_time_end - malloc_time_start).count()
              << " ms" << std::endl;


    // 数据拷贝加计算
    for (int i = 0; i < file_name.size(); i++){
        cv::Mat RawImage = cv::imread(old_path + file_name[i]);

        auto start_undistort_image = std::chrono::system_clock::now();
        {
            err = cudaMemcpy(pCamKData, camK.data, 9 * sizeof(float), cudaMemcpyHostToDevice);
            err = cudaMemcpy(pInvNewCamKData, invCamK.data, 9 * sizeof(float), cudaMemcpyHostToDevice);
            err = cudaMemcpy(pDistortData, D.data, 4 * sizeof(float), cudaMemcpyHostToDevice);
            err = cudaMemcpy(pMapxData, mapx.data, outImgHeight * outImgWidth * sizeof(float), cudaMemcpyHostToDevice);
            err = cudaMemcpy(pMapyData, mapy.data, outImgHeight * outImgWidth * sizeof(float), cudaMemcpyHostToDevice);
            err = cudaMemcpy(pSrcImgData, RawImage.data, imgHeight * imgWidth * sizeof(uchar) * channels,
                             cudaMemcpyHostToDevice);
        }

        dim3 block(16, 16);
        dim3 grid((imgWidth + block.x - 1) / block.x, (imgHeight + block.y - 1) / block.y);
        //创建Map
        cudaBuildMap << < grid, block >> > (pCamKData, pDistortData, pInvNewCamKData, pMapxData, pMapyData, outImgWidth, outImgHeight);
        cudaThreadSynchronize();
        //Remap
        cudaRemap << < grid, block >> > (pSrcImgData, pDstImgData, pMapxData, pMapyData, imgWidth, imgHeight, outImgWidth, outImgHeight, channels);
        cudaThreadSynchronize();
        err = cudaGetLastError();
        //拷贝数据
        err = cudaMemcpy(undistortImg.data, pDstImgData, outImgHeight * outImgWidth * sizeof(uchar) * channels,
                         cudaMemcpyDeviceToHost);


        auto end_undistort_image = std::chrono::system_clock::now();
        std::cout << "undistortImage time is " // 只统计模型预测时间, 不包含图像预处理后处理
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                          end_undistort_image - start_undistort_image).count()
                  << " ms" << std::endl;
        cv::imwrite(new_path + file_name[i], undistortImg);
    }


    // cv::imwrite("/media/ros/A666B94D66B91F4D/ros/test_port/camera/car1_line/undistortImg.jpg", undistortImg);

    return 0;
}
