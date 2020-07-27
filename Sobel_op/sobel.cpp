/*
*创建人：我改名字了
*程序描述：sobel算子
*/
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace cv;
const int fac[9] = { 1, 1, 2, 6, 24, 120, 720, 5040, 40320 };

Mat SmoothKernel(int kernelsize) {
	Mat smooth = Mat::zeros(Size(kernelsize, 1), CV_32FC1);
	for (int i = 0; i < kernelsize; i++) {
		smooth.at<float>(0, i) = float(fac[kernelsize - i] / (fac[i] * fac[kernelsize - 1 - i]));
	}
	return smooth;
}

Mat DiffKernel(int kernelsize) {
	Mat Diff = Mat::zeros(Size(kernelsize, 1), CV_32FC1);
	Mat preDiff = SmoothKernel(kernelsize - 1);
	for (int i = 0; i < kernelsize; i++) {
		if (i == 0) {
			Diff.at<float>(0, i) = 1;
		}
		else if (i == kernelsize - 1) {
			Diff.at<float>(0, i) = -1;
		}
		else
		{
			Diff.at<float>(0, i) = preDiff.at<float>(0, i) - preDiff.at<float>(0, i - 1);
		}
	}
	return Diff;
}

void conv2D(InputArray src, InputArray kernel, OutputArray dst, int dep, Point anchor = Point(-1, -1), int borderType = BORDER_DEFAULT) {
	Mat kernelFlip;
	flip(kernel, kernelFlip, -1);
	filter2D(src, dst, dep, kernelFlip, anchor, 0.0, borderType);
}

void sepConv2D_Y_X(InputArray src, OutputArray dst, int dep, InputArray kernelY, InputArray kernelX, Point anchor = Point(-1, -1), int borderType = BORDER_DEFAULT) {
	Mat Y;
	conv2D(src, kernelY, Y, dep, anchor, borderType);
	conv2D(Y, kernelX, dst, dep, anchor, borderType);
}

void sepConv2D_X_Y(InputArray src, OutputArray dst, int dep, InputArray kernelX, InputArray kernelY, Point anchor = Point(-1, -1), int borderType = BORDER_DEFAULT) {
	Mat X;
	conv2D(src, kernelX, X, dep, anchor, borderType);
	conv2D(X, kernelY, dst, dep, anchor, borderType);
}

Mat Sobel(Mat &src, int x_flag, int y_flag, int kSize, int borderType) {
	Mat Smooth = SmoothKernel(kSize);
	Mat Diff = DiffKernel(kSize);
	Mat dst;
	if (x_flag) {
		sepConv2D_Y_X(src, dst, CV_32FC1, Smooth.t(), Diff, Point(-1, -1), borderType);
	}
	else if (x_flag == 0 && y_flag) {
		sepConv2D_X_Y(src, dst, CV_32FC1, Smooth, Diff.t(), Point(-1, -1), borderType);
	}
	return dst;
}
int main()
{
	Mat src = imread(".jpg");
	Mat gray;
	cvtColor(src, gray, CV_BGR2GRAY);
	Mat dst1 = Sobel(gray, 1, 0, 3, BORDER_DEFAULT);
	Mat dst2 = Sobel(gray, 0, 1, 3, BORDER_DEFAULT);
	//转8位灰度图显示
	convertScaleAbs(dst1, dst1);
	convertScaleAbs(dst2, dst2);
	imshow("image", gray);
	imshow("image-X", dst1);
	imshow("image-Y", dst2);
	imwrite(".jpg", dst1);
	imwrite(".jpg", dst2);
	waitKey(0);
	return 0;
}
