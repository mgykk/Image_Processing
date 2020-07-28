/*
*创建人：我改名字了
*程序描述：多尺度融合提升图像细节
*/
#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace cv;

void GaussFilter(const Mat &src, Mat &dst, int ksize, double sigma) {
	CV_Assert(src.channels() == 1 || src.channels() == 3);
	double *matrix = new double[ksize];
	double sum = 0;
	int origin = ksize / 2;
	for (int i = 0; i < ksize; i++) {
		double g = exp(-(1 - origin) * (1 - origin) / (2 * sigma * sigma));
		sum += g;
		matrix[i] = g;
	}
	for (int i = 0; i < ksize; i++) {
		matrix[i] = matrix[i] / sum;
	}
	int border = ksize / 2;
	copyMakeBorder(src, dst, border, border, border, border, BORDER_CONSTANT);
	int channels = dst.channels();
	int rows = dst.rows - border;
	int cols = dst.cols - border;
	for (int i = border; i < rows; i++) {
		for (int j = border; j < cols; j++) {
			double sum[3] = { 0 };
			for (int k = -border; k <= border; k++) {
				if (channels == 1) {
					sum[0] += matrix[border + k] * dst.at<uchar>(i, j + k);
				}
				else if (channels == 3) {
					Vec3b rgb = dst.at<Vec3b>(i, j + k);
					sum[0] += matrix[border + k] * rgb[0];
					sum[1] += matrix[border + k] * rgb[1];
					sum[2] += matrix[border + k] * rgb[2];
				}
			}
			for (int k = 0; k < channels; k++) {
				if (sum[k] < 0) sum[k] = 0;
				else if (sum[k] > 255) sum[k] = 255;
			}
			if (channels == 1) {
				dst.at<Vec3b>(i, j) = static_cast<uchar>(sum[0]);
			}
			else if (channels == 3) {
				Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				dst.at<Vec3b>(i, j) = rgb;
			}
		}
	}
	for (int i = border; i < rows; i++) {
		for (int j = border; j < cols; j++) {
			double sum[3] = { 0 };
			for (int k = -border; k <= border; k++) {
				if (channels == 1) {
					sum[0] += matrix[border + k] * dst.at<uchar>(i + k, j);
				}
				else if (channels == 3) {
					Vec3b rgb = dst.at<Vec3b>(i + k, j);
					sum[0] += matrix[border + k] * rgb[0];
					sum[1] += matrix[border + k] * rgb[1];
					sum[2] += matrix[border + k] * rgb[2];
				}
			}
			for (int k = 0; k < channels; k++) {
				if (sum[k] < 0) sum[k] = 0;
				else if (sum[k] > 255) sum[k] = 255;
			}
			if (channels == 1)
				dst.at<Vec3b>(i, j) = static_cast<uchar>(sum[0]);
			else if (channels == 3) {
				Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				dst.at<Vec3b>(i, j) = rgb;
			}
		}
	}
	delete[] matrix;
}

Mat MultiScaleDetailBoosting(Mat src, int Radius) {
	int rows = src.rows;
	int cols = src.cols;
	Mat B1, B2, B3;
	GaussFilter(src, B1, Radius, 1.0);
	GaussFilter(src, B2, Radius * 2 - 1, 2.0);
	GaussFilter(src, B3, Radius * 4 - 1, 4.0);
	float w1 = 0.5, w2 = 0.5, w3 = 0.25;
	Mat dst(rows, cols, CV_8UC3);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < 3; k++) {
				int D1 = src.at<Vec3b>(i, j)[k] - B1.at<Vec3b>(i, j)[k];
				int D2 = B1.at<Vec3b>(i, j)[k] - B2.at<Vec3b>(i, j)[k];
				int D3 = B2.at<Vec3b>(i, j)[k] - B3.at<Vec3b>(i, j)[k];
				int sign = D1 > 0 ? 1 : -1;
				dst.at<Vec3b>(i, j)[k] = saturate_cast<uchar>((1 - w1 * sign) * D1 - w2 * D2 + w3 * D3 + src.at<Vec3b>(i, j)[k]);
			}
		}
	}
	return dst;
}