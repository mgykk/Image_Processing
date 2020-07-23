/*
*�����ˣ��Ҹ�������
*����������ͼ��BBox����任
*/

#include<iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<math.h>
#define PI acos(-1)

int main(int argc, char** argv) {
	if (argc != 11) {
		std::cout << "Args: AffI IMAGE BBOX_TOP_LEFT_X BBOX_TOP_LEFT_Y BBOX_W BBOX_H ROTATE_ANGLE SHEAR_FACTOER OUT_W OUT_H OUT_PATH" << std::endl;
		exit(0);
	}

	cv::Mat image;
	image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);  //��ȡͼƬ

	cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Image", image);    //��ʾԭʼͼ��

	float bbox_x = atoi(argv[2]);
	float bbox_y = atoi(argv[3]);
	float bbox_w = atoi(argv[4]);
	float bbox_h = atoi(argv[5]);
	float bbox_center_x = bbox_x + bbox_w / 2;
	float bbox_center_y = bbox_y + bbox_h / 2;

	float bbox_new_w = atoi(argv[8]);
	float bbox_new_h = atoi(argv[9]);

	//�����ü�����
	cv::Mat Crop_Mat(3, 3, CV_32F);
	Crop_Mat.at<float>(0, 0) = 1;
	Crop_Mat.at<float>(0, 1) = 0;
	Crop_Mat.at<float>(0, 2) = 0;
	Crop_Mat.at<float>(1, 0) = 0;
	Crop_Mat.at<float>(1, 1) = 1;
	Crop_Mat.at<float>(1, 2) = 0;
	Crop_Mat.at<float>(2, 0) = 0;
	Crop_Mat.at<float>(2, 1) = 0;
	Crop_Mat.at<float>(2, 2) = 1;

	Crop_Mat.at<float>(0, 2) = -bbox_x;
	Crop_Mat.at<float>(1, 2) = -bbox_y;

	//�������ž���
	cv::Mat Scale_Mat(3, 3, CV_32F);
	Scale_Mat.at<float>(0, 0) = 1;
	Scale_Mat.at<float>(0, 1) = 0;
	Scale_Mat.at<float>(0, 2) = 0;
	Scale_Mat.at<float>(1, 0) = 0;
	Scale_Mat.at<float>(1, 1) = 1;
	Scale_Mat.at<float>(1, 2) = 0;
	Scale_Mat.at<float>(2, 0) = 0;
	Scale_Mat.at<float>(2, 1) = 0;
	Scale_Mat.at<float>(2, 2) = 1;

	Scale_Mat.at<float>(0, 0) = bbox_new_w / bbox_w;
	Scale_Mat.at<float>(1, 1) = bbox_new_h / bbox_h;

	//����ƽ�ƾ���
	cv::Mat Shift_Mat(3, 3, CV_32F);
	Shift_Mat.at<float>(0, 0) = 1;
	Shift_Mat.at<float>(0, 1) = 0;
	Shift_Mat.at<float>(0, 2) = 0;
	Shift_Mat.at<float>(1, 0) = 0;
	Shift_Mat.at<float>(1, 1) = 1;
	Shift_Mat.at<float>(1, 2) = 0;
	Shift_Mat.at<float>(2, 0) = 0;
	Shift_Mat.at<float>(2, 1) = 0;
	Shift_Mat.at<float>(2, 2) = 1;

	Shift_Mat.at<float>(0, 2) = -(bbox_new_w / 2);
	Shift_Mat.at<float>(1, 2) = -(bbox_new_h / 2);

	//������ת����
	cv::Mat Rotate_Mat(3, 3, CV_32F);
	float rotate = atoi(argv[6]);
	rotate = rotate / 180 * PI;
	const float cos_rotate = std::cos(rotate);
	const float sin_rotate = std::sin(rotate);
	Rotate_Mat.at<float>(0, 0) = 1;
	Rotate_Mat.at<float>(0, 1) = 0;
	Rotate_Mat.at<float>(0, 2) = 0;
	Rotate_Mat.at<float>(1, 0) = 0;
	Rotate_Mat.at<float>(1, 1) = 1;
	Rotate_Mat.at<float>(1, 2) = 0;
	Rotate_Mat.at<float>(2, 0) = 0;
	Rotate_Mat.at<float>(2, 1) = 0;
	Rotate_Mat.at<float>(2, 2) = 1;

	Rotate_Mat.at<float>(0, 0) = cos_rotate;
	Rotate_Mat.at<float>(0, 1) = sin_rotate;
	Rotate_Mat.at<float>(1, 0) = -sin_rotate;
	Rotate_Mat.at<float>(1, 1) = cos_rotate;

	//б�о���
	cv::Mat Shear_Mat(3, 3, CV_32F);
	Shear_Mat.at<float>(0, 0) = 1;
	Shear_Mat.at<float>(0, 1) = 0;
	Shear_Mat.at<float>(0, 2) = 0;
	Shear_Mat.at<float>(1, 0) = 0;
	Shear_Mat.at<float>(1, 1) = 1;
	Shear_Mat.at<float>(1, 2) = 0;
	Shear_Mat.at<float>(2, 0) = 0;
	Shear_Mat.at<float>(2, 1) = 0;
	Shear_Mat.at<float>(2, 2) = 1;

	float shear_factor = atof(argv[7]);
	Shear_Mat.at<float>(0, 1) = shear_factor;
	Shear_Mat.at<float>(1, 0) = shear_factor;

	//��βƽ�ƾ���
	cv::Mat Shift_Mat1(2, 3, CV_32F);
	Shift_Mat1.at<float>(0, 0) = 1;
	Shift_Mat1.at<float>(0, 1) = 0;
	Shift_Mat1.at<float>(0, 2) = 0;
	Shift_Mat1.at<float>(1, 0) = 0;
	Shift_Mat1.at<float>(1, 1) = 1;
	Shift_Mat1.at<float>(1, 2) = 0;

	Shift_Mat1.at<float>(0, 2) = bbox_new_w / 2;
	Shift_Mat1.at<float>(1, 2) = bbox_new_h / 2;

	cv::Mat out;
	//�任�������Ƚ��еĲ����������ұ�
	cv::Mat trans_mat = Shift_Mat * Shear_Mat * Rotate_Mat * Shift_Mat * Scale_Mat * Crop_Mat;
	cv::Size new_size(bbox_new_h, bbox_new_h);
	cv::warpAffine(image, out, trans_mat, new_size, CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	cv::imwrite(argv[10], out);
	cv::namedWindow("affi image", cv::WINDOW_AUTOSIZE);
	cv::imshow("affi image", out);
	cv::waitKey();

	return 0;
}