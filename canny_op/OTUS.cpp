int OTSU(Mat src){
	int row = src.rows;
	int col = src.cols;
	int PixelCount[256] = {0};
	float PixelPro[256] = {0};
	float total_Pixel = row * col;
	float threshold = 0;

	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++){
			PixelCount[src.at<int>(i, j)]++;
		}
	}

	for (int i = 0; i < 256; i++){
		PixelPro[i] = (float)(PixelCount[i]) / (float)(total_Pixel);
	}

	float w0, w1, u0tmp, u1tmp, u0, u1, u, deltaTmp, deltaMax = 0;
	for (int i = 0; i < 256; i++){
		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;
		for (int j = 0; j < 256; j++){
            if (j <= i){
            	w0 += PixelPro[j];
            	u0tmp += j * PixelPro[j];
            }else{
            	w1 += PixelPro[j];
            	u1tmp += j * PixelPro[j];
            }
		}
		//计算每一类的平均灰度
		u0 = u0tmp / w0;
		u1 = u1tmp / w1;
		u = u0 + u1;
		//计算类间方差
		//g = w0 * w1 * (u0 - u1) * (u0 - u1)
		deltaTmp = w0 * (u0 - u) * (u0 - u) + w1 * (u1 - u) * (u1 - u);
		//更新最大方差及其阈值
		if (deltaTmp > deltaMax){
			deltaMax = deltaTmp;
			threshold = i;
		}
	}
    return threshold;
}