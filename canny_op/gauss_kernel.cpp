#define PI 3.1415926
//生成小数形式的高斯模板
void generateGaussianTemplate(double window[][11], int ksize, double sigma){
    int center = ksize / 2; //模板的中心位置，也就是坐标原点
    double x2, y2;
    for(int i = 0; i < ksize; i++){
        x2 = pow(i - center, 2);
        for(int j = 0; j < ksize; j++){
            y2 = pow(j - center, 2);
            double g = exp(-(x2 + y2)) / (2 * sigma * sigma);
            g /= 2 * PI * sigma;
            window[i][j] = g;
        }
    }
    //归一化左上角的数为1
    double k = 1 / window[0][0];
    for(int i = 0; i < ksize; i++) {
        for (int j = 0; j < ksize; j++) {
            window[i][j] *= k;
        }
    }
}