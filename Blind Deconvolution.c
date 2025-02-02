#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;
Mat createGaussianKernel(int size, double sigma) {
    Mat kernel(size, size, CV_32F);
    float sum = 0.0;
    int mid = size / 2;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float x = i - mid;
            float y = j - mid;
            kernel.at<float>(i, j) = exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel.at<float>(i, j);
        }
    }
    return kernel / sum;
}
Mat wienerDeconvolution(Mat &blurred, Mat &psf, double noisePower) {
    Mat blurredF, psfF, psfConj, deblurredF, deblurred;
    blurred.convertTo(blurredF, CV_32F);
    psf.convertTo(psfF, CV_32F);
    dft(blurredF, blurredF, DFT_COMPLEX_OUTPUT);
    dft(psfF, psfF, DFT_COMPLEX_OUTPUT);
    Mat planes[] = { Mat::zeros(psfF.size(), CV_32F), Mat::zeros(psfF.size(), CV_32F) };
    split(psfF, planes);
    planes[1] = -planes[1];  
    merge(planes, 2, psfConj);
    Mat psfMagnitudeSquared;
    magnitude(planes[0], planes[1], psfMagnitudeSquared);
    psfMagnitudeSquared = psfMagnitudeSquared.mul(psfMagnitudeSquared) + noisePower;
    Mat wienerFilter;
    divide(psfConj, psfMagnitudeSquared, wienerFilter);
    mulSpectrums(blurredF, wienerFilter, deblurredF, 0);
    dft(deblurredF, deblurred, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
    normalize(deblurred, deblurred, 0, 255, NORM_MINMAX);
    deblurred.convertTo(deblurred, CV_8U);
    return deblurred;
}
int main() {
    Mat blurred = imread("blurred_image.jpg", IMREAD_GRAYSCALE);
    if (blurred.empty()) {
        cout << "Could not open image!" << endl;
        return -1;
    }
    int psfSize = 15;  
    double sigma = 2.0;
    Mat psf = createGaussianKernel(psfSize, sigma);
    double noisePower = 1e-3;
    Mat deblurred = wienerDeconvolution(blurred, psf, noisePower);
    imshow("Blurred Image", blurred);
    imshow("Deblurred Image", deblurred);
    waitKey(0);
    return 0;
}
