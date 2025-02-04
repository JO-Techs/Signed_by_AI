#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Load input image (grayscale)
    Mat img = imread("image.jpg", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    // Apply Laplacian filter
    Mat laplacian, sharpened;
    Laplacian(img, laplacian, CV_16S, 3); // 3x3 kernel, 16-bit signed output

    // Convert to 8-bit and take absolute value
    convertScaleAbs(laplacian, laplacian);

    // Sharpening: Subtract the Laplacian from the original image
    sharpened = img - 0.5 * laplacian;  // Adjust 0.5 for stronger/weaker sharpening

    // Normalize to displayable range
    normalize(sharpened, sharpened, 0, 255, NORM_MINMAX);
    sharpened.convertTo(sharpened, CV_8U);

    // Show images
    imshow("Original Image", img);
    imshow("Laplacian", laplacian);
    imshow("Sharpened Image", sharpened);
    waitKey(0);
    return 0;
}
