#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

using namespace cv;

int Load_Image() {	
	IplImage* img = cvLoadImage("C:/Users/Chochstr/Pictures/Beau J. Hochstrasser/DSC_0108.jpg");
	cvNamedWindow("Example1",CV_WINDOW_NORMAL);
	cvShowImage("Example1", img);
	cvWaitKey(0);
	cvReleaseImage(&img);
	cvDestroyWindow("Example1");
	return 0;
}

int GaussianBlur() {
	namedWindow("Gaussian-in", CV_WINDOW_NORMAL);
	namedWindow("Gaussian-out", CV_WINDOW_NORMAL);
	Mat src = imread("C:/Users/Chochstr/Pictures/Beau J. Hochstrasser/DSC_0108.jpg",1);
	imshow("Gaussian-in", src);
	Mat dst = src.clone();
	GaussianBlur(src,dst,Size(333,333),100,100,BORDER_DEFAULT);
	imshow("Gaussian-out",dst);
	waitKey(0);
	return 0;
}

int HomogeneousBlur() {
	namedWindow("Homogeneous-in", CV_WINDOW_NORMAL);
	namedWindow("Homogeneous-out", CV_WINDOW_NORMAL);
	Mat src = imread("C:/Users/Chochstr/Pictures/Beau J. Hochstrasser/DSC_0108.jpg",1);
	imshow("Homogeneous-in", src);
	Mat dst = src.clone();
	blur(src,dst,Size(333,333), Point(-1,-1));
	imshow("Homogeneous-out",dst);
	waitKey(0);
	return 0;
}

int MedianBlur() {
	namedWindow("Median-in", CV_WINDOW_NORMAL);
	namedWindow("Median-out", CV_WINDOW_NORMAL);
	Mat src = imread("C:/Users/Chochstr/Pictures/Beau J. Hochstrasser/DSC_0108.jpg",1);
	imshow("Median-in", src);
	Mat dst = src.clone();
	medianBlur(src,dst,33);
	imshow("Median-out",dst);
	waitKey(0);
	return 0;
}

int BilateralFilter() {
	namedWindow("Bilater-in", CV_WINDOW_NORMAL);
	namedWindow("Bilater-out", CV_WINDOW_NORMAL);
	Mat src = imread("C:/Users/Chochstr/Pictures/Beau J. Hochstrasser/DSC_0108.jpg",1);
	imshow("Bilater-in", src);
	Mat dst = src.clone();
	bilateralFilter(src,dst,33,66,33/2);
	imshow("Bilater-out",dst);
	waitKey(0);
	return 0;
}

int Eroding() {
	namedWindow("Eroding-in", CV_WINDOW_NORMAL);
	namedWindow("Eroding-out", CV_WINDOW_NORMAL);
	Mat src = imread("C:/Users/Chochstr/Pictures/Beau J. Hochstrasser/DSC_0108.jpg",1);
	imshow("Eroding-in", src);
	Mat dst = src.clone();
	int erosion_type;
	Mat element = getStructuringElement(MORPH_RECT,Size(33,33),Point(-1,-1));
	erode(src,dst,element);
	imshow("Eroding-out", dst);
	waitKey(0);
	return 0;
}

int Dilating() {
	namedWindow("Dilating-in", CV_WINDOW_NORMAL);
	namedWindow("Dilating-out", CV_WINDOW_NORMAL);
	Mat src = imread("C:/Users/Chochstr/Pictures/Beau J. Hochstrasser/DSC_0108.jpg",1);
	imshow("Dilating-in", src);
	Mat dst = src.clone();
	int erosion_type;
	Mat element = getStructuringElement(MORPH_RECT,Size(33,33),Point(-1,-1));
	dilate(src,dst,element);
	imshow("Dilating-out", dst);
	waitKey(0);
	return 0;
}

int	Morph_Open() {
	namedWindow("Open-in", CV_WINDOW_NORMAL);
	namedWindow("Open-out", CV_WINDOW_NORMAL);
	Mat src = imread("C:/Users/Chochstr/Pictures/Beau J. Hochstrasser/DSC_0108.jpg",1);
	imshow("Open-in", src);
	Mat dst = src.clone();
	int erosion_type;
	Mat element = getStructuringElement(MORPH_RECT,Size(33,33),Point(-1,-1));
	morphologyEx(src,dst,2,element);
	imshow("Open-out", dst);
	waitKey(0);
	return 0;
}

int	Morph_Close() {
	namedWindow("Close-in", CV_WINDOW_NORMAL);
	namedWindow("Close-out", CV_WINDOW_NORMAL);
	Mat src = imread("C:/Users/Chochstr/Pictures/Beau J. Hochstrasser/DSC_0108.jpg",1);
	imshow("Close-in", src);
	Mat dst = src.clone();
	int erosion_type;
	Mat element = getStructuringElement(MORPH_RECT,Size(33,33),Point(-1,-1));
	morphologyEx(src,dst,3,element);
	imshow("Close-out", dst);
	waitKey(0);
	return 0;
}

int	Morph_Gradient() {
	namedWindow("Gradient-in", CV_WINDOW_NORMAL);
	namedWindow("Gradient-out", CV_WINDOW_NORMAL);
	Mat src = imread("C:/Users/Chochstr/Pictures/Beau J. Hochstrasser/DSC_0108.jpg",1);
	imshow("Gradient-in", src);
	Mat dst = src.clone();
	int erosion_type;
	Mat element = getStructuringElement(MORPH_RECT,Size(33,33),Point(-1,-1));
	morphologyEx(src,dst,4,element);
	imshow("Gradient-out", dst);
	waitKey(0);
	return 0;
}

int	Morph_TopHat() {
	namedWindow("TopHat-in", CV_WINDOW_NORMAL);
	namedWindow("TopHat-out", CV_WINDOW_NORMAL);
	Mat src = imread("C:/Users/Chochstr/Pictures/Beau J. Hochstrasser/DSC_0108.jpg",1);
	imshow("TopHat-in", src);
	Mat dst = src.clone();
	int erosion_type;
	Mat element = getStructuringElement(MORPH_RECT,Size(33,33),Point(-1,-1));
	morphologyEx(src,dst,5,element);
	imshow("TopHat-out", dst);
	waitKey(0);
	return 0;
}

int	Morph_BlackHat() {
	namedWindow("BlackHat-in", CV_WINDOW_NORMAL);
	namedWindow("BlackHat-out", CV_WINDOW_NORMAL);
	Mat src = imread("C:/Users/Chochstr/Pictures/Beau J. Hochstrasser/DSC_0108.jpg",1);
	imshow("BlackHat-in", src);
	Mat dst = src.clone();
	int erosion_type;
	Mat element = getStructuringElement(MORPH_RECT,Size(33,33),Point(-1,-1));
	morphologyEx(src,dst,6,element);
	imshow("BlackHat-out", dst);
	waitKey(0);
	return 0;
}