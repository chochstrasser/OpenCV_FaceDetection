#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;

int Load_Video() {
	cvNamedWindow("Example2", CV_WINDOW_AUTOSIZE);
	CvCapture* capture = cvCreateCameraCapture(0);
	Mat frame;
	while(1) {
		frame = cvQueryFrame(capture);
		if( frame.empty() ) break;
		imshow("Example2", frame);
		char c = cvWaitKey(33);
		if(c==27) break;
	}
	cvReleaseCapture(&capture);
	cvDestroyWindow("Example2");
	return 0;
}

int Remap_Video() {
	Load_Video();
	Mat src, dst, map_x, map_y; 
	cvNamedWindow("Example2", CV_WINDOW_AUTOSIZE);
	CvCapture* capture = cvCreateCameraCapture(0);
	while(1) {
		src = cvQueryFrame(capture);
		dst.create(src.size(), CV_32FC1);
		map_x.create(src.size(), CV_32FC1);
		map_y.create(src.size(), CV_32FC1);
		for( int j = 0; j < src.rows; j++ ) {
			for( int i = 0; i < src.cols; i++ ) {
				map_x.at<float>(j,i) = src.cols - i;
				map_y.at<float>(j,i) = j;
			}
		}
		remap(src, dst, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
		imshow("Example2", dst);
		char c = cvWaitKey(33);
		if(c==27) break;
	}
	cvReleaseCapture(&capture);
	cvDestroyWindow("Example2");
	return 0;
}