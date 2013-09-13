#include "opencv2/highgui/highgui.hpp"
using namespace cv;
int Load_Video() {
	cvNamedWindow("Example2", CV_WINDOW_AUTOSIZE);
	CvCapture* capture = cvCreateCameraCapture(0);
	IplImage* frame;
	while(1) {
		frame = cvQueryFrame(capture);
		if(!frame) break;
		cvShowImage("Example2", frame);
		char c = cvWaitKey(33);
		if(c==27) break;
	}
	cvReleaseCapture(&capture);
	cvDestroyWindow("Example2");
	return 0;
}
