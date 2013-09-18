#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void detectAndDisplay(Mat);

String face_cascade_name = "C:/Users/Chochstr/Documents/GitHub/OpenCV_FaceDetection/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "C:/Users/Chochstr/Documents/GitHub/OpenCV_FaceDetection/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

int DetectFace() {
	CvCapture* capture;
	Mat frame;
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    capture = cvCaptureFromCAM(-1);
    if(capture) {
		while(1) {
			frame = cvQueryFrame(capture);
			if ( !frame.empty() ) {
				detectAndDisplay(frame);
			}
			imshow( window_name, frame );
			char c = waitKey(1);
			if(c == 27) break;
		}
    }
	cvReleaseCapture(&capture);
	cvDestroyWindow("Capture - Face detection");
	return 0;
}

void detectAndDisplay( Mat frame ) {
	vector<Rect> faces;
    Mat frame_gray;
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray);
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30,30) );
	for ( int i = 0; i < faces.size(); i++ ) {
		Point center( faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height*0.5 );
		ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 255, 255), 4, 8, 0);
		Mat faceROI = frame_gray( faces[i] );
		vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30,30) );
	}
	imshow( window_name, frame );
}