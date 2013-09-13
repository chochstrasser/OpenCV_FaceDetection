#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;
String face_cascade_name = "C:/Users/Chochstr/Downloads/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "C:/Users/Chochstr/Downloads/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
RNG rng(12345);
int DetectFace() {
	CvCapture* capture;
	Mat frame;
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    capture = cvCreateCameraCapture(0);
    if(capture) {
		while(1) {
			frame = cvQueryFrame(capture);
			if(frame.empty()) break;
			std::vector<Rect> faces;
			Mat frame_gray;
			cvtColor( frame, frame_gray, CV_BGR2GRAY );
			equalizeHist( frame_gray, frame_gray );
			face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
			for( int i = 0; i < faces.size(); i++ ) {
				Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
				rectangle(frame,Size(faces[i].width*1, faces[i].height*0.5),Size(faces[i].width*1.5, faces[i].height*1), Scalar( 255, 255, 255 ),4,8,0);
				Mat faceROI = frame_gray( faces[i] );
				std::vector<Rect> eyes;
				//eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );
				//for( int j = 0; j < eyes.size(); j++ ) {
				//	Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
				//	int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
				//	circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
				//}
			}
			imshow("Dectection", frame );
			if(waitKey(10) == 27) break;
		}
    }
	cvReleaseCapture(&capture);
	cvDestroyWindow("Dectection");
	return 0;
}