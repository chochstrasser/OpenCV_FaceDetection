#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>

using namespace std;
using namespace cv;

void SaveFace() {	
	CascadeClassifier haar_save;
	haar_save.load("C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml");
	string person;
	int count = 1;
	cout << "Please enter your name: ";
	getline(cin,person);
	Mat frame;
	CvCapture* capture;
	while(count < 100){
		capture = cvCaptureFromCAM(-1);
		frame = cvQueryFrame(capture);
		Mat dst = frame.clone();
		Mat gray;
		String text;
		cvtColor(dst, gray, CV_BGR2GRAY);
		equalizeHist(gray,gray);
		vector<Rect> faces;
		haar_save.detectMultiScale(gray, faces, 1.4, 4, CV_HAAR_DO_CANNY_PRUNING,Size(50,50));
		for(int i = 0; i < faces.size(); i++) {
			Rect face_i = faces[i];
			Mat face = gray(face_i);
			Mat face_resized;
			resize(face, face_resized, Size(92, 112), 1.0, 1.0, INTER_CUBIC);
			rectangle(dst, face_i, CV_RGB(255,255,255), 5);
			int pos_x = max(face_i.tl().x - 10, 0);
			int pos_y = max(face_i.tl().y - 10, 0);
			putText(dst, to_string(count), Point(pos_x,pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 2.0);
			string filename = "C:/Users/Chochstr/Pictures/New folder/" + person + format("%d",count) + ".png";
			imwrite(filename, face_resized);
			count++;
		}
		namedWindow("Database Face Builder", CV_WINDOW_NORMAL);
		imshow("Database Face Builder", dst);
		waitKey(10);
	}
	cvReleaseCapture(&capture);
	destroyWindow("Database Face Builder");
}