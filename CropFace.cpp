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

CascadeClassifier haar;

void CropFace() {
    string fileName = string("C:/Users/Chochstr/Pictures/att_faces/Myfileslist.txt");
	Mat frame = imread(fileName);
	int im_width = frame.rows;
	int im_height = frame.cols;
	Mat face_resized;
    haar.load("C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml");
	cvNamedWindow("Crop - Face detection", CV_WINDOW_NORMAL);
	ifstream file(fileName.c_str(), ifstream::in);
	string line;
    while ( getline(file, line) ) {
		stringstream liness(line);
		line.erase( 0, line.find_first_of("U"));
		string path = line.substr(0, line.find(';'));
		Mat frame = imread(path);
		int im_width = frame.cols;
		int im_height = frame.rows;
		Mat gray;
        cvtColor(frame, gray, CV_BGR2GRAY);
		equalizeHist(gray,gray);
        vector< Rect_<int> > faces;
        haar.detectMultiScale(gray, faces, 2, 2, 0|CV_HAAR_SCALE_IMAGE);
        for(int i = 0; i < faces.size(); i++) {
            Rect face_i = faces[i];
            Mat face = gray(face_i);
            resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
			imwrite(path, face_resized) ;
			imshow("Crop - Face detection", face_resized);
        }
        if((char) waitKey(1) == 27) break;
    }
}