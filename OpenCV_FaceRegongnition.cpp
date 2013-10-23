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

CascadeClassifier haar_cascade, haar_cascade2, haar_cascade3, haar_cascade4, haar_cascade5, lbp_cascade;

// Haar
string cascade_alt = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
string cascade_alt2 = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml";
string cascade_default = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_frontalface_default.xml";
string cascade_alt_tree = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml";
string cascade_profile = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_profileface.xml";

// LBP
string lbpcascade_front ="C:/Users/Chochstr/My Programs/opencv/data/lbpcascades/lbpcascade_frontalface.xml";
string lbpcascade_profile ="C:/Users/Chochstr/My Programs/opencv/data/lbpcascades/lbpcascade_profileface.xml";

// Eyes
string cascade_eyes = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_eyes.xml";

string Window_name = "Detect Face(s) in image";
int im_width;
int im_height;
Ptr<FaceRecognizer> model;
string file_csv = "C:/Users/Chochstr/Pictures/classmates_faces/Myfileslist.txt";
vector<Mat> images;
vector<int> labels;
CvCapture* capture;


static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
		line.erase( 0, line.find_first_of("C"));
		path = line.substr(0, line.find(separator));
		classlabel = line.substr(line.find(separator)+1,line.length());
        if(!path.empty() && !classlabel.empty()) {
            images.push_back( imread(path, CV_LOAD_IMAGE_GRAYSCALE) );
            labels.push_back( atoi(classlabel.c_str()) );
        }
    }
}

Mat DetectFace(Mat frame) {
	Mat original = frame.clone();
	Mat gray;
	string text;
	cvtColor(original, gray, CV_BGR2GRAY);
	equalizeHist(gray, gray);
	vector<Rect> faces;
	bool check = true;
	haar_cascade.detectMultiScale(gray, faces, 1.4, 4, CV_HAAR_DO_CANNY_PRUNING,Size(50,50));
	for(int i = 0; i < faces.size(); i++) {
		Rect face_i = faces[i];
		Mat face = gray(face_i);
		Mat face_resized;
		resize(face, face_resized, Size(im_width,im_height), 1.0, 1.0, INTER_CUBIC);
		int predicted_label = -1;
		double predicted_confidence = 0.0;
		model->predict(face_resized, predicted_label, predicted_confidence);
		rectangle(original, face_i, CV_RGB(255,255,255), 1);
		string result;
		if (predicted_label == 1)
			result = "Chase";
		else if (predicted_label == -1)
			result = "Unknown";
		else if (predicted_label == 0)
			result = "Todd";
		else
			result = format("Prediction = %d", predicted_label);
		string box_text = result;
		int pos_x = max(face_i.tl().x - 10, 0);
		int pos_y = max(face_i.tl().y - 10, 0);
		putText(original, box_text, Point(pos_x,pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1.0);
	}
	return original;
}

int Image_Detect() {
	haar_cascade.load(cascade_alt2);
	try {
		read_csv(file_csv, images, labels);
	} catch (Exception& e) {
		cerr << "Error opening file \"" << file_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}
	im_width = images[0].rows;
	im_height = images[0].cols;
	model = createFisherFaceRecognizer(0,1000);
	model->train(images, labels);
	Mat frame;
	while(1){
		capture = cvCaptureFromCAM(-1);
		frame = cvQueryFrame(capture);
		Mat dst;
		dst = DetectFace(frame);
		namedWindow(Window_name, CV_WINDOW_AUTOSIZE);
		imshow(Window_name, dst);
		char c = waitKey(1);
		if (c >= 0) break;
	}
	cvReleaseCapture(&capture);
	destroyWindow(Window_name);
	return 0;
}