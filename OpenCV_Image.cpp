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
	Mat element = getStructuringElement(MORPH_RECT,Size(33,33),Point(-1,-1));
	morphologyEx(src,dst,6,element);
	imshow("BlackHat-out", dst);
	waitKey(0);
	return 0;
}


int Remap() {
	Mat src, dst, map_x, map_y;
	namedWindow("Inverse-in", CV_WINDOW_NORMAL);
	namedWindow("Inverse-out", CV_WINDOW_NORMAL);
	src = imread("C:/Users/Chochstr/Pictures/Beau J. Hochstrasser/DSC_0108.jpg",1);
	resizeWindow("Inverse-in",480,480);
	imshow("Inverse-in",src);
	dst.create(src.size(), src.type());
	map_x.create(src.size(), CV_32FC1);
	map_y.create(src.size(), CV_32FC1);
	for( int j = 0; j < src.rows; j++ ) {
		for( int i = 0; i < src.cols; i++ ) {
			map_x.at<float>(j,i) = src.cols - i;
			map_y.at<float>(j,i) = j;
		}
	}
	remap(src, dst, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
	resizeWindow("Inverse-out",480,480);
	imshow("Inverse-out", dst);
	waitKey(0);
	return 0;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    ifstream file(filename.c_str(), ifstream::in);
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

Mat DetectFace(Mat);

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
string file_csv = "C:/Users/Chochstr/Pictures/att_faces/Myfileslist.txt";
vector<Mat> images;
vector<int> labels;
CvCapture* capture;

int Image_Detect() {	
	//haar_cascade.load(cascade_alt);
	haar_cascade2.load(cascade_alt2);
	//haar_cascade3.load(cascade_default);
	//haar_cascade4.load(cascade_alt_tree);
	//lbp_cascade.load(lbpcascade_front);
	try {
		read_csv(file_csv, images, labels);
	} catch (Exception& e) {
		cerr << "Error opening file \"" << file_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}
	im_width = images[0].rows;
	im_height = images[0].cols;
	model = createFisherFaceRecognizer();
	model->train(images, labels);
	Mat frame;
	string filename = "C:/Users/Chochstr/Pictures/faces.jpg";
	while(1){
		capture = cvCaptureFromCAM(-1);
		//frame = imread(filename);
		frame = cvQueryFrame(capture);
		Mat dst = DetectFace(frame);
		namedWindow(Window_name, CV_WINDOW_NORMAL);
		imshow(Window_name, dst);
		char c = waitKey(1);
		if (c==27) break;
	}
	return 0;
}

Mat DetectFace(Mat frame) {
	Mat original = frame.clone();
	Mat gray;
	string text;
	cvtColor(original, gray, CV_BGR2GRAY);
	equalizeHist(gray,gray);
	vector<Rect> faces, faces2, faces3, faces4, faces5;
	//haar_cascade.detectMultiScale(gray, faces);
	haar_cascade2.detectMultiScale(gray, faces2, 2.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30,30));
	//haar_cascade3.detectMultiScale(gray, faces3);
	//haar_cascade4.detectMultiScale(gray, faces4);
	//haar_cascade5.detectMultiScale(gray,faces5);
	//lbp_cascade.detectMultiScale(gray, faces2, 3, 1, 0|CASCADE_DO_CANNY_PRUNING,Size(30,30));
	/*for(int i = 0; i < faces.size(); i++) {
		Rect face_i = faces[i];
		Mat face = gray(face_i);
		Mat face_resized;
		resize(face,face_resized,Size(im_width,im_height),1.0,1.0,INTER_CUBIC);
		rectangle(original,face_i,CV_RGB(0,255,0),10);
		int pos_x = max(face_i.tl().x - 10, 0);
		int pos_y = max(face_i.tl().y - 10, 0);
		putText(original,"",Point(pos_x,pos_y), FONT_HERSHEY_PLAIN,1.0,CV_RGB(0,255,0),2.0);
	}*/
	for(int i = 0; i < faces2.size(); i++) {
		Rect face_i = faces2[i];
		Mat face = gray(face_i);
		Mat face_resized;
		resize(face,face_resized,Size(im_width,im_height),1.0,1.0,INTER_CUBIC);
		int prediction = model->predict(face_resized);
		rectangle(original,face_i,CV_RGB(0,255,255),10);
		string box_text = format("Prediction = %d", prediction);
		int pos_x = max(face_i.tl().x - 10, 0);
		int pos_y = max(face_i.tl().y - 10, 0);
		putText(original,box_text,Point(pos_x,pos_y), FONT_HERSHEY_PLAIN,1.0,CV_RGB(0,255,0),2.0);
	}
	/*for(int i = 0; i < faces3.size(); i++) {
		Rect face_i = faces3[i];
		Mat face = gray(face_i);
		Mat face_resized;
		resize(face,face_resized,Size(im_width,im_height),1.0,1.0,INTER_CUBIC);
		//rectangle(original,face_i,CV_RGB(200,100,100),10);
		int pos_x = max(face_i.tl().x - 10, 0);
		int pos_y = max(face_i.tl().y - 10, 0);
		putText(original,"",Point(pos_x,pos_y), FONT_HERSHEY_PLAIN,1.0,CV_RGB(0,255,0),2.0);
	}
	for(int i = 0; i < faces4.size(); i++) {
		Rect face_i = faces4[i];
		Mat face = gray(face_i);
		Mat face_resized;
		resize(face,face_resized,Size(im_width,im_height),1.0,1.0,INTER_CUBIC);
		//rectangle(original,face_i,CV_RGB(50,100,100),10);
		int pos_x = max(face_i.tl().x - 10, 0);
		int pos_y = max(face_i.tl().y - 10, 0);
		putText(original,"",Point(pos_x,pos_y), FONT_HERSHEY_PLAIN,1.0,CV_RGB(0,255,0),2.0);
	}
	for(int i = 0; i < faces5.size(); i++) {
		Rect face_i = faces5[i];
		Mat face = gray(face_i);
		Mat face_resized;
		resize(face,face_resized,Size(im_width,im_height),1.0,1.0,INTER_CUBIC);
		//rectangle(original,face_i,CV_RGB(50,100,100),10);
		int pos_x = max(face_i.tl().x - 10, 0);
		int pos_y = max(face_i.tl().y - 10, 0);
		putText(original,"",Point(pos_x,pos_y), FONT_HERSHEY_PLAIN,1.0,CV_RGB(0,255,0),2.0);
	}*/
	return original;
}