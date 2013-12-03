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
#include <stdlib.h>
#include <Windows.h>
#include <process.h>
#include <thread>

using namespace std;
using namespace cv;

// Cascade classifier(s)
CascadeClassifier haar_cascade, haar_cascade2, haar_cascade3, haar_cascade4, haar_cascade5, lbp_cascade;

// Haar cascade classifer
string cascade_alt = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
string cascade_alt2 = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml";
string cascade_default = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_frontalface_default.xml";
string cascade_alt_tree = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml";
string cascade_profile = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_profileface.xml";

// LBP cascade classifer
string lbpcascade_front ="C:/Users/Chochstr/My Programs/opencv/data/lbpcascades/lbpcascade_frontalface.xml";
string lbpcascade_profile ="C:/Users/Chochstr/My Programs/opencv/data/lbpcascades/lbpcascade_profileface.xml";

// Eyes
string cascade_eyes = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_eyes.xml";

// Window name
string Window_name = "Detect Face(s) in image";

// Image width & height
int im_width;
int im_height;

// Point Face Recongnizer
Ptr<FaceRecognizer> Fisher_model;
Ptr<FaceRecognizer> Eigen_model;
Ptr<FaceRecognizer> LBPH_model;

// File of list path
string file_csv = "C:/Users/Chochstr/Pictures/classmates_faces/Myfileslist.txt";

// Store person name here
struct person {
	int num;
	string name;
} p;

// Vectors
vector<Mat> images;
vector<int> labels;
vector<Rect> faces;
vector<person> people;

Rect face_i;
CvCapture* capture;
int prev_predict, predict_series;
Mat face_resized, original, gray, face;

// Prediction Labels
int Fisher_Predict, Eigen_Predict, LBPH_Predict;

// Prediction Confidence
double Fisher_Confidence = 0.0, Eigen_Confidence = 0.0, LBPH_Confidence = 0.0;

// Read file of images and store for Recognition
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
			path.resize(path.find_last_of("/\\"));
			p.name = path.substr(path.find_last_of("/\\")+1);
			p.num = atoi(classlabel.c_str());
			people.emplace_back(p);
        }
    }
}

// Predict fisher face
unsigned int _stdcall Fisher_method(void*) {
	Fisher_model->predict(face_resized, Fisher_Predict, Fisher_Confidence);
	return 0;
}

// Preedict Eigen face
unsigned int _stdcall Eigen_method(void*) {
	Eigen_model->predict(face_resized, Eigen_Predict, Eigen_Confidence);
	return 0;
}

// Predict LBPH face
unsigned int _stdcall LBPH_method(void*) {
	LBPH_model->predict(face_resized, LBPH_Predict, LBPH_Confidence);
	return 0;
}

// Display Results in Console Window
void Display_Results() {
	int predict;
	double confidence;
	string result;
	
	// Loop through each
	for (int i = 0; i < 3; i++) {
		
		switch (i) {
			case 0: 
				predict = Fisher_Predict;
				confidence = Fisher_Confidence;
		
				// Find Person by prediction
				for (vector<person>::size_type i = 0; i != people.size(); i++) {	
					if (predict == people[i].num)
						result = people[i].name;
				}

				// Create text to display
				cout << "Fisher Prediction: " << predict << endl;
				cout.precision(3);
				cout << "Fisher Confidence: " << confidence << endl;
				cout << "Name: " << result << endl;
				cout << endl;

				break;
			case 1: 
				predict = Eigen_Predict;
				confidence = Eigen_Confidence;
				
				// Find Person by prediction
				for (vector<person>::size_type i = 0; i != people.size(); i++) {	
					if (predict == people[i].num)
						result = people[i].name;
				}

				// Create text to display
				cout << "Eigen Prediction: " << predict << endl;
				cout.precision(3);
				cout << "Eigen Confidence: " << confidence << endl;
				cout << "Name: " << result << endl;
				cout << endl;

				break;
			case 2: 
				predict = LBPH_Predict;
				confidence = LBPH_Confidence;
			
				// Find Person by prediction
				for (vector<person>::size_type i = 0; i != people.size(); i++) {	
					if (predict == people[i].num)
						result = people[i].name;
				}

				// Create text to display
				cout << "LBPH Prediction: " << predict << endl;
				cout.precision(3);
				cout << "LBPH Confidence: " << confidence << endl;
				cout << "Name: " << result << endl;
				cout << endl;

				break;
		}
	}
}

// Result for Standard Deviation
vector<double> dbl;

// Standard Deviation
double SDV(vector<double> data) {
	double sum = 0.0;
	double size = data.size();
    for(double a : data){
		sum += a;
    }
	double mean = sum/size;
    double temp = 0;
    for (double a :data){
		temp += (mean-a)*(mean-a);
    }
	double var = temp/size;
    return sqrt(var);
}


// Detect Faces in image
Mat DetectFace(Mat frame) {

	// Make a clone of the image
	original = frame.clone();

	// Convert to gray color space
	cvtColor(original, gray, CV_BGR2GRAY);

	// Normalize the gray scaled image
	equalizeHist(gray, gray);

	// Detect face in gray image
	haar_cascade.detectMultiScale(gray, faces, 1.3, 3, CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING, Size(30,30));

	// Loop over faces found in image
	for(int i = 0; i < faces.size(); i++) {

		// Get the face(s)
		face_i = faces[i];

		// Gray out face
		face = gray(face_i);

		// Get the face only
		resize(face, face_resized, Size(im_width,im_height), 1.0, 1.0, INTER_CUBIC);
	
		// Handler
		HANDLE t[3];

		// Fischer Recognizer
		t[0] = (HANDLE) _beginthreadex(0, 0, &Fisher_method, (void*)0, 0, 0);
		
		// Eigen Recognizer
		t[1] = (HANDLE) _beginthreadex(0, 0, &Eigen_method, (void*)0, 0, 0);
	
		// LBPH Recognizer
		t[2] = (HANDLE) _beginthreadex(0, 0, &LBPH_method, (void*)0, 0, 0);

		// Terminate all Threads
		WaitForMultipleObjects(3, t, TRUE, INFINITE);

		// Close Handler(s)
		CloseHandle(t[0]);
		CloseHandle(t[1]);
		CloseHandle(t[2]);

		// Add to list of Confidence values
		dbl.push_back(LBPH_Confidence);

		// Get the Standard Deviation
		double STANDARD = SDV(dbl);

		// Draw a rectangle
		rectangle(original, face_i, CV_RGB(255,255,255), 1);

		// Display Results
		Display_Results();
	}
	return original;
}


// Creat FisheFace model for face recognizer and train it with
// images and labels read from the given CSV file.
unsigned int _stdcall Model_Fisher(void*) {
	Fisher_model = createFisherFaceRecognizer();
	Fisher_model->set("threshold",DBL_MAX);
	Fisher_model->train(images, labels);
	return 0;
}

// Creat EigenFaces model for face recognizer and train it with
// images and labels read from the given CSV file.
unsigned int _stdcall Model_Eigen(void*) {
	Eigen_model = createEigenFaceRecognizer();
	Eigen_model->set("threshold",DBL_MAX);
	Eigen_model->train(images, labels);
	return 0;
}

// Creat LBPHFaces model for face recognizer and train it with
// images and labels read from the given CSV file.
unsigned int _stdcall Model_LBPH(void*) {
	LBPH_model = createLBPHFaceRecognizer();
	LBPH_model->set("threshold",DBL_MAX);
	LBPH_model->train(images, labels);
	return 0;
}

int Image_Detect() {
	
	// Read in Images from file
	try {
		read_csv(file_csv, images, labels);
	} catch (Exception& e) {
		cerr << "Error opening file \"" << file_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}

	// Get image width & height from first image
	im_width = images[0].rows;
	im_height = images[0].cols;

	// Load cascade classifier for detection
	haar_cascade.load(cascade_alt2);
	
	// Handle array
	HANDLE t[3];

	// Model Fisher Face Recognizer
	t[0] = (HANDLE) _beginthreadex(0, 0, &Model_Fisher, (void*)0, 0, 0);
	
	// Model Eigen Face Recognizer
	t[1] = (HANDLE) _beginthreadex(0, 0, &Model_Eigen, (void*)0, 0, 0);

	// Model LBPH Face Recognizer
	t[2] = (HANDLE) _beginthreadex(0, 0, &Model_LBPH, (void*)0, 0, 0);

	// Terminate all threads
	WaitForMultipleObjects(3, t, TRUE, INFINITE);

	// Close Handle(s)
	CloseHandle(t[0]);
	CloseHandle(t[1]);
	CloseHandle(t[2]);

	// Define
	Mat frame, dst;
	
	// Capture image from file
	const char* filename = "C:/Users/Chochstr/Videos/Eye-Fi/12-2-2013/HHD00001.MOV";
	
	// Capture frame from video file
	capture = cvCaptureFromFile(filename);
	
	// INFINITE LOOP
	while(1) {

		// Get frame image and retrieve
		frame = cvQueryFrame(capture);

		// Check if end of video
		if (frame.empty()) break;

		// Detect Face(s)
		dst = DetectFace(frame);

		// Set Window
		namedWindow(Window_name, CV_WINDOW_AUTOSIZE);

		// Show image to user
		imshow(Window_name, dst);

		// Exit on key stroke
		char c = waitKey(20);
		if (c >= 0) break;
	}

	// Release all captures
	cvReleaseCapture(&capture);

	// Destroy Window
	destroyWindow(Window_name);

	// Return to exit program
	return 0;
}