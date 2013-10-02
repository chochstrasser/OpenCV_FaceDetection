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

void detectAndDisplay(Mat);

String face_cascade_name = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "C:/Users/Chochstr/My Programs/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
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
			if ( !frame.empty() ) {	detectAndDisplay(frame); }
			imshow( window_name, frame );
			if((char)waitKey(1) == 27) break;
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
		//Point center( faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height*0.5 );
		Point pt1( faces[i].x - faces[i].width / 6, faces[i].y - faces[i].height / 6 );
		Point pt2( faces[i].x + faces[i].width, faces[i].y + faces[i].height );
		rectangle( frame, pt1, pt2, Scalar(255,255,255),4,8,0);
		//ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 255, 255), 4, 8, 0);
		Mat faceROI = frame_gray( faces[i] );
		vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30,30) );
		for( int j = 0; j < eyes.size(); j++ )
		{
			Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
			int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
			circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
		}
	}
	imshow( window_name, frame );
}

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
        //getline(liness, path, separator);
        //getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back( imread(path, CV_LOAD_IMAGE_GRAYSCALE) );
            labels.push_back( atoi(classlabel.c_str()) );
        }
    }
}

int LBPH_Face() {
	// Get the path to your CSV:
    string fn_haar = face_cascade_name;
    string fn_csv = string("C:/Users/Chochstr/Pictures/att_faces/Myfileslist.txt");
    int deviceId = 0;
    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<int> labels;
    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size AND we need to reshape incoming faces to this size:
    int im_width = images[0].cols;
    int im_height = images[0].rows;
    // Create a FaceRecognizer and train it on the given images:
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();	
    model->train(images, labels);
    // That's it for learning the Face Recognition model. You now
    // need to create the classifier for the task of Face Detection.
    // We are going to use the haar cascade you have specified in the
    // command line arguments:
    //
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);
    // Get a handle to the Video device:
    VideoCapture cap(deviceId);
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        return -1;
    }
    // Holds the current frame from the Video device:
    Mat frame, src, dst, map_x, map_y, gray;
    while(1) {
        cap >> frame;
        // Clone the current frame:
        src = frame.clone();
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
        // Convert the current frame to grayscale:
        cvtColor(dst, gray, CV_BGR2GRAY);
        // Find the faces in the frame:
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces,1.1,2,0|CV_HAAR_SCALE_IMAGE,Size(30,30),Size(480,480));
        // At this point you have the position of the faces in
        // faces. Now we'll get the faces, make a prediction and
        // annotate it in the video. Cool or what?
        for(int i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect face_i = faces[i];
            // Crop the face from the image. So simple with OpenCV C++:
            Mat face = gray(face_i);
            // Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
            // verify this, by reading through the face recognition tutorial coming with OpenCV.
            // Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
            // input data really depends on the algorithm used.
            //
            // I strongly encourage you to play around with the algorithms. See which work best
            // in your scenario, LBPH should always be a contender for robust face recognition.
            //
            // Since I am showing the Fisherfaces algorithm here, I also show how to resize the
            // face you have just found:
            Mat face_resized;
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            // Now perform the prediction, see how easy that is:
            int prediction = model->predict(face_resized);
            // And finally write all we've found out to the original image!
            // First of all draw a green rectangle around the detected face:
            rectangle(dst, face_i, CV_RGB(0, 255,0), 1);
            // Create the text we will annotate the box with:
            //string box_text = format("Prediction = %d", prediction);
			string box_text = "Hello Human!";
            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:
            putText(dst, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }
        // Show the result:
        imshow("face_recognizer", dst);
        // Exit this loop on escape:
        if((char) waitKey(20) == 27) break;
    }
    return 0;
}

int FisherFace() {
    // Get the path to your CSV:
    string fn_haar = face_cascade_name;
    string fn_csv = string("C:/Users/Chochstr/Pictures/att_faces/Myfileslist.txt");
    int deviceId = 0;
    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<int> labels;
    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size AND we need to reshape incoming faces to this size:
    int im_width = images[0].cols;
    int im_height = images[0].rows;
    // Create a FaceRecognizer and train it on the given images:
    //Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    model->train(images, labels);
    // That's it for learning the Face Recognition model. You now
    // need to create the classifier for the task of Face Detection.
    // We are going to use the haar cascade you have specified in the
    // command line arguments:
    //
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);
    // Get a handle to the Video device:
    VideoCapture cap(deviceId);
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        return -1;
    }
    // Holds the current frame from the Video device:
    Mat frame, face, face_resized, src, dst, map_x, map_y, gray;
    while(1) {
        cap >> frame;
        // Clone the current frame:
  //      src = frame.clone();
		//dst.create(src.size(), CV_32FC1);
		//map_x.create(src.size(), CV_32FC1);
		//map_y.create(src.size(), CV_32FC1);
		//for( int j = 0; j < src.rows; j++ ) {
		//	for( int i = 0; i < src.cols; i++ ) {
		//		map_x.at<float>(j,i) = src.cols - i;
		//		map_y.at<float>(j,i) = j;
		//	}
		//}
		//remap(src, dst, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
        // Convert the current frame to grayscale:
        cvtColor(frame, gray, CV_BGR2GRAY);
        // Find the faces in the frame:
        vector< Rect_<int> > faces;
        haar_cascade.detectMultiScale(gray, faces);
        // At this point you have the position of the faces in
        // faces. Now we'll get the faces, make a prediction and
        // annotate it in the video. Cool or what?
        for(int i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect face_i = faces[i];
            // Crop the face from the image. So simple with OpenCV C++:
            face = gray(face_i);
            // Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
            // verify this, by reading through the face recognition tutorial coming with OpenCV.
            // Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
            // input data really depends on the algorithm used.
            //
            // I strongly encourage you to play around with the algorithms. See which work best
            // in your scenario, LBPH should always be a contender for robust face recognition.
            //
            // Since I am showing the Fisherfaces algorithm here, I also show how to resize the
            // face you have just found:
            face_resized;
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            // Now perform the prediction, see how easy that is:
            int prediction = model->predict(face_resized);
            // And finally write all we've found out to the original image!
            // First of all draw a green rectangle around the detected face:
            //rectangle(frame, face_i, CV_RGB(0, 255,0), 1);
            // Create the text we will annotate the box with:
            string box_text = format("Hello = %d", prediction);
            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:
            putText(frame, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }
        // Show the result:
        imshow("face_recognizer", frame);
        // Exit this loop on escape:
        if((char) waitKey(20) == 27) break;
    }
    return 0;
}

int face() {
	CvMemStorage *memStorage = cvCreateMemStorage(0);
	//CvHaarClassifierCascade haar=cvLoadHaarClassifierCascade(ha
	return 0;
}
