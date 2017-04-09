#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "camera.hpp"
#include "feature.hpp"
#include "map.hpp"
#include "quaternion.hpp"
#include "util.hpp"

using namespace cv;
using namespace std;

bool getOptions(int argc, char *argv[], string& cameraFile, int& patternRows, int& patternCols,
                double& squareSize);
void printUsage(const char *execName);

void drawFeatures(const Mat& frame, const vector<Feature>& features);
Mat findCandidates(const Mat& frame, const vector<Feature>& features);

int main(int argc, char *argv[]) {

    string cameraFile;
    int patternRows, patternCols;
    double squareSize;

    if (!getOptions(argc, argv, cameraFile, patternRows, patternCols, squareSize)) {

        printUsage(argv[0]);
        return -1;
    }

    FileStorage fs(cameraFile, FileStorage::READ);

    if (!fs.isOpened()) {

        cout << "Could not open \"" << cameraFile << "\": camera settings not loaded" << endl;
        return -1;
    }

    Size frameSize;
    Mat K, distCoeffs;

    fs["Image_Size"] >> frameSize;
    fs["Intrinsic_Matrix"] >> K;
    fs["Distortion_Coefficients"] >> distCoeffs;

    fs.release();

    // Features patch size
    int patchSize = 11;

    // Build new map
    Map map(K, distCoeffs, frameSize, patchSize);

    // Chessboard size
    Size patternSize(patternCols - 1, patternRows - 1);

    // Camera initial state variances
    vector<double> var({0.02, 0.02, 0.02, 0.18});

    // Initialize video feed device
    VideoCapture cap(0);

    if (!cap.isOpened())
        return -1;

    // Create a window
    string window = "Test";
    namedWindow(window, CV_WINDOW_AUTOSIZE);

    Mat frame, gray;

    bool init;

    for (;;) {

        cap.read(frame);

        imshow(window, frame);

        cvtColor(frame, gray, CV_BGR2GRAY);

        if ((init = map.initMap(gray, patternSize, squareSize, var)))
            break;

        if (waitKey(1) == 27)
            break;
    }

    if (!init)
        return 0;

    for (;;) {

        cap.read(frame);

        cvtColor(frame, gray, CV_BGR2GRAY);

        drawFeatures(frame, map.features);

        Mat corners = findCandidates(gray, map.features);

        // Draw a circle around corners
        for (int j = 0; j < corners.rows; j++) {
            for (int i = 0; i < corners.cols; i++) {

                Point p = corners.at<Point2f>(j, i);
                circle(frame, p, 5, Scalar(0, 0, 255), 1, CV_AA, 0);
            }
        }

        imshow(window, frame);

        if (waitKey(1) == 27)
            break;
    }
}

bool getOptions(int argc, char *argv[], string& cameraFile, int& patternRows, int& patternCols,
                double& squareSize) {

    // Look for the help option
    for (int i = 1; i < argc; i++)
        if (string(argv[i]) == "-h")
            return false;

    // Check the number of input options
    if (argc != 5)
        return false;

    cameraFile = argv[1];

    patternRows = (int) strtol(argv[2], NULL, 10);
    if (patternRows == 0 || errno == ERANGE) return false;

    patternCols = (int) strtol(argv[3], NULL, 10);
    if (patternCols == 0 || errno == ERANGE) return false;

    squareSize = strtod(argv[4], NULL);
    if (squareSize == 0 || errno == ERANGE) return false;

    return true;
}

void printUsage(const char *execName) {

    cerr << "Usage: " << execName << " cameraFile patternRows patternCols squareSize (-h for help)" << endl;
}

void drawFeatures(const Mat& frame, const vector<Feature>& features) {

    for (unsigned int i = 0; i < features.size(); i++) {

        Rect roi = features[i].roi;

        int x = roi.x + roi.width / 2;
        int y = roi.y + roi.height / 2;

        circle(frame, Point2i(x, y), 5, Scalar(0, 0, 255), 1, CV_AA, 0);
    }
}

Mat findCandidates(const Mat& frame, const vector<Feature>& features) {

    Mat corners;

    int maxCorners = 15 - features.size();
    double qualityLevel = 0.2;
    double minDistance = 60;

    Mat mask(frame.size(), CV_8UC1, Scalar(0));

    // Set a margin around the mask to enclose the detected corners inside
    // a rectangle and avoid premature corner loss due to camera movement
    int pad = 30;
    mask(Rect(pad, pad, mask.cols - 2 * pad, mask.rows - 2 * pad)).setTo(Scalar(255));

    // Compute the additional length that a feature patch must have
    // so that new detected corners are the same minimum distance apart
    // from already initialized features than from themselves
    int len = 2 * minDistance - features[0].roi.width;

    for (unsigned int i = 0; i < features.size(); i++) {

        Rect roi = features[i].roi;

        roi.x = max(0, roi.x - len / 2);
        roi.y = max(0, roi.y - len / 2);

        roi.width += len;
        roi.height += len;

        if (roi.x + roi.width > frame.cols)
            roi.width = frame.cols - roi.x;
        if (roi.y + roi.height > frame.rows)
            roi.height = frame.rows - roi.y;

        // Update the mask to reject the region around the ith feature
        mask(roi).setTo(Scalar(0));
    }

    goodFeaturesToTrack(frame, corners, maxCorners, qualityLevel, minDistance, mask);

    return corners;
}