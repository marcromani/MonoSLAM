#include <iostream>
#include <string>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

vector<Point3f> computeObjPoints(Size patternSize, double squareSize);
void computeCornerPoints(int numImages, int rows, int cols, double squareSize,
                         vector<vector<Point2f>> &imagePoints,
                         vector<vector<Point3f>> &objectPoints);

int main(int argc, char **argv) {

    //int numImages = 15;
    //int rows = 6;
    //int cols = 8;
    //double squareSize = 0.0275;

    int numImages = 9;
    int rows = 7;
    int cols = 10;
    double squareSize = 0.0255;

    vector<vector<Point2f>> imagePoints;
    vector<vector<Point3f>> objectPoints;

    computeCornerPoints(numImages, rows, cols, squareSize, imagePoints, objectPoints);

    cout << "Calibrating with " << imagePoints.size() << "/" << numImages << " samples" << endl;

    //Size imageSize(640, 480);
    Size imageSize(3264, 1836);
    Mat cameraMat, distCoeff;
    vector<Mat> rvecs, tvecs;

    double rmse = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMat, distCoeff, rvecs, tvecs);

    cout << "Intrinsic matrix: " << endl;
    cout << cameraMat << endl;
    cout << "RMS reprojection error (pixels): " << rmse << endl;
    cout << "Distortion coeficients: " << distCoeff << endl;
}

vector<Point3f> computeObjPoints(Size patternSize, double squareSize) {

    vector<Point3f> objPoints;

    for (int i = 0; i < patternSize.height; i++) {

        for (int j = 0; j < patternSize.width; j++) {

            objPoints.push_back(Point3f(squareSize * j, squareSize * i, 0));
        }
    }

    return objPoints;
}

void computeCornerPoints(int numImages, int rows, int cols, double squareSize,
                         vector<vector<Point2f>> &imagePoints,
                         vector<vector<Point3f>> &objectPoints) {

    int pointsPerRow = cols - 1;
    int pointsPerCol = rows - 1;

    Size patternSize(pointsPerRow, pointsPerCol);

    namedWindow("Calibration", CV_WINDOW_AUTOSIZE);

    for (int i = 0; i < numImages; i++) {

        string idx = to_string(i + 1);

        //Mat img = imread("Calibration samples/chess" + idx + ".jpg", IMREAD_UNCHANGED);
        Mat img = imread("Calibration samples 2/chess" + idx + ".jpg", IMREAD_UNCHANGED);

        vector<Point2f> imgPoints;

        bool patternFound = findChessboardCorners(img, patternSize, imgPoints,
                            CALIB_CB_ADAPTIVE_THRESH +
                            CALIB_CB_NORMALIZE_IMAGE);

        if (patternFound) {

            Mat img_gray;

            cvtColor(img, img_gray, CV_BGR2GRAY);

            cornerSubPix(img_gray, imgPoints, Size(11, 11), Size(-1, -1),
                         TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

            imagePoints.push_back(imgPoints);
            objectPoints.push_back(computeObjPoints(patternSize, squareSize));

            drawChessboardCorners(img, patternSize, imgPoints, patternFound);

            imshow("Calibration", img);

            waitKey(0);

        } else {

            cout << "Image " << idx << " not used" << endl;
        }
    }
}