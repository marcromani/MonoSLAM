#include <chrono>
#include <iostream>
#include <string>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

typedef chrono::high_resolution_clock clock_;
typedef chrono::duration<double, ratio<1>> second_;

bool getOptions(int argc, char *argv[], int& rows, int& cols, double& squareSize, double& tol,
                string& filename);
void printUsage(const char *execName);

vector<Point3f> computeObjPoints(Size patternSize, double squareSize);
bool addCornerPoints(const Mat& image, int rows, int cols, double squareSize,
                     vector<vector<Point2f>>& imagePoints,
                     vector<vector<Point3f>>& objectPoints);

int main(int argc, char *argv[]) {

    int rows, cols;
    double squareSize, tol;
    string filename;

    tol = 1.5; // Default sample reprojection error tolerance

    if (!getOptions(argc, argv, rows, cols, squareSize, tol, filename)) {

        printUsage(argv[0]);
        return -1;
    }

    filename += string(".xml");

    cout << "Calibrating with a " << cols << "x" << rows
         << " chessboard with a square size of " << squareSize << " meters" << endl
         << "Sample reprojection error tolerance is set to " << tol << " pixels" << endl;

    // Initialize video feed device
    VideoCapture cap(0);

    if (!cap.isOpened())
        return -1;

    // Create a window
    string window = "Camera Calibration";
    namedWindow(window, CV_WINDOW_AUTOSIZE);

    Mat frame, patternFrame;
    vector<vector<Point2f>> imagePoints;
    vector<vector<Point3f>> objectPoints;

    // Create the image to be shown
    cap.read(frame);
    Mat showFrame(frame.rows, 2*frame.cols, frame.type());

    int i = 0;
    chrono::time_point<clock_> t0 = clock_::now();

    while (true) {

        cap.read(frame);

        frame.copyTo(showFrame(Rect(0, 0, frame.cols, frame.rows)));

        double elapsed = chrono::duration_cast<second_>(clock_::now() - t0).count();

        // Leave some seconds between each pattern capture
        if (elapsed > 2) {

            frame.copyTo(patternFrame);

            if (addCornerPoints(patternFrame, rows, cols, squareSize, imagePoints, objectPoints)) {

                patternFrame.copyTo(showFrame(Rect(frame.cols-1, 0, frame.cols, frame.rows)));
                cout << "\r" << ++i << " samples captured" << flush;
                t0 = clock_::now();
            }
        }

        imshow(window, showFrame);

        if (waitKey(1) == 27)
            break;
    }

    if (i == 0) {

        cout << "Calibration canceled: no samples to work with" << endl;
        return -1;
    }

    cout << endl;

    Size imageSize = frame.size();
    Mat cameraMat, distCoeffs, stdDevIntrinsics, stdDevExtrinsics, perViewErrors;
    vector<Mat> rvecs, tvecs;

    double rmse = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMat, distCoeffs, rvecs, tvecs,
                                  stdDevIntrinsics, stdDevExtrinsics, perViewErrors);

    vector<vector<Point2f>> newImagePoints;
    vector<vector<Point3f>> newObjectPoints;

    // Remove samples with high reprojection error
    for (int i = 0; i < perViewErrors.rows; i++) {

        if (perViewErrors.at<double>(i, 0) < tol) {

            newImagePoints.push_back(imagePoints[i]);
            newObjectPoints.push_back(objectPoints[i]);
        }
    }

    if (newImagePoints.empty()) {

        cout << "Calibration canceled: no good samples to work with" << endl;
        return -1;
    }

    if (newImagePoints.size() < imagePoints.size()) {

        // Recalibrate with good samples
        rmse = calibrateCamera(newObjectPoints, newImagePoints, imageSize, cameraMat, distCoeffs, rvecs, tvecs,
                               stdDevIntrinsics, stdDevExtrinsics, perViewErrors);

        cout << "Camera calibrated using only " << newImagePoints.size() << " samples" << endl;
    }

    cout << "Intrinsic matrix:" << endl
         << cameraMat << endl
         << "Distortion coeficients: " << distCoeffs << endl
         << "RMS reprojection error (pixels): " << rmse << endl;

    FileStorage fs(filename, FileStorage::WRITE);

    if (!fs.isOpened()) {

        cout << "Could not open \"" << filename << "\": camera settings not saved" << endl;
        return -1;
    }

    fs << "Image_Size" << imageSize;
    fs << "Intrinsic_Matrix" << cameraMat;
    fs << "Distortion_Coefficients" << distCoeffs;

    cout << "Camera settings saved to \"" << filename << "\"" << endl;
}

bool getOptions(int argc, char *argv[], int& rows, int& cols, double& squareSize, double& tol,
                string& filename) {

    // Look for the help option
    for (int i = 1; i < argc; i++)
        if (string(argv[i]) == "-h" || string(argv[i]) == "--help")
            return false;

    // Check the number of input options
    if (argc != 5 && argc != 6)
        return false;

    rows = (int) strtol(argv[1], NULL, 10);
    if (rows == 0 || errno == ERANGE) return false;

    cols = (int) strtol(argv[2], NULL, 10);
    if (cols == 0 || errno == ERANGE) return false;

    squareSize = strtod(argv[3], NULL);
    if (squareSize == 0 || errno == ERANGE) return false;

    if (argc == 6) {

        tol = strtod(argv[4], NULL);
        if (tol == 0 || errno == ERANGE) return false;
    }

    filename = (argc == 6) ? argv[5] : argv[4];

    return true;
}

void printUsage(const char *execName) {

    cerr << "Usage: " << execName << " rows cols square_size [tol] filename (-h, --help for help)" << endl;
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

bool addCornerPoints(const Mat& image, int rows, int cols, double squareSize,
                     vector<vector<Point2f>>& imagePoints,
                     vector<vector<Point3f>>& objectPoints) {

    int pointsPerRow = cols - 1;
    int pointsPerCol = rows - 1;

    Size patternSize(pointsPerRow, pointsPerCol);

    vector<Point2f> imgPoints;

    bool patternFound = findChessboardCorners(image, patternSize, imgPoints,
                        CALIB_CB_ADAPTIVE_THRESH +
                        CALIB_CB_NORMALIZE_IMAGE +
                        CALIB_CB_FAST_CHECK);

    if (patternFound) {

        Mat gray;

        cvtColor(image, gray, CV_BGR2GRAY);

        cornerSubPix(gray, imgPoints, Size(11, 11), Size(-1, -1),
                     TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

        imagePoints.push_back(imgPoints);
        objectPoints.push_back(computeObjPoints(patternSize, squareSize));

        drawChessboardCorners(image, patternSize, imgPoints, patternFound);
    }

    return patternFound;
}