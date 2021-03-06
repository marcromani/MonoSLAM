#include <chrono>
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

typedef chrono::duration<double, ratio<1, 1>> seconds_;
typedef chrono::high_resolution_clock clock_;

bool getOptions(int argc, char *argv[], string& cameraFile, int& patternRows, int& patternCols,
                double& squareSize);
void printUsage(const char *execName);

void readFrame(VideoCapture& cap, Mat& frame);

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

        cerr << "Could not open \"" << cameraFile << "\": camera settings not loaded" << endl;
        return -1;
    }

    Size frameSize;
    Mat K, distCoeffs;

    fs["Image_Size"] >> frameSize;
    fs["Intrinsic_Matrix"] >> K;
    fs["Distortion_Coefficients"] >> distCoeffs;

    fs.release();

    int patchSize = 22;
    int minDensity = 8;
    int maxDensity = 18;
    double failTolerance = 0.5;
    Mat accelerationVariances = (Mat_<double>(6, 1) << 0.025, 0.025, 0.025, 0.6, 0.6, 0.6);
    Mat measurementNoiseVariances = (Mat_<double>(2, 1) << 9, 9);

    // Build new map
    Map map(K, distCoeffs, frameSize, patchSize, minDensity, maxDensity, failTolerance,
            accelerationVariances, measurementNoiseVariances);

    // Chessboard size
    Size patternSize(patternCols - 1, patternRows - 1);

    // Camera initial state variances
    vector<double> var({0.0001, 0.0001, 0.0001,
                        0., 0., 0., 0.,
                        0.0001, 0.0001, 0.0001,
                        0.008, 0.008, 0.008,
                       });

    // Initialize video feed device
    VideoCapture cap(0);

    if (!cap.isOpened())
        return -1;

    // Create a window
    string window = "MonoSLAM";
    namedWindow(window, CV_WINDOW_AUTOSIZE);

    Mat frame, gray;

    bool init;

    chrono::time_point<clock_> t0;

    for (;;) {

        readFrame(cap, frame);

        t0 = clock_::now();

        imshow(window, frame);

        cvtColor(frame, gray, CV_BGR2GRAY);

        if ((init = map.initMap(gray, patternSize, squareSize, var)))
            break;

        if (waitKey(1) == 27)
            break;
    }

    if (!init)
        return 0;

    chrono::time_point<clock_> t1;

    for (;;) {

        cout << "Visible features: "
             << map.numVisibleFeatures << "/" << map.features.size() << endl;

        map.trackNewCandidates(gray);

        readFrame(cap, frame);

        t1 = clock_::now();
        double dt = chrono::duration_cast<seconds_>(t1 - t0).count();
        t0 = t1;

        cvtColor(frame, gray, CV_BGR2GRAY);
        cvtColor(gray, frame, CV_GRAY2RGB);

        map.predict(dt);
        map.update(gray, frame);

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

void readFrame(VideoCapture& cap, Mat& frame) {

    chrono::time_point<clock_> t0;
    double elapsed;

    double tol = 1 / double(30);

    do {

        t0 = clock_::now();
        cap.grab();
        elapsed = chrono::duration_cast<seconds_>(clock_::now() - t0).count();

    } while (elapsed < tol);

    cap.retrieve(frame);
}