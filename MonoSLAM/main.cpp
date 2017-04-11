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

    int patchSize = 25;
    int minDensity = 8;
    int maxDensity = 18;
    double failTol = 0.4;

    // Build new map
    Map map(K, distCoeffs, frameSize, patchSize, minDensity, maxDensity, failTol);

    // Chessboard size
    Size patternSize(patternCols - 1, patternRows - 1);

    // Camera initial state variances
    vector<double> var({0.02, 0.02, 0.02,
                        0.02, 0.02, 0.02, 0.02,
                        0.1, 0.1, 0.1,
                        0.18, 0.18, 0.18
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

        map.update();
        map.drawVisibleFeatures(frame);
        map.trackNewCandidates(gray);

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