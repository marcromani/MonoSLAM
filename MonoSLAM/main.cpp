#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "buffer.hpp"
#include "camera.hpp"
#include "feature.hpp"
#include "map.hpp"
#include "quaternion.hpp"
#include "util.hpp"

using namespace cv;
using namespace std;

constexpr double dt = 1./30;

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
    int minDensity = 10;
    int maxDensity = 12;
    double failTolerance = 0.5;
    Mat accelerationVariances = (Mat_<double>(6, 1) << 1, 1, 1, 2.46, 2.46, 2.46);
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
    VideoCapture cap("/dev/video2");

    if (!cap.isOpened())
        return -1;

    // Set capture size to 1024x768 at 60 fps
    // Default is 640x480 at 120 fps and images are darker due to low exposure
    cap.set(CAP_PROP_FRAME_WIDTH, 1024);
    cap.set(CAP_PROP_FRAME_HEIGHT, 768);

    // Create a window
    string window = "MonoSLAM";
    namedWindow(window, CV_WINDOW_AUTOSIZE);

    Mat frame, gray, grayThread;

    bool init;

    thread candidatesThread;
    Buffer<Feature> goodCandidates(32);
    atomic<bool> threadCompleted(true);

    for (;;) {

        cap.read(frame);
        resize(frame, frame, frameSize);

        cvtColor(frame, gray, CV_BGR2GRAY);
        cvtColor(gray, frame, CV_GRAY2RGB);

        if ((init = map.initMap(gray, patternSize, squareSize, var)))
            break;

        imshow(window, frame);

        if (waitKey(1) == 27)
            break;
    }

    if (!init)
        return 0;

    for (;;) {

        /*cout << "Visible features: "
             << map.numVisibleFeatures << "/" << map.features.size() << endl
             << "Candidates: " << map.candidates.size() << endl;*/

        if (threadCompleted)
            map.trackNewCandidates(gray);

        cap.read(frame);
        resize(frame, frame, frameSize);

        cvtColor(frame, gray, CV_BGR2GRAY);
        cvtColor(gray, frame, CV_GRAY2RGB);

        while (!goodCandidates.empty()) {

            Feature feature = goodCandidates.pop();

            map.features.push_back(feature);

            map.x.resize(map.x.rows + 3);
            feature.pos.copyTo(map.x(Rect(0, map.x.rows - 3, 1, 3)));

            Mat tmp(map.P.rows + 3, map.P.cols + 3, CV_64FC1, Scalar(0));
            map.P.copyTo(tmp(Rect(0, 0, map.P.cols, map.P.rows)));
            feature.P.copyTo(tmp(Rect(map.P.cols, map.P.rows, 3, 3)));
            map.P = tmp;
        }

        map.predict(dt);
        map.update(gray, frame);

        if (!map.candidates.empty() && threadCompleted) {

            if (candidatesThread.joinable())
                candidatesThread.join();

            threadCompleted = false;

            gray.copyTo(grayThread);

            candidatesThread = thread(&Map::updateCandidates, &map,
                                      ref(grayThread), ref(goodCandidates), ref(threadCompleted));
        }

        imshow(window, frame);

        int key = waitKey(1);

        if (key == 32)
            for (unsigned int j = 0; j < map.features.size(); j++)
                cout << map.x(Rect(0, 13 + 3*j, 1, 3)).t() << endl;

        else if (key == 27)
            break;
    }

    if (candidatesThread.joinable())
        candidatesThread.join();
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