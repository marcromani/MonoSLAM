#include <atomic>
#include <chrono>
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

typedef chrono::duration<double, ratio<1, 1>> seconds_;
typedef chrono::high_resolution_clock clock_;

constexpr double FPS = 60;

bool getOptions(int argc, char *argv[], string& cameraFile, int& patternRows, int& patternCols,
                double& squareSize);
void printUsage(const char *execName);

void readFrame(VideoCapture& cap, Mat& frame);

void updateCandidates(Map& map,
                      atomic<bool>& candidatesAvailable,
                      atomic<bool>& dataAvailable,
                      atomic<bool>& exitThread,
                      Buffer<Feature>& goodCandidates,
                      const Mat& gray,
                      const Mat& x,
                      const Mat& P);

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

    // Set capture size to 1024x768 (60 fps)
    cap.set(CAP_PROP_FRAME_WIDTH, 1024);
    cap.set(CAP_PROP_FRAME_HEIGHT, 768);

    // Create a window
    string window = "MonoSLAM";
    namedWindow(window, CV_WINDOW_AUTOSIZE);

    Mat frame, gray;

    bool init;

    chrono::time_point<clock_> t0, t1;

    atomic<bool> candidatesAvailable(false);
    atomic<bool> dataAvailable(false);
    atomic<bool> exitThread(false);

    Buffer<Feature> goodCandidates(100);

    Mat grayThread(frameSize, CV_8UC1);
    Mat xThread = map.x(Rect(0, 0, 1, 7)).clone();
    Mat PThread = map.P(Rect(0, 0, 7, 7)).clone();

    thread candidatesThread(updateCandidates,
                            ref(map),
                            ref(candidatesAvailable),
                            ref(dataAvailable),
                            ref(exitThread),
                            ref(goodCandidates),
                            ref(grayThread),
                            ref(xThread),
                            ref(PThread));

    for (;;) {

        readFrame(cap, frame);
        resize(frame, frame, frameSize);

        t0 = clock_::now();

        cvtColor(frame, gray, CV_BGR2GRAY);
        cvtColor(gray, frame, CV_GRAY2RGB);

        if ((init = map.initMap(gray, patternSize, squareSize, var)))
            break;

        imshow(window, frame);

        if (waitKey(1) == 27) {

            exitThread = true;
            break;
        }
    }

    if (!init) {

        candidatesThread.join();
        return 0;
    }

    for (;;) {

        cout << "Visible features: "
             << map.numVisibleFeatures << "/" << map.features.size() << endl
             << "Candidates: " << map.candidates.size() << endl;

        //cout << map.x.at<double>(0, 0) << " " << map.x.at<double>(1, 0) << " " << map.x.at<double>(2, 0) << endl;

        if (!candidatesAvailable) {

            if (map.trackNewCandidates(gray))
                candidatesAvailable = true;
        }

        readFrame(cap, frame);
        resize(frame, frame, frameSize);

        t1 = clock_::now();
        double dt = chrono::duration_cast<seconds_>(t1 - t0).count();
        t0 = t1;

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

        if (candidatesAvailable) {

            // Copy data
            gray.copyTo(grayThread);
            map.x(Rect(0, 0, 1, 7)).copyTo(xThread);
            map.P(Rect(0, 0, 7, 7)).copyTo(PThread);

            //dataAvailable = true;
        }

        imshow(window, frame);

        if (waitKey(1) == 27) {

            exitThread = true;
            break;
        }
    }

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

void readFrame(VideoCapture& cap, Mat& frame) {

    chrono::time_point<clock_> t0;
    double elapsed;

    double tol = 1 / double(2 * FPS);

    do {

        t0 = clock_::now();
        cap.grab();
        elapsed = chrono::duration_cast<seconds_>(clock_::now() - t0).count();

    } while (elapsed < tol);

    cap.retrieve(frame);
}

void updateCandidates(Map& map,
                      atomic<bool>& candidatesAvailable,
                      atomic<bool>& dataAvailable,
                      atomic<bool>& exitThread,
                      Buffer<Feature>& goodCandidates,
                      const Mat& gray,
                      const Mat& x,
                      const Mat& P) {

    for (;;) {

        while (!candidatesAvailable || !dataAvailable) {

            if (exitThread)
                return;
        };

        dataAvailable = false;

        map.updateCandidates(gray, x, P, goodCandidates);

        if (map.candidates.empty())
            candidatesAvailable = false;

        if (exitThread)
            return;

        this_thread::sleep_for(chrono::milliseconds(500));
    }
}