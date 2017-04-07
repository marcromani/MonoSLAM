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

int main() {

    string cameraConfig("../Calibration/camera.xml");

    FileStorage fs(cameraConfig, FileStorage::READ);

    if (!fs.isOpened()) {

        cout << "Could not open \"" << cameraConfig << "\": camera settings not loaded" << endl;
        return -1;
    }

    Size frameSize;
    Mat K, distCoeffs;

    fs["Image_Size"] >> frameSize;
    fs["Intrinsic_Matrix"] >> K;
    fs["Distortion_Coefficients"] >> distCoeffs;

    // Features patch size
    int patchSize = 11;

    // Build new map
    Map map(K, distCoeffs, frameSize, patchSize);

    // Chessboard physical properties
    Size patternSize(9, 6);
    double squareSize = 0.0255;

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

    for (;;) {

        cap.read(frame);

        imshow(window, frame);

        cvtColor(frame, gray, CV_BGR2GRAY);

        if (map.initMap(gray, patternSize, squareSize, var)) {

            cout << map.x << endl;
            cout << map.P << endl;

            map.x.at<double>(0, 0) = -1;

            map.x.at<double>(13, 0) = -1;
            map.x.at<double>(16, 0) = -1;
            map.x.at<double>(19, 0) = -1;
            map.x.at<double>(22, 0) = -1;

            cout << map.camera.r << endl;

            cout << map.features[0].pos << endl;
            cout << map.features[1].pos << endl;
            cout << map.features[2].pos << endl;
            cout << map.features[3].pos << endl;

            map.P.at<double>(0, 0) = -1;

            map.P.at<double>(13, 13) = -1;
            map.P.at<double>(16, 16) = -1;
            map.P.at<double>(19, 19) = -1;
            map.P.at<double>(22, 22) = -1;

            cout << map.camera.P.at<double>(0, 0) << endl;

            cout << map.features[0].P.at<double>(0, 0) << endl;
            cout << map.features[1].P.at<double>(0, 0) << endl;
            cout << map.features[2].P.at<double>(0, 0) << endl;
            cout << map.features[3].P.at<double>(0, 0) << endl;

            break;
        }

        if (waitKey(1) == 27)
            break;
    }
}