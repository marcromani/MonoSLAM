#include <iostream>

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

    // Camera intrinsic matrix
    double K_[] = {700.0852831876811, 0, 309.660602685939,
                   0, 700.3188537735915, 240.0018995844528,
                   0, 0, 1
                  };

    Mat K(3, 3, CV_64FC1, K_);

    // Camera distortion coeffs
    double distCoeffs_[] = {0.05753383038985068,
                            1.237208677293374,
                            0.003663057067132502,
                            0.0002173028769866434,
                            -5.930454782147544
                           };

    Mat distCoeffs(5, 1, CV_64FC1, distCoeffs_);

    // Camera image size
    Size frameSize(640, 480);

    // Features patch size
    int patchSize = 11;

    // Build new map
    Map map(K, distCoeffs, frameSize, patchSize);

    // Chessboard physical properties
    Size patternSize(9, 6);
    double squareSize = 0.0255;

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

        if (map.initMap(gray, patternSize, squareSize)) {

            cout << map.x << endl;
            cout << map.P << endl;
            map.x.at<double>(22, 0) = -1;
            cout << map.features[0].pos << endl;
            cout << map.features[1].pos << endl;
            cout << map.features[2].pos << endl;
            cout << map.features[3].pos << endl;

            cout << map.features[0].roi << endl;
            imshow(window, map.features[0].image(map.features[0].roi));

            if (waitKey(1000000) == 27)
                break;

            //break;
        }

        if (waitKey(1) == 27)
            break;
    }


}