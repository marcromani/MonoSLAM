#include <iostream>
#include <vector>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "camera.hpp"
#include "feature.hpp"
#include "map.hpp"
#include "poly.hpp"
#include "quaternion.hpp"
#include "util.hpp"

using namespace cv;
using namespace std;

int main() {

    vector<Point3d> vertices;
    vector<Point3i> faces;

    vertices.push_back(Point3d(-1, -1, -1));
    vertices.push_back(Point3d(-1, -1, 1));
    vertices.push_back(Point3d(-1, 1, -1));
    vertices.push_back(Point3d(-1, 1, 1));
    vertices.push_back(Point3d(1, -1, -1));
    vertices.push_back(Point3d(1, -1, 1));
    vertices.push_back(Point3d(1, 1, -1));
    vertices.push_back(Point3d(1, 1, 1));

    faces.push_back(Point3d(0, 1, 5));
    faces.push_back(Point3d(0, 4, 5));
    faces.push_back(Point3d(2, 3, 7));
    faces.push_back(Point3d(2, 6, 7));
    faces.push_back(Point3d(0, 1, 3));
    faces.push_back(Point3d(0, 2, 3));
    faces.push_back(Point3d(4, 5, 7));
    faces.push_back(Point3d(4, 6, 7));
    faces.push_back(Point3d(1, 3, 5));
    faces.push_back(Point3d(3, 5, 7));
    faces.push_back(Point3d(0, 2, 4));
    faces.push_back(Point3d(2, 4, 6));

    Polyhedron poly(vertices, faces);

    cout << poly.edges << endl;

    cout << poly.intersect(poly).vertices << endl;

    /*
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
    }*/
}