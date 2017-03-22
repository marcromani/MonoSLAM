#include <cmath>
#include <iostream>

#include "opencv2/highgui/highgui.hpp"

#include "camera.hpp"
#include "feature.hpp"
#include "quaternion.hpp"
#include "util.hpp"

using namespace cv;
using namespace std;

int main() {

    // Test 1

    double angle = 2 * (4*atan(1)) / 3;
    double rotv_[] = {angle/sqrt(3), angle/sqrt(3), angle/sqrt(3)};

    Mat rotv = Mat(3, 1, CV_64FC1, rotv_);

    Mat q = computeQuaternion(rotv);

    double i_[] = {1, 0, 0};
    double j_[] = {0, 1, 0};
    double k_[] = {0, 0, 1};

    Mat i = Mat(3, 1, CV_64FC1, i_);
    Mat j = Mat(3, 1, CV_64FC1, j_);
    Mat k = Mat(3, 1, CV_64FC1, k_);

    Mat R = getRotationMatrix(q);

    Mat ri = R * i;
    Mat rj = R * j;
    Mat rk = R * k;

    cout << ri << endl;
    cout << rj << endl;
    cout << rk << endl;

    // Test 2

    double axis_[] = {3/sqrt(14), 6/sqrt(14), 9/sqrt(14)};
    Mat axis = Mat(3, 1, CV_64FC1, axis_);
    Mat qq = computeQuaternion(axis);

    double vec_[] = {2, 4, 6};
    Mat vec = Mat(3, 1, CV_64FC1, vec_);

    Mat RR = getRotationMatrix(qq);
    Mat rotatedVec = RR * vec;

    cout << rotatedVec << endl;

    // Test 3

    double v2_[] = {0, -asin(1), 0};
    Mat v2 = Mat(3, 1, CV_64FC1, v2_);
    Mat qqq = computeQuaternion(v2);

    Mat R2 = getRotationMatrix(qqq);

    Mat rri = R2 * ri;
    Mat rrj = R2 * rj;
    Mat rrk = R2 * rk;

    cout << rri << endl;
    cout << rrj << endl;
    cout << rrk << endl;

    // Test 4

    Mat q3 = quaternionMultiply(qqq, q);
    Mat inv = quaternionInv(q3);

    double p1_[] = {0, 1, 0, 0};
    double p2_[] = {0, 0, 1, 0};
    double p3_[] = {0, 0, 0, 1};

    Mat p1 = Mat(4, 1, CV_64FC1, p1_);
    Mat p2 = Mat(4, 1, CV_64FC1, p2_);
    Mat p3 = Mat(4, 1, CV_64FC1, p3_);

    Mat x = quaternionMultiply(q3, quaternionMultiply(p1, inv));
    Mat y = quaternionMultiply(q3, quaternionMultiply(p2, inv));
    Mat z = quaternionMultiply(q3, quaternionMultiply(p3, inv));

    cout << x << endl;
    cout << y << endl;
    cout << z << endl;

    // Test 5

    // No rotation quaternion
    double q_[] = {1, 0, 0, 0};
    Mat noRot(4, 1, CV_64FC1, q_);

    // Camera intrinsic matrix
    double K_[] = {700.0852831876811, 0, 309.660602685939,
                   0, 700.3188537735915, 240.0018995844528,
                   0, 0, 1
                  };
    Mat K = Mat(3, 3, CV_64FC1, K_);

    // Camera distortion coeffs
    double distCoeffs_[] = {0.05753383038985068,
                            1.237208677293374,
                            0.003663057067132502,
                            0.0002173028769866434,
                            -5.930454782147544
                           };
    Mat distCoeffs = Mat(5, 1, CV_64FC1, distCoeffs_);

    Camera cam(q, q, q, q, q, K, distCoeffs, Size(640, 480));

    // Feature image
    Mat image = imread("../Calibration/Calibration samples/chess5.jpg", IMREAD_GRAYSCALE);

    // Patch region
    Rect roi = buildSquare(Point2i(320, 240), 479);

    // Feature position
    double pos_[] = {0, 0, 0};
    Mat pos(3, 1, CV_64FC1, pos_);

    // Feature normal
    double normal_[] = {1, 0, 0};
    Mat normal(3, 1, CV_64FC1, normal_);

    // Camera rotation
    double R_[] = {0, 1, 0, 0, 0, -1, -1, 0, 0};
    R = Mat(3, 3, CV_64FC1, R_);

    // Camera position
    double t_[] = {0.7, 0, 0};
    Mat t = Mat(3, 1, CV_64FC1, t_);

    Mat P;

    Feature feature(image, roi, pos, normal, R, t, P);

    double t2_[3] = {0.5, 0.5, 0};
    Mat t2 = Mat(3, 1, CV_64FC1, t2_);

    double c = 1 / sqrt(2);
    double R2_[9] = {-c, c, 0, 0, 0, -1, -c, -c, 0};
    R2 = Mat(3, 3, CV_64FC1, R2_);

    Mat warped = cam.warpPatch(feature.pos, feature.normal, feature.image, feature.roi, R, t, R2, t2);

    string window = "Warp Test";
    namedWindow(window, CV_WINDOW_AUTOSIZE);
    imshow(window, warped);

    while (true)
        if (waitKey(1) == 27)
            break;

    Mat dir;
    Point2d interval(5, 10);
    int samples = 100;

    Feature feature2(image, roi, R, t, dir, interval, samples);

    for (int i = 0; i < samples; i++) {

        cout << feature2.depths[i] << " " << feature2.probs[i] << endl;
    }
}