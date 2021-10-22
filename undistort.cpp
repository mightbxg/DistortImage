#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    cout << "1. Undistort: input camera params and one distorted image, then get the undistorted image." << endl;

    string fn_image = "../images/left03.jpg";
    string fn_param = "params.json";
    if (argc > 1)
        fn_image = argv[1];
    if (argc > 2)
        fn_param = argv[2];
    Mat image = imread(fn_image, IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "cannot load image: " << fn_image << endl;
        return 0;
    }
    Mat cam_mtx, dis_cef;
    Size image_size;
    auto loadParams = [&]() {
        FileStorage fs;
        if (!fs.open(fn_param, FileStorage::READ))
            return false;
        fs["cam_mtx"] >> cam_mtx;
        fs["dis_cef"] >> dis_cef;
        fs["image_size"] >> image_size;
        return !cam_mtx.empty() && !dis_cef.empty() && image_size.area() > 0;
    };
    if (!loadParams()) {
        cout << "cannot load camera params from " << fn_param << endl;
        return 0;
    }

    Mat map_x, map_y;
    initUndistortRectifyMap(cam_mtx, dis_cef, Mat(), cam_mtx, image_size, CV_32FC1, map_x, map_y);
    Mat image_dst;
    remap(image, image_dst, map_x, map_y, INTER_LINEAR);
    string fn_dst = "undistort.png";
    imwrite(fn_dst, image_dst);
    cout << "image saved to " << fn_dst << endl;

    return 0;
}
