#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    cout << "2. Distort: input camera params and one undistorted image, then get the distorted image." << endl;

    string fn_image = "undistort.png";
    string fn_param = "params.json";
    if (argc > 1)
        fn_image = argv[1];
    if (argc > 2)
        fn_param = argv[2];
    Mat image_ud = imread(fn_image, IMREAD_GRAYSCALE);
    if (image_ud.empty()) {
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

    Mat R;
    Mat cam_mtx_ud = cam_mtx.clone();

    // add distortion
    Mat map_x = Mat(image_size, CV_32FC1);
    Mat map_y = Mat(image_size, CV_32FC1);
    vector<Point2f> pts_ud, pts_distort;
    for (int y = 0; y < image_size.height; ++y)
        for (int x = 0; x < image_size.width; ++x)
            pts_distort.emplace_back(x, y);
    undistortPoints(pts_distort, pts_ud, cam_mtx, dis_cef, R, cam_mtx_ud);
    for (int y = 0; y < image_size.height; ++y) {
        float* ptr1 = map_x.ptr<float>(y);
        float* ptr2 = map_y.ptr<float>(y);
        for (int x = 0; x < image_size.width; ++x) {
            const auto& pt = pts_ud[y * image_size.width + x];
            ptr1[x] = pt.x;
            ptr2[x] = pt.y;
        }
    }
    Mat image_distort;
    remap(image_ud, image_distort, map_x, map_y, INTER_LINEAR);

    string fn_dst = "distort.png";
    imwrite(fn_dst, image_distort);
    cout << "image saved to " << fn_dst << endl;

    return 0;
}
