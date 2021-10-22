#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    cout << "0. Calibrate: input some images then get camera intrinsic params." << endl;

    string fp_images = "../images";
    if (argc > 1)
        fp_images = argv[1];
    cout << "image folder: " << fp_images << endl;

    // find images
    vector<String> fns;
    glob(fp_images + "/*.jpg", fns);
    cout << "num of images: " << fns.size() << endl;
    if (fns.size() < 3) {
        cout << "too few images, abort" << endl;
        return 0;
    }

    // detect points
    const int rows = 6;
    const int cols = 9;
    const float space = 50;
    vector<Point3f> tmp;
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x)
            tmp.emplace_back(x * space, y * space, 0.f);
    vector<vector<Point2f>> vpts2d;
    vector<vector<Point3f>> vpts3d;
    Size image_size;
    for (const auto& fn : fns) {
        auto image = imread(fn, IMREAD_GRAYSCALE);
        if (image.empty())
            continue;
        image_size = image.size();
        vector<Point2f> pts;
        if (findChessboardCorners(image, { cols, rows }, pts)) {
            vpts2d.push_back(pts);
            vpts3d.push_back(tmp);
        }
    }
    if (vpts2d.size() < 3) {
        cout << "too few valid images, abort" << endl;
        return 0;
    }

    // calibrate
    Mat cam_mtx, dis_cef;
    vector<Vec3d> rvecs, tvecs;
    calibrateCamera(vpts3d, vpts2d, image_size, cam_mtx, dis_cef, rvecs, tvecs);

    // output
    FileStorage fs;
    string fn_params = "params.json";
    if (fs.open(fn_params, FileStorage::WRITE)) {
        fs << "cam_mtx" << cam_mtx;
        fs << "dis_cef" << dis_cef;
        fs << "image_size" << image_size;
        cout << "params written to " << fn_params << endl;
    }

    return 0;
}
