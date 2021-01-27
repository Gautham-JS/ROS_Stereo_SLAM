#include <iostream>
#include <vector>
#include <string>

#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/Header.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "pcl_ros/point_cloud.h"
#include "pcl_conversions/pcl_conversions.h"
#include <pcl/filters/statistical_outlier_removal.h>
//#include "pcl_ros/filters/statistical_outlier_removal.h"

using namespace std;
using namespace cv;

typedef pcl::PointCloud<pcl::PointXYZRGB> cloudType;

class StereoProcess{
    private:
        ros::NodeHandle nh;
        ros::Publisher pub; 
    public:       
        const char*lFptr; const char*rFptr;
        
        double baseline = 0.5707;
        double focal_x = 7.188560000000e+02;
        double cx = 6.071928000000e+02;
        double focal_y = 7.188560000000e+02;
        double cy = 1.852157000000e+02;

        cv::Mat K = (cv::Mat1d(3,3) << focal_x, 0, cx, 0, focal_y, cy, 0, 0, 1);

        bool init=false;

        cv::Mat lImg, rImg, prevImg;
        vector<cv::Point3f> tri3dPoints, color3dMap;

        StereoProcess(const char* lptr, const char* rptr){
            lFptr = lptr;
            rFptr = rptr;
            pub = nh.advertise<cloudType>("StereoCloud",1);
            ros::Rate loop_rate(10);
        }

        void pclPublish(vector<Point3f>&pts3d, vector<cv::Point3f>&colorMap);
        cv::Mat getImg(const char* fptr, int iter);
        void mainLoop();
        void stereoTriangulate(cv::Mat im1, cv::Mat im2, vector<cv::Point3f>&out3d);
        cv::Mat stereoMatch(int iter);
        void reprojectDisparity(cv::Mat disp, vector<cv::Point3f>&reproject3dPoints, vector<cv::Point3f>&colorMap);
        void visualizeCloud(vector<cv::Point3f>pts3d, vector<cv::Point3f>colorMap);
        void monocularTriangulate(cv::Mat im1, cv::Mat im2, vector<cv::Point3f>&out3d);
        void pyrLKTracking(cv::Mat refimg, cv::Mat curImg, vector<cv::Point2f>refPts,
                            vector<cv::Point3f>ref3dPts, vector<cv::Point2f>&tracked2dPoints,
                            vector<cv::Point3f>tracked3dPoints);
        vector<int> removeRedundancy(vector<cv::Point2f>&base2d, vector<cv::Point2f>&newPts,
                            vector<int>&mask, int radius=10);

};
