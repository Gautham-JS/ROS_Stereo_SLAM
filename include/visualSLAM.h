#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <stdio.h>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

#include <pangolin/pangolin.h>
#include <unistd.h>

#include "ros/ros.h"
#include "pcl_ros/point_cloud.h"
#include "pcl_conversions/pcl_conversions.h"
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/ply_io.h>

#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/Header.h"
#include "nav_msgs/Path.h"
#include "geometry_msgs/PoseStamped.h"

#include "DBoW2/DBoW2.h"

#include "poseGraph.h"
#include "DloopDet.h"
#include "TemplatedLoopDetector.h"
#include "monoUtils.h"

using namespace std;
using namespace cv;
using namespace DLoopDetector;
using namespace DBoW2;

#define X_BOUND 1000
#define Y_BOUND 1500

typedef pcl::PointCloud<pcl::PointXYZRGB> cloudType;
typedef TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB> KeyFrameSelection;

struct keyFrame{
    int idx = -1;
    bool retrack = false;
    Mat R,t;
    vector<Point3f> ref3dCoords;
    vector<Point3f> transformed3d;
    vector<Point2f> refFeats;
};

class visualSLAM{
    private:
        ros::NodeHandle nh;

        ros::Publisher mapPublisher;
        ros::Publisher posePublisher;
        ros::Publisher trajectoryPublisher;

        nav_msgs::Path trajectoryMsg;

    public:
        int seqNo;
        double baseline = 0.54;
        int Xbias = 750;
        int Ybias = 200;
        int LCidx = 0;
        int cooldownTimer = 0;
        bool LC_FLAG = false;
        bool SHUTDOWN_FLAG = false;
        bool RENDER_SHUTDOWN = false;
        bool DENSE_FLAG = true;

        string absPath;
        const char* lFptr; const char* rFptr;

        double focal_x = 7.188560000000e+02;
        double cx = 6.071928000000e+02;
        double focal_y = 7.188560000000e+02;
        double cy = 1.852157000000e+02;
        
        Mat K = (Mat1d(3,3) << focal_x, 0, cx, 0, focal_y, cy, 0, 0, 1);
        
        Mat referenceImg, currentImage;
        vector<Point3f> referencePoints3D, mapPts, untransformed, colors;
        vector<Point2f> referencePoints2D;
        vector<vector<Point3f>> mapHistory, colorHistory;
        vector<cv::Mat> trajectory;
        vector<cv::Mat> Rhistory;
        vector<keyFrame> keyFrameHistory;
        vector<vector<double>> gtTraj;
        vector<Eigen::Isometry3d> isoVector;

        vector<Point2f> inlierReferencePyrLKPts;
        Mat canvas = Mat::zeros(X_BOUND, Y_BOUND, CV_8UC3);
        Mat ret;

        globalPoseGraph poseGraph;

        std::shared_ptr<OrbLoopDetector> loopDetector;
        std::shared_ptr<OrbVocabulary> voc;
        std::shared_ptr<KeyFrameSelection> KFselector;
        
        std::string vocfile;
        std::string plySavepath = "map.ply";
        string trajectory_file = "trajectory.txt";

        visualSLAM(int Seq, const char*Lfptr, const char*Rfptr, std::string vocPath){
            lFptr = Lfptr;
            rFptr = Rfptr;
            vocfile = vocPath;

            Params param;
            param.image_rows = 1241;
            param.image_cols = 376;
            param.use_nss = true;
            param.alpha = 0.9;
            param.k = 1;
            param.geom_check = GEOM_DI;
            param.di_levels = 2;

            voc.reset(new OrbVocabulary());
            cerr<<"Loading Place Recognition vocabulary : "<<vocfile<<endl;
            voc->load(vocfile);
            cerr<<"Done"<<endl;

            loopDetector.reset(new OrbLoopDetector(*voc, param));
            loopDetector->allocate(4000);

            mapPublisher = nh.advertise<cloudType>("SLAM/map",1);
            posePublisher = nh.advertise<geometry_msgs::PoseStamped>("SLAM/pose",1);
            trajectoryPublisher = nh.advertise<nav_msgs::Path>("SLAM/trajectory",1);
        }

        void restructure (cv::Mat& plain, vector<FORB::TDescriptor> &descriptors){  
            const int L = plain.rows;
            descriptors.resize(L);
            for (unsigned int i = 0; i < (unsigned int)plain.rows; i++) {
                descriptors[i] = plain.row(i);
            }
        }

        vector<KeyPoint> denseKeypointExtractor(Mat img, int stepSize);
        void denseLKtracking(Mat refImg, Mat curImg, vector<Point2f>&refPts, vector<Point2f>&trackPts);
        void FmatThresholding(vector<Point2f>&refPts, vector<Point2f>&trkPts);

        void checkLoopDetectorStatus(Mat img, int idx);
        void stereoTriangulate(Mat im1, Mat im2, 
                            vector<Point3f>&ref3dPts, 
                            vector<Point2f>&ref2dPts);
        void PyrLKtrackFrame2Frame(Mat refimg, Mat curImg, vector<Point2f>refPts, vector<Point3f>ref3dpts,
                                            vector<Point2f>&refRetpts, vector<Point3f>&ref3dretPts);
        vector<int> removeDuplicates(vector<Point2f>&baseref2dFeatures, vector<Point2f>&newref2dFeatures,
                                    vector<int>&mask, int radius=10);
        void insertKeyFrames(int start, Mat imL, Mat imR, Mat&pose4dTransform, vector<Point2f>&ftrPts, vector<Point3f>&ref3dCoords);
        vector<Point3f> update3dtransformation(vector<Point3f>& pt3d, Mat& pose4dTransform);
        Mat loadImageL(int iter);
        Mat loadImageR(int iter);
        void PerspectiveNpointEstimation(Mat&prevImg, Mat&curImg, vector<Point2f>&ref2dPoints, vector<Point3f>&ref3dPoints, 
                                        vector<Point2f>&tracked2dPoints, vector<Point3f>&tracked3dPoints, Mat&rvec, Mat&tvec,vector<int>&inliers);
        void initSequence();

        void initPangolin();
        void DrawTrajectory(vector<Eigen::Isometry3d>&poses, vector<vector<Point3f>>&pts3,vector<vector<Point3f>>&colorData);

        void stageForPGO(Mat Rl, Mat tl, Mat Rg, Mat tg, bool loopClose);
        void updateOdometry(vector<Eigen::Isometry3d>&T);

        void SORcloud(vector<Point3f>&ref3d, vector<Point3f>&colorMap);
        void rosPublish(vector<vector<Point3f>>&pt3d, Mat&trajROS, Mat&Rmat);
};