/*
-->GauthWare, LSM, 01/2021
*/
#ifndef ODOM_H
#define ODOM_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "DBoW2/DBoW2.h"

#include <iostream>
#include <vector>
#include <algorithm>

#include "poseGraph.h"
#include "DloopDet.h"
#include "TemplatedLoopDetector.h"

using namespace cv;
using namespace std;
using namespace DLoopDetector;
using namespace DBoW2;

typedef TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB> KeyFrameSelection;

struct imageMetadata{
    Mat im1, im2, R, t;
    vector<Point2f> refPts, trkPts;
    vector<Point3f> pts3d;
};


class monoOdom{
    public:
        int iter = 0;
        int LCidx = 0;
        int cooldownTimer = 0;
        int inLength = 0;
        bool LC_FLAG = false;
        
        const char* lFptr;
        const char* rFptr;
        Mat im1, im2, R, t, rvec, tvec, Rmono, tmono;

        double focal_x = 7.188560000000e+02;
        double cx = 6.071928000000e+02;
        double focal_y = 7.188560000000e+02;
        double cy = 1.852157000000e+02;
        vector<Mat> trajectory;

        Mat canvas = Mat::zeros(1000,1500, CV_8UC3);
        Mat debug1, debug2, debug3; 
        Mat K = (Mat1d(3,3) << focal_x, 0, cx, 0, focal_y, cy, 0, 0, 1);
        Mat im1prev, im2prev;
        
        std::shared_ptr<OrbLoopDetector> loopDetector;
        std::shared_ptr<OrbVocabulary> voc;
        std::shared_ptr<KeyFrameSelection> KFselector;
        std::string vocfile = "/home/gautham/Documents/Projects/LargeScaleMapping/orb_voc00.yml.gz";

        vector<Point2f> prevFeatures;
        vector<Point3f> prev3d;

        imageMetadata imMeta;
        imageMetadata prevImMeta;

        globalPoseGraph poseGraph;

        std::string LC_debug_status = "No loop closure found yet";

        monoOdom(int seq, const char* lptr, const char* rptr){
            lFptr = lptr; rFptr = rptr;
            
            Params param;
            param.image_rows = 1241;
            param.image_cols = 376;
            param.use_nss = true;
            param.alpha = 0.9;
            param.k = 1;
            param.geom_check = GEOM_DI;
            param.di_levels = 2;

            voc.reset(new OrbVocabulary());
            cerr<<"Loading vocabulary file : "<<vocfile<<endl;
            voc->load(vocfile);
            cerr<<"Done"<<endl;

            loopDetector.reset(new OrbLoopDetector(*voc, param));
            loopDetector->allocate(4000);
        }
        void restructure (cv::Mat& plain, vector<FORB::TDescriptor> &descriptors){  
            const int L = plain.rows;
            descriptors.resize(L);
            for (unsigned int i = 0; i < (unsigned int)plain.rows; i++) {
                descriptors[i] = plain.row(i);
            }
        }

        void checkLoopDetector(Mat img, int idx);
        void relocalize(int start, Mat imL, Mat imR, Mat&inv_transform, vector<Point2f>&ftrPts, vector<Point3f>&pts3d);
        void PyrLKtrackFrame2Frame(Mat refimg, Mat curImg, vector<Point2f>refPts, vector<Point3f>ref3dpts,
                                            vector<Point2f>&refRetpts, vector<Point3f>&ref3dretPts);
        vector<KeyPoint> denseKeypointExtractor(Mat img, int stepSize);
        void stageForPGO(Mat Rl, Mat tl, Mat Rg, Mat tg, bool loopClose);
        Mat drawDeltas(Mat im, vector<Point2f> in1, vector<Point2f> in2);
        void monoTriangulate(Mat img1, Mat img2,vector<Point2f>&ref2dPts, vector<Point2f>&trk2dPts,vector<Point3f>&ref3dpts);
        Mat drawDeltasErr(Mat img1, vector<Point2f>inlier1, vector<Point2f>inlier2);
        void pyrLKtracking(Mat refImg, Mat curImg, vector<Point2f>&refPts, vector<Point2f>&trackPts);
        void FmatThresholding(vector<Point2f>&refPts, vector<Point2f>&trkPts);
        void relocalizeFrames(int start, Mat img1, Mat img2, Mat&invTransform, vector<Point2f>&ftrPts, vector<Point3f>pts3d);
        Mat loadImage(int iter);
        void updateOdometry(vector<Eigen::Isometry3d>&T);
        void initSequence();
        void loopSequence();
        void pureMonocularSequence();
};

#endif