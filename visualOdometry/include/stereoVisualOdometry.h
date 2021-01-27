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
#include <opencv2/features2d/features2d.hpp>


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

typedef TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB> KeyFrameSelection;

struct keyFrame{
    int idx = -1;
    bool retrack = false;
    Mat R,t;
    vector<Point3f> pts3d;
};

class visualOdometry{
    public:
        int seqNo;
        double baseline = 0.54;
        int Xbias = 750;
        int Ybias = 200;
        int LCidx = 0;
        int cooldownTimer = 0;
        bool LC_FLAG = false;

        string absPath;
        const char* lFptr; const char* rFptr;

        double focal_x = 7.188560000000e+02;
        double cx = 6.071928000000e+02;
        double focal_y = 7.188560000000e+02;
        double cy = 1.852157000000e+02;
        
        Mat K = (Mat1d(3,3) << focal_x, 0, cx, 0, focal_y, cy, 0, 0, 1);
        
        Mat referenceImg, currentImage;
        vector<Point3f> referencePoints3D, mapPts, untransformed;
        vector<Point2f> referencePoints2D;
        vector<vector<Point3f>> mapHistory;
        vector<cv::Mat> trajectory;
        vector<keyFrame> keyFrameHistory;
        vector<vector<double>> gtTraj;

        vector<Point2f> inlierReferencePyrLKPts;
        Mat canvas = Mat::zeros(X_BOUND, Y_BOUND, CV_8UC3);
        Mat ret;

        globalPoseGraph poseGraph;

        std::shared_ptr<OrbLoopDetector> loopDetector;
        std::shared_ptr<OrbVocabulary> voc;
        std::shared_ptr<KeyFrameSelection> KFselector;
        std::string vocfile = "/home/gautham/Documents/Projects/LargeScaleMapping/orb_voc00.yml.gz";

        visualOdometry(int Seq, const char*Lfptr, const char*Rfptr){
            lFptr = Lfptr;
            rFptr = Rfptr;

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
        Mat loadImageL(int iter);
        Mat loadImageR(int iter);

        void checkLoopDetectorStatus(Mat img, int idx);
        void stereoTriangulate(Mat im1, Mat im2, 
                            vector<Point3f>&ref3dPts, 
                            vector<Point2f>&ref2dPts);
        Mat drawDeltas(Mat im, vector<Point2f> in1, vector<Point2f> in2);
        void RANSACThreshold(Mat refImg, Mat curImg, vector<Point2f>refPts, vector<Point3f>ref3dpts, vector<Point2f>&inTrkPts, vector<Point3f>&in3dpts);
        void PyrLKtrackFrame2Frame(Mat refimg, Mat curImg, vector<Point2f>refPts, vector<Point3f>ref3dpts,
                                            vector<Point2f>&refRetpts, vector<Point3f>&ref3dretPts);
        vector<int> removeDuplicates(vector<Point2f>&baseFeatures, vector<Point2f>&newFeatures,
                                    vector<int>&mask, int radius=10);
        void repjojectionError(Mat im, vector<Point2f> pt2d, vector<Point3f>pts3d);
        void relocalizeFrames(int start, Mat imL, Mat imR, Mat&inv_transform, vector<Point2f>&ftrPts, vector<Point3f>&pts3d);
        vector<Point3f> update3dtransformation(vector<Point3f>& pt3d, Mat& inv_transform);

        void initSequence();

                                            
};