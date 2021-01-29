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

#include "./include/poseGraph.h"
#include "./include/DloopDet.h"
#include "./include/TemplatedLoopDetector.h"
#include "./include/monoUtils.h"

using namespace std;
using namespace cv;
using namespace DLoopDetector;
using namespace DBoW2;

#define X_BOUND 1000
#define Y_BOUND 1500

typedef TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB> KeyFrameSelection;

struct metaData{
    int idx = -1;
    vector<Point2f> refPts;
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
        vector<Point3f> referencePoints3D, mapPts;
        vector<Point2f> referencePoints2D;
        vector<vector<Point3f>> mapHistory;
        vector<cv::Mat> trajectory;
        vector<metaData> metaHistory;

        vector<Point2f> inlierReferencePyrLKPts;
        Mat canvas = Mat::zeros(X_BOUND, Y_BOUND, CV_8UC3);
        Mat ret;

        globalPoseGraph poseGraph;
        Eigen::Isometry3d lcMeasurementT;

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

        void checkLoopDetectorStatus(Mat img, int idx){
            Ptr<FeatureDetector> orb = ORB::create();
            vector<KeyPoint> kp;
            Mat desc;
            vector<FORB::TDescriptor> descriptors;

            orb->detectAndCompute(img, Mat(), kp, desc);
            restructure(desc, descriptors);
            DetectionResult result;
            loopDetector->detectLoop(kp, descriptors, result);
            if(result.detection() &&(result.query-result.match > 100) && cooldownTimer==0){
                cerr<<"Found Loop Closure between "<<idx<<" and "<<result.match<<endl;
                LC_FLAG = true;
                LCidx = result.match-1;
                cooldownTimer = 100;
            }
        }

        void stereoTriangulate(Mat im1, Mat im2, 
                            vector<Point3f>&ref3dPts, 
                            vector<Point2f>&ref2dPts){
            
            //Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create(1000);
            Ptr<FeatureDetector> detector = xfeatures2d::SURF::create(1500);
            //Ptr<FeatureDetector> detector = ORB::create(4500);
            // Ptr<FeatureDetector> fast = xfeatures2d::StarDetector::create();
            // Ptr<DescriptorExtractor> brief = xfeatures2d::BriefDescriptorExtractor::create();
            if(!im1.data || !im2.data){
                cout<<"NULL IMG"<<endl;
                return;
            }

            vector<KeyPoint> kp1, kp2;
            Mat desc1, desc2;

            std::thread left([&](){
                detector->detect(im1, kp1);
                detector->compute(im1, kp1, desc1);
            });
            std::thread right([&](){    
                detector->detect(im2, kp2);
                detector->compute(im2, kp2, desc2);
            });
            left.join();
            right.join();

            desc1.convertTo(desc1, CV_32F);
            desc2.convertTo(desc2, CV_32F);

            BFMatcher matcher;
            vector<vector<DMatch>> matches;
            matcher.knnMatch(desc1, desc2, matches, 2);

            vector<Point2f> pt1, pt2;
            for(int i=0; i<matches.size(); i++){
                DMatch &m = matches[i][0]; DMatch &n = matches[i][1];
                if(m.distance<0.8*n.distance){
                    pt1.emplace_back(kp1[m.queryIdx].pt);
                    pt2.emplace_back(kp2[m.trainIdx].pt);
                }
            }

            vector<Point3f> pts3d;

            Mat P1 = Mat::zeros(3,4, CV_64F);
            Mat P2 = Mat::zeros(3,4, CV_64F);
            P1.at<double>(0,0) = 1; P1.at<double>(1,1) = 1; P1.at<double>(2,2) = 1;
            P2.at<double>(0,0) = 1; P2.at<double>(1,1) = 1; P2.at<double>(2,2) = 1;
            P2.at<double>(0,3) = -baseline;

            P1 = K*P1;
            P2 = K*P2;

            Mat est3d;
            triangulatePoints(P1, P2, pt1, pt2, est3d);
            //cout<<est3d.size()<<endl;

            for(int i=0; i<est3d.cols; i++){
                Point3f localpt;
                localpt.x = est3d.at<float>(0,i) / est3d.at<float>(3,i);
                localpt.y = est3d.at<float>(1,i) / est3d.at<float>(3,i);
                localpt.z = est3d.at<float>(2,i) / est3d.at<float>(3,i);
                pts3d.emplace_back(localpt);
            }

            vector<Point2f> reprojection;
            for(int k=0; k<pts3d.size(); k++){
                Point2f projection; Point3f pt3d = pts3d[k];
                projection.x = pt3d.x; projection.y = pt3d.y;
                reprojection.push_back(projection);
            }
            //cout<<reprojection.size()<<" PTSIZE "<<pt1.size()<<endl;
            ret = drawDeltas(im1, pt1, reprojection);

            ref3dPts = pts3d;
            ref2dPts = pt1;
        }

        Mat drawDeltas(Mat im, vector<Point2f> in1, vector<Point2f> in2){
            Mat frame;
            im.copyTo(frame);

            for(int i=0; i<in1.size(); i++){
                Point2f pt1 = in1[i];
                Point2f pt2 = in2[i];
                line(frame, pt1, pt2, Scalar(0,255,0),2);
                circle(frame, pt1, 5, Scalar(0,0,255));
                circle(frame, pt2, 5, Scalar(255,0,0));
            }
            return frame;
        }

        void RANSACThreshold(Mat refImg, Mat curImg, vector<Point2f>refPts, vector<Point3f>ref3dpts, vector<Point2f>&inTrkPts, vector<Point3f>&in3dpts){
            Mat F;
            vector<uchar> mask;
            vector<Point2f>inlierRef, inlierTrk;
            vector<Point3f>inlier3dPts;
            F = findFundamentalMat(refPts, inTrkPts, CV_RANSAC, 3.0, 0.99, mask);
            for(size_t j=0; j<mask.size(); j++){
                if(mask[j]==1){
                    inlierRef.emplace_back(refPts[j]);
                    inlierTrk.emplace_back(inTrkPts[j]);
                    inlier3dPts.emplace_back(ref3dpts[j]);
                }
            }
            inTrkPts.clear();

        }

        void PyrLKtrackFrame2Frame(Mat refimg, Mat curImg, vector<Point2f>refPts, vector<Point3f>ref3dpts,
                                            vector<Point2f>&refRetpts, vector<Point3f>&ref3dretPts){
            vector<Point2f> trackPts;
            vector<uchar> Idx;
            vector<float> err;

            calcOpticalFlowPyrLK(refimg, curImg, refPts, trackPts,Idx, err);

            vector<Point2f> inlierRefPts;
            vector<Point3f> inlierRef3dPts;
            vector<Point2f> inlierTracked;
            vector<int> res;

            for(int j=0; j<refPts.size(); j++){
                if(Idx[j]==1){
                    inlierRefPts.push_back(refPts[j]);
                    ref3dretPts.push_back(ref3dpts[j]);
                    refRetpts.push_back(trackPts[j]);
                }
            }
            //refRetpts = inlierTracked;
            //ref3dretPts = inlierRef3dPts;
            inlierReferencePyrLKPts = inlierRefPts;
        }

        vector<int> removeDuplicates(vector<Point2f>&baseFeatures, vector<Point2f>&newFeatures,
                                    vector<int>&mask, int radius=10){
            vector<int> res;
            for(int i=0; i<newFeatures.size(); i++){
                Point2f&p2 = newFeatures[i];
                bool inRange=false;
                
                for(auto j:mask){
                    Point2f&p1 = baseFeatures[j];
                    if(norm(p1-p2)<radius){
                        inRange=true;
                        break;
                    }
                }

                if(!inRange){res.push_back(i);}
            }
            return res;
        }

        void repjojectionError(Mat im, vector<Point2f> pt2d, vector<Point3f>pts3d){

        }


        void relocalizeFrames(int start, Mat imL, Mat imR, Mat&inv_transform, vector<Point2f>&ftrPts, vector<Point3f>&pts3d){
            vector<Point2f> new2d;
            vector<Point3f> new3d;
            
            ftrPts.clear();
            pts3d.clear();

            stereoTriangulate(imL, imR, new3d, new2d);

            for(int i=0; i<new3d.size(); i++){
                Point3f pt = new3d[i];
                Point3f p;

                p.x = inv_transform.at<double>(0,0)*pt.x + inv_transform.at<double>(0,1)*pt.y + inv_transform.at<double>(0,2)*pt.z + inv_transform.at<double>(0,3);
                p.y = inv_transform.at<double>(1,0)*pt.x + inv_transform.at<double>(1,1)*pt.y + inv_transform.at<double>(1,2)*pt.z + inv_transform.at<double>(1,3);
                p.z = inv_transform.at<double>(2,0)*pt.x + inv_transform.at<double>(2,1)*pt.y + inv_transform.at<double>(2,2)*pt.z + inv_transform.at<double>(2,3);

                pts3d.emplace_back(p);
                ftrPts.emplace_back(new2d[i]);
            }
        }

        Mat loadImageL(int iter){
            char FileName[200];
            sprintf(FileName, lFptr, iter);

            Mat im = imread(FileName);
            if(!im.data){
                cout<<"yikes, failed to fetch frame, check the paths"<<endl;
            }
            return im;
        }
        Mat loadImageR(int iter){
            char FileName[200];
            sprintf(FileName, rFptr, iter);

            Mat im = imread(FileName);
            if(!im.data){
                cout<<"yikes, failed to fetch frame, check the paths"<<endl;
            }
            return im;
        }

        void getLCMeasurement(metaData refKF, metaData matchKF){
            Mat imref = loadImageL(refKF.idx);
            Mat imtrk = loadImageL(matchKF.idx);
            Mat tv,rv;Mat distCoeffs = Mat::zeros(4,1,CV_64F);
            vector<int> inliers;

            vector<Point2f> trkFeatures; vector<Point3f> trk3dpts;
            PyrLKtrackFrame2Frame(imref, imtrk, refKF.refPts, refKF.pts3d, trkFeatures, trk3dpts);

            solvePnPRansac(trk3dpts, trkFeatures, K, distCoeffs, rv, tv, false, 100, 0.1, 0.999, inliers);

            Mat R; Rodrigues(rv, R);

            R = R.t();
            tv = -R*tv;

            lcMeasurementT = cvMat2Eigen(R,tv);
        }

        void initSequence(){
            int iter = 0;
            metaData meta;

            char FileName1[200], filename2[200];
            sprintf(FileName1, lFptr, iter);
            sprintf(filename2, rFptr, iter);

            // Mat imL = imread(FileName1);
            // Mat imR = imread(filename2);

            Mat imL = loadImageL(iter);
            Mat imR = loadImageR(iter);

            referenceImg = imL;

            vector<Point2f> features;
            vector<Point3f> pts3d;
            stereoTriangulate(imL, imR, pts3d, features);
            poseGraph.initializeGraph();
            meta.pts3d = pts3d; meta.refPts = features; meta.idx = 0;

            //metaHistory.reserve(4000);
            //metaHistory.emplace_back(meta);

            for(int iter=1; iter<4500; iter++){
                //cout<<"PROCESSING FRAME "<<iter<<endl;
                currentImage = loadImageL(iter);
                
                vector<Point3f> refPts3d; vector<Point2f> refFeatures;
                PyrLKtrackFrame2Frame(referenceImg, currentImage, features, pts3d, refFeatures, refPts3d);
                
                Mat distCoeffs = Mat::zeros(4,1,CV_64F);
                Mat rvec, tvec; vector<int> inliers;


                checkLoopDetectorStatus(currentImage,iter);

                solvePnPRansac(refPts3d, refFeatures, K, distCoeffs, rvec, tvec, false,100, 1.0, 0.99, inliers);
                if(inliers.size()<10){
                    cout<<"Low inlier count at "<<inliers.size()<<", skipping frame "<<iter<<endl;
                    break;
                }
                Mat R,Rt;
                Rodrigues(rvec, Rt);
    
                R = Rt.t();
                Mat t = -R*tvec;

                if(LC_FLAG){
                    cerr<<"LCmeas bw"<<iter-1<<" and "<<LCidx<<endl;
                    //getLCMeasurement(metaHistory[iter-1], metaHistory[LCidx]);
                    stageForPGO(R, t, R, t, true);
                    stageForPGO(R, t, R, t, false);
                    std::vector<Eigen::Isometry3d> trans = poseGraph.globalOptimize();
                    Mat interT = Eigen2cvMat(trans[trans.size()-1]);
                    t = interT.t();
                    //trajectory.clear();
                    //updateOdometry(trans);
                }
                else{
                    stageForPGO(R, t, R, t, false);
                }

                Mat inv_transform = Mat::zeros(3,4, CV_64F);
                R.col(0).copyTo(inv_transform.col(0));
                R.col(1).copyTo(inv_transform.col(1));
                R.col(2).copyTo(inv_transform.col(2));
                t.copyTo(inv_transform.col(3));


                cerr<<"Inlier Size : "<<inliers.size()<<endl;
                if(inliers.size()<150){
                    cerr<<"ENTERING KEYFRAME AT "<<iter<<endl;
                    Mat i1 = loadImageL(iter); Mat i2 = loadImageR(iter);
                    relocalizeFrames(0, i1, i2, inv_transform, features, pts3d);
                    drawMapPoints(pts3d);
                    //mapHistory.emplace_back(pts3d);
                }
                else{
                    pts3d = refPts3d;
                    features = refFeatures;
                }

                if(cooldownTimer!=0){
                    cooldownTimer--;
                }
                metaData meta;
                meta.pts3d = pts3d; meta.refPts = features;
                meta.idx = iter;

                referenceImg = currentImage;
                LC_FLAG = false;

                t.convertTo(t, CV_32F);
                Mat frame = drawDeltas(currentImage, inlierReferencePyrLKPts, refFeatures);
                
                double Xgt, Ygt, Zgt;
                getAbsoluteScale(iter, Xgt, Ygt, Zgt);

                
                Point2f centerGT = Point2f(int(Xgt)+Xbias, int(Zgt)+Ybias);
                Point2f center = Point2f(int(t.at<float>(0)) + Xbias, int(t.at<float>(2)) + Ybias);
                circle(canvas, center ,1, Scalar(0,0,255), 2);
                circle(canvas, centerGT,1,Scalar(0,255,0),2);
                rectangle(canvas, Point2f(10, 30), Point2f(550, 50),  Scalar(0,0,0), cv::FILLED);

                imshow("frame", frame);
                imshow("trajectory", canvas);

                int k = waitKey(100);
                if (k=='q'){
                    break;
                }
            }
            cerr<<"Total map size :"<<mapPts.size()<<endl;
            poseGraph.saveStructure();
            vector<Eigen::Isometry3d> res = poseGraph.globalOptimize();
            updateOdometry(res);
            
            cerr<<"\nRedrawing Trajectory\nPress any key to quit (even power key, lmao.)"<<endl;
            for(Mat& position : trajectory){
                Point2f centerMono = Point2f(int(position.at<double>(0)) + 750, int(position.at<double>(2)) + 200);
                circle(canvas, centerMono ,1, Scalar(255,0,0), 1);
            }
            imshow("trajectory", canvas);
            waitKey(0);
            imwrite("Trajectory.png",canvas);
            cerr<<"Trajectory Saved"<<endl;
        }
        void stageForPGO(Mat Rl, Mat tl, Mat Rg, Mat tg, bool loopClose){
            Eigen::Isometry3d localT, globalT;
            localT = cvMat2Eigen(Rl, tl);
            globalT = cvMat2Eigen(Rg, tg);
            Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
            if(loopClose){
                LC_FLAG = true;
                poseGraph.addLoopClosure(globalT,LCidx+1);
            }
            else{
                poseGraph.augmentNode(localT, globalT);
            } 
        }

        void updateOdometry(vector<Eigen::Isometry3d>&T){
            cerr<<"\n\nUpdating global odometry measurements"<<endl;
            for(Eigen::Isometry3d &isoMatrix : T){
                Mat t = Eigen2cvMat(isoMatrix);
                trajectory.emplace_back(t.clone());
            }
        }

        void drawMapPoints(vector<Point3f> &pts3d){
            vector<double> x3,y3,z3;
            int boundInvalidity = 0;
            int domainImvalidity = 0; 
            int depthInvalidity = 0;
            x3.reserve(pts3d.size()); y3.reserve(pts3d.size()); z3.reserve(pts3d.size());
            for(Point3f pt: pts3d){
                x3.emplace_back(pt.x);
                y3.emplace_back(pt.y*-1);
                z3.emplace_back(pt.z*-1);
            }

            double x3max,x3min, y3max,y3min, z3max,z3min;
            x3max = *max_element(x3.begin(), x3.end());
            x3min = *min_element(x3.begin(), x3.end());

            y3max = *max_element(y3.begin(), y3.end());
            y3min = *min_element(y3.begin(), y3.end());

            z3max = *max_element(z3.begin(), z3.end());
            z3min = *min_element(z3.begin(), z3.end());
            
            double yl = 0.1*(y3max-y3min);
            double ym = 0.5*(y3max-y3min);
            double zm = 0.2*(y3max-y3min);

            for(Point3f p: pts3d){
                if(-1*p.z>zm){
                    depthInvalidity++;
                    continue;
                }
                if(p.x+Xbias>Y_BOUND or p.z+Ybias>X_BOUND){
                    boundInvalidity++;
                    continue;
                }
                //mapPts.emplace_back(p);
                //cerr<<"Setting pt color at "<<p.x+Xbias<<" "<<p.z+Ybias<<endl;
                Vec3b color; color[0] = 200; color[1] = 200; color[2] = 200;
                canvas.at<Vec3b>(Point(p.x+Xbias, p.z+Ybias)) = color;
            }
            cerr<<"Depth Invalidity : "<<depthInvalidity<<endl;
            cerr<<"Bound Invalidity : "<<boundInvalidity<<endl<<endl;
        }
};

int main(){
    const char* impathL = "/media/gautham/Seagate Backup Plus Drive/Datasets/dataset/sequences/00/image_0/%0.6d.png";
    const char* impathR = "/media/gautham/Seagate Backup Plus Drive/Datasets/dataset/sequences/00/image_1/%0.6d.png";

    vector<Point2f> ref2d; vector<Point3f> ref3d;

    visualOdometry VO(0, impathL, impathR);
    char FileName1[200], filename2[200];
    sprintf(FileName1, impathL, 0);
    sprintf(filename2, impathR, 0);

    Mat im1 = imread(FileName1);
    Mat im2 = imread(filename2);
    //VO.stereoTriangulate(im1, im2, ref3d, ref2d);
    //visualOdometry* VO = new visualOdometry(0, impathR, impathL);
    VO.initSequence();

}








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

#include "./include/poseGraph.h"
#include "./include/DloopDet.h"
#include "./include/TemplatedLoopDetector.h"
#include "./include/monoUtils.h"

using namespace std;
using namespace cv;
using namespace DLoopDetector;
using namespace DBoW2;

#define X_BOUND 1000
#define Y_BOUND 1500

typedef TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB> KeyFrameSelection;

struct metaData{
    int idx = -1;
    vector<Point2f> refPts;
    vector<Point2f> trkPts;
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
        vector<Point3f> referencePoints3D, mapPts;
        vector<Point2f> referencePoints2D;
        vector<vector<Point3f>> mapHistory;
        vector<cv::Mat> trajectory;

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

        void checkLoopDetectorStatus(Mat img, int idx){
            Ptr<FeatureDetector> orb = ORB::create();
            vector<KeyPoint> kp;
            Mat desc;
            vector<FORB::TDescriptor> descriptors;

            orb->detectAndCompute(img, Mat(), kp, desc);
            restructure(desc, descriptors);
            DetectionResult result;
            loopDetector->detectLoop(kp, descriptors, result);
            if(result.detection() &&(result.query-result.match > 100) && cooldownTimer==0){
                cerr<<"Found Loop Closure between "<<idx<<" and "<<result.match<<endl;
                LC_FLAG = true;
                LCidx = result.match-1;
                cooldownTimer = 100;
            }
        }

        void stereoTriangulate(Mat im1, Mat im2, 
                            vector<Point3f>&ref3dPts, 
                            vector<Point2f>&ref2dPts){
            
            //Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create(1000);
            Ptr<FeatureDetector> detector = xfeatures2d::SURF::create(1200);
            //Ptr<FeatureDetector> detector = ORB::create(4500);
            // Ptr<FeatureDetector> fast = xfeatures2d::StarDetector::create();
            // Ptr<DescriptorExtractor> brief = xfeatures2d::BriefDescriptorExtractor::create();
            if(!im1.data || !im2.data){
                cout<<"NULL IMG"<<endl;
                return;
            }

            vector<KeyPoint> kp1, kp2;
            Mat desc1, desc2;

            std::thread left([&](){
                detector->detect(im1, kp1);
                detector->compute(im1, kp1, desc1);
            });
            std::thread right([&](){    
                detector->detect(im2, kp2);
                detector->compute(im2, kp2, desc2);
            });
            left.join();
            right.join();

            desc1.convertTo(desc1, CV_32F);
            desc2.convertTo(desc2, CV_32F);

            BFMatcher matcher;
            vector<vector<DMatch>> matches;
            matcher.knnMatch(desc1, desc2, matches, 2);

            vector<Point2f> pt1, pt2;
            for(int i=0; i<matches.size(); i++){
                DMatch &m = matches[i][0]; DMatch &n = matches[i][1];
                if(m.distance<0.8*n.distance){
                    pt1.emplace_back(kp1[m.queryIdx].pt);
                    pt2.emplace_back(kp2[m.trainIdx].pt);
                }
            }

            vector<Point3f> pts3d;

            Mat P1 = Mat::zeros(3,4, CV_64F);
            Mat P2 = Mat::zeros(3,4, CV_64F);
            P1.at<double>(0,0) = 1; P1.at<double>(1,1) = 1; P1.at<double>(2,2) = 1;
            P2.at<double>(0,0) = 1; P2.at<double>(1,1) = 1; P2.at<double>(2,2) = 1;
            P2.at<double>(0,3) = -baseline;

            P1 = K*P1;
            P2 = K*P2;

            Mat est3d;
            triangulatePoints(P1, P2, pt1, pt2, est3d);
            //cout<<est3d.size()<<endl;

            for(int i=0; i<est3d.cols; i++){
                Point3f localpt;
                localpt.x = est3d.at<float>(0,i) / est3d.at<float>(3,i);
                localpt.y = est3d.at<float>(1,i) / est3d.at<float>(3,i);
                localpt.z = est3d.at<float>(2,i) / est3d.at<float>(3,i);
                pts3d.emplace_back(localpt);
            }

            vector<Point2f> reprojection;
            for(int k=0; k<pts3d.size(); k++){
                Point2f projection; Point3f pt3d = pts3d[k];
                projection.x = pt3d.x; projection.y = pt3d.y;
                reprojection.push_back(projection);
            }
            //cout<<reprojection.size()<<" PTSIZE "<<pt1.size()<<endl;
            ret = drawDeltas(im1, pt1, reprojection);

            ref3dPts = pts3d;
            ref2dPts = pt1;
        }

        Mat drawDeltas(Mat im, vector<Point2f> in1, vector<Point2f> in2){
            Mat frame;
            im.copyTo(frame);

            for(int i=0; i<in1.size(); i++){
                Point2f pt1 = in1[i];
                Point2f pt2 = in2[i];
                line(frame, pt1, pt2, Scalar(0,255,0),2);
                circle(frame, pt1, 5, Scalar(0,0,255));
                circle(frame, pt2, 5, Scalar(255,0,0));
            }
            return frame;
        }

        void RANSACThreshold(Mat refImg, Mat curImg, vector<Point2f>refPts, vector<Point3f>ref3dpts, vector<Point2f>&inTrkPts, vector<Point3f>&in3dpts){
            Mat F;
            vector<uchar> mask;
            vector<Point2f>inlierRef, inlierTrk;
            vector<Point3f>inlier3dPts;
            F = findFundamentalMat(refPts, inTrkPts, CV_RANSAC, 3.0, 0.99, mask);
            for(size_t j=0; j<mask.size(); j++){
                if(mask[j]==1){
                    inlierRef.emplace_back(refPts[j]);
                    inlierTrk.emplace_back(inTrkPts[j]);
                    inlier3dPts.emplace_back(ref3dpts[j]);
                }
            }
            inTrkPts.clear();

        }

        void PyrLKtrackFrame2Frame(Mat refimg, Mat curImg, vector<Point2f>refPts, vector<Point3f>ref3dpts,
                                            vector<Point2f>&refRetpts, vector<Point3f>&ref3dretPts){
            vector<Point2f> trackPts;
            vector<uchar> Idx;
            vector<float> err;

            calcOpticalFlowPyrLK(refimg, curImg, refPts, trackPts,Idx, err);

            vector<Point2f> inlierRefPts;
            vector<Point3f> inlierRef3dPts;
            vector<Point2f> inlierTracked;
            vector<int> res;

            for(int j=0; j<refPts.size(); j++){
                if(Idx[j]==1){
                    inlierRefPts.push_back(refPts[j]);
                    ref3dretPts.push_back(ref3dpts[j]);
                    refRetpts.push_back(trackPts[j]);
                }
            }
            //refRetpts = inlierTracked;
            //ref3dretPts = inlierRef3dPts;
            inlierReferencePyrLKPts = inlierRefPts;
        }

        vector<int> removeDuplicates(vector<Point2f>&baseFeatures, vector<Point2f>&newFeatures,
                                    vector<int>&mask, int radius=10){
            vector<int> res;
            for(int i=0; i<newFeatures.size(); i++){
                Point2f&p2 = newFeatures[i];
                bool inRange=false;
                
                for(auto j:mask){
                    Point2f&p1 = baseFeatures[j];
                    if(norm(p1-p2)<radius){
                        inRange=true;
                        break;
                    }
                }

                if(!inRange){res.push_back(i);}
            }
            return res;
        }

        void repjojectionError(Mat im, vector<Point2f> pt2d, vector<Point3f>pts3d){

        }


        void relocalizeFrames(int start, Mat imL, Mat imR, Mat&inv_transform, vector<Point2f>&ftrPts, vector<Point3f>&pts3d){
            vector<Point2f> new2d;
            vector<Point3f> new3d;
            
            ftrPts.clear();
            pts3d.clear();

            stereoTriangulate(imL, imR, new3d, new2d);

            for(int i=0; i<new3d.size(); i++){
                Point3f pt = new3d[i];
                Point3f p;

                p.x = inv_transform.at<double>(0,0)*pt.x + inv_transform.at<double>(0,1)*pt.y + inv_transform.at<double>(0,2)*pt.z + inv_transform.at<double>(0,3);
                p.y = inv_transform.at<double>(1,0)*pt.x + inv_transform.at<double>(1,1)*pt.y + inv_transform.at<double>(1,2)*pt.z + inv_transform.at<double>(1,3);
                p.z = inv_transform.at<double>(2,0)*pt.x + inv_transform.at<double>(2,1)*pt.y + inv_transform.at<double>(2,2)*pt.z + inv_transform.at<double>(2,3);

                pts3d.emplace_back(p);
                ftrPts.emplace_back(new2d[i]);
            }
        }

        Mat loadImageL(int iter){
            char FileName[200];
            sprintf(FileName, lFptr, iter);

            Mat im = imread(FileName);
            if(!im.data){
                cout<<"yikes, failed to fetch frame, check the paths"<<endl;
            }
            return im;
        }
        Mat loadImageR(int iter){
            char FileName[200];
            sprintf(FileName, rFptr, iter);

            Mat im = imread(FileName);
            if(!im.data){
                cout<<"yikes, failed to fetch frame, check the paths"<<endl;
            }
            return im;
        }

        void initSequence(){
            int iter = 0;
            char FileName1[200], filename2[200];
            sprintf(FileName1, lFptr, iter);
            sprintf(filename2, rFptr, iter);

            // Mat imL = imread(FileName1);
            // Mat imR = imread(filename2);

            Mat imL = loadImageL(iter);
            Mat imR = loadImageR(iter);

            referenceImg = imL;

            vector<Point2f> features;
            vector<Point3f> pts3d;
            stereoTriangulate(imL, imR, pts3d, features);
            poseGraph.initializeGraph();

            for(int iter=1; iter<4000; iter++){
                //cout<<"PROCESSING FRAME "<<iter<<endl;
                currentImage = loadImageL(iter);
                
                vector<Point3f> refPts3d; vector<Point2f> refFeatures;
                PyrLKtrackFrame2Frame(referenceImg, currentImage, features, pts3d, refFeatures, refPts3d);
                
                Mat distCoeffs = Mat::zeros(4,1,CV_64F);
                Mat rvec, tvec; vector<int> inliers;


                checkLoopDetectorStatus(currentImage,iter);

                solvePnPRansac(refPts3d, refFeatures, K, distCoeffs, rvec, tvec, false,100, 1.0, 0.99, inliers);
                if(inliers.size()<10){
                    cout<<"Low inlier count at "<<inliers.size()<<", skipping frame "<<iter<<endl;
                    continue;
                }
                Mat R,Rt;
                Rodrigues(rvec, Rt);
    
                R = Rt.t();
                Mat t = -R*tvec;

                if(LC_FLAG){
                    stageForPGO(R, t, R, t, true);
                    stageForPGO(R, t, R, t, false);
                    std::vector<Eigen::Isometry3d> trans = poseGraph.globalOptimize();
                    Mat interT = Eigen2cvMat(trans[trans.size()-1]);
                    t = interT.t();
                    trajectory.clear();
                    updateOdometry(trans);
                }
                else{
                    stageForPGO(R, t, R, t, false);
                }

                Mat inv_transform = Mat::zeros(3,4, CV_64F);
                R.col(0).copyTo(inv_transform.col(0));
                R.col(1).copyTo(inv_transform.col(1));
                R.col(2).copyTo(inv_transform.col(2));
                t.copyTo(inv_transform.col(3));


                //cerr<<"Inlier Size : "<<inliers.size()<<endl;
                if(inliers.size()<150 or LC_FLAG==true){
                    cerr<<"ENTERING KEYFRAME AT "<<iter<<endl;
                    Mat i1 = loadImageL(iter); Mat i2 = loadImageR(iter);
                    relocalizeFrames(0, i1, i2, inv_transform, features, pts3d);
                    drawMapPoints(pts3d);
                    mapHistory.emplace_back(pts3d);
                }
                else{
                    pts3d = refPts3d;
                    features = refFeatures;
                }

                if(cooldownTimer!=0){
                    cooldownTimer--;
                }
                referenceImg = currentImage;
                LC_FLAG = false;

                t.convertTo(t, CV_32F);
                Mat frame = drawDeltas(currentImage, inlierReferencePyrLKPts, refFeatures);
                
                double Xgt, Ygt, Zgt;
                getAbsoluteScale(iter, Xgt, Ygt, Zgt);

                
                Point2f centerGT = Point2f(int(Xgt)+Xbias, int(Zgt)+Ybias);
                Point2f center = Point2f(int(t.at<float>(0)) + Xbias, int(t.at<float>(2)) + Ybias);
                circle(canvas, center ,1, Scalar(0,0,255), 2);
                circle(canvas, centerGT,1,Scalar(0,255,0),2);
                rectangle(canvas, Point2f(10, 30), Point2f(550, 50),  Scalar(0,0,0), cv::FILLED);

                imshow("frame", frame);
                imshow("trajectory", canvas);

                int k = waitKey(100);
                if (k=='q'){
                    break;
                }
            }
            cerr<<"Total map size :"<<mapPts.size()<<endl;
            poseGraph.saveStructure();
            vector<Eigen::Isometry3d> res = poseGraph.globalOptimize();
            updateOdometry(res);
            
            cerr<<"\nRedrawing Trajectory\nPress any key to quit (even power key, lmao.)"<<endl;
            for(Mat& position : trajectory){
                Point2f centerMono = Point2f(int(position.at<double>(0)) + 750, int(position.at<double>(2)) + 200);
                circle(canvas, centerMono ,1, Scalar(255,0,0), 1);
            }
            imshow("trajectory", canvas);
            waitKey(0);
            imwrite("Trajectory.png",canvas);
            cerr<<"Trajectory Saved"<<endl;
        }
        void stageForPGO(Mat Rl, Mat tl, Mat Rg, Mat tg, bool loopClose){
            Eigen::Isometry3d localT, globalT;
            localT = cvMat2Eigen(Rl, tl);
            globalT = cvMat2Eigen(Rg, tg);
            if(loopClose){
                LC_FLAG = true;
                poseGraph.addLoopClosure(globalT,LCidx);
            }
            else{
                poseGraph.augmentNode(localT, globalT);
            } 
        }

        void updateOdometry(vector<Eigen::Isometry3d>&T){
            cerr<<"\n\nUpdating global odometry measurements"<<endl;
            for(Eigen::Isometry3d &isoMatrix : T){
                Mat t = Eigen2cvMat(isoMatrix);
                trajectory.emplace_back(t.clone());
            }
        }

        void drawMapPoints(vector<Point3f> &pts3d){
            vector<double> x3,y3,z3;
            int boundInvalidity = 0;
            int domainImvalidity = 0; 
            int depthInvalidity = 0;
            x3.reserve(pts3d.size()); y3.reserve(pts3d.size()); z3.reserve(pts3d.size());
            for(Point3f pt: pts3d){
                x3.emplace_back(pt.x);
                y3.emplace_back(pt.y*-1);
                z3.emplace_back(pt.z*-1);
            }

            double x3max,x3min, y3max,y3min, z3max,z3min;
            x3max = *max_element(x3.begin(), x3.end());
            x3min = *min_element(x3.begin(), x3.end());

            y3max = *max_element(y3.begin(), y3.end());
            y3min = *min_element(y3.begin(), y3.end());

            z3max = *max_element(z3.begin(), z3.end());
            z3min = *min_element(z3.begin(), z3.end());
            
            double yl = 0.1*(y3max-y3min);
            double ym = 0.5*(y3max-y3min);
            double zm = 0.2*(y3max-y3min);

            for(Point3f p: pts3d){
                if(-1*p.z>zm){
                    depthInvalidity++;
                    continue;
                }
                if(p.x+Xbias>Y_BOUND or p.z+Ybias>X_BOUND){
                    boundInvalidity++;
                    continue;
                }
                mapPts.emplace_back(p);
                //cerr<<"Setting pt color at "<<p.x+Xbias<<" "<<p.z+Ybias<<endl;
                Vec3b color; color[0] = 200; color[1] = 200; color[2] = 200;
                canvas.at<Vec3b>(Point(p.x+Xbias, p.z+Ybias)) = color;
            }
            cerr<<"Depth Invalidity : "<<depthInvalidity<<endl;
            cerr<<"Bound Invalidity : "<<boundInvalidity<<endl<<endl;
        }
};

int main(){
    const char* impathL = "/media/gautham/Seagate Backup Plus Drive/Datasets/dataset/sequences/00/image_0/%0.6d.png";
    const char* impathR = "/media/gautham/Seagate Backup Plus Drive/Datasets/dataset/sequences/00/image_1/%0.6d.png";

    vector<Point2f> ref2d; vector<Point3f> ref3d;

    visualOdometry VO(0, impathL, impathR);
    char FileName1[200], filename2[200];
    sprintf(FileName1, impathL, 0);
    sprintf(filename2, impathR, 0);

    Mat im1 = imread(FileName1);
    Mat im2 = imread(filename2);
    //VO.stereoTriangulate(im1, im2, ref3d, ref2d);
    //visualOdometry* VO = new visualOdometry(0, impathR, impathL);
    VO.initSequence();
}


#include "../include/visualSLAM.h"

void visualSLAM::stereoTriangulate(Mat im1, Mat im2, 
                            vector<Point3f>&ref3dPts, 
                            vector<Point2f>&ref2dPts){
    
    //Ptr<FeatureDetector> detector = xref2dFeatures2d::SIFT::create(1000);
    //Ptr<FeatureDetector> detector = ORB::create(4500);
    // Ptr<FeatureDetector> fast = xref2dFeatures2d::StarDetector::create();
    // Ptr<DescriptorExtractor> brief = xref2dFeatures2d::BriefDescriptorExtractor::create();
    if(!im1.data || !im2.data){
        cout<<"NULL IMG"<<endl;
        return;
    }
    vector<Point2f> pt1, pt2;

    if(DENSE_FLAG){
        vector<KeyPoint> dkps;
        dkps = denseKeypointExtractor(im1, 20);

        vector<Point2f> refPts;
        for(size_t i=0; i<dkps.size(); i++){
            refPts.emplace_back(dkps[i].pt);
        }

        vector<Point2f> trkPts;
        pyrLKtracking(im1, im2, refPts, trkPts);
        FmatThresholding(refPts, trkPts);

        pt1 = refPts, pt2 = trkPts;
    }
    else{
        Ptr<FeatureDetector> detector = xfeatures2d::SURF::create(1200);
        vector<KeyPoint> kp1, kp2;
        Mat desc1, desc2;

        std::thread left([&](){
            detector->detect(im1, kp1);
            detector->compute(im1, kp1, desc1);
        });
        std::thread right([&](){    
            detector->detect(im2, kp2);
            detector->compute(im2, kp2, desc2);
        });
        left.join();
        right.join();

        desc1.convertTo(desc1, CV_32F);
        desc2.convertTo(desc2, CV_32F);

        BFMatcher matcher;
        vector<vector<DMatch>> matches;
        matcher.knnMatch(desc1, desc2, matches, 2);

        for(int i=0; i<matches.size(); i++){
            DMatch &m = matches[i][0]; DMatch &n = matches[i][1];
            if(m.distance<0.8*n.distance){
                pt1.emplace_back(kp1[m.queryIdx].pt);
                pt2.emplace_back(kp2[m.trainIdx].pt);
            }
        }
    }


    vector<Point3f> ref3dCoords, color3d;

    getColors(im1, pt1, color3d);
    colors = color3d;

    Mat P1 = Mat::zeros(3,4, CV_64F);
    Mat P2 = Mat::zeros(3,4, CV_64F);
    P1.at<double>(0,0) = 1; P1.at<double>(1,1) = 1; P1.at<double>(2,2) = 1;
    P2.at<double>(0,0) = 1; P2.at<double>(1,1) = 1; P2.at<double>(2,2) = 1;
    P2.at<double>(0,3) = -baseline;

    P1 = K*P1;
    P2 = K*P2;

    Mat est3d;
    triangulatePoints(P1, P2, pt1, pt2, est3d);

    for(int i=0; i<est3d.cols; i++){
        Point3f localpt;
        localpt.x = est3d.at<float>(0,i) / est3d.at<float>(3,i);
        localpt.y = est3d.at<float>(1,i) / est3d.at<float>(3,i);
        localpt.z = est3d.at<float>(2,i) / est3d.at<float>(3,i);
        ref3dCoords.emplace_back(localpt);
    }
    ref3dPts = ref3dCoords;
    ref2dPts = pt1;
}

void visualSLAM::checkLoopDetectorStatus(Mat img, int idx){
    Ptr<FeatureDetector> orb = ORB::create();
    vector<KeyPoint> kp;
    Mat desc;
    vector<FORB::TDescriptor> descriptors;

    orb->detectAndCompute(img, Mat(), kp, desc);
    restructure(desc, descriptors);
    DetectionResult result;
    loopDetector->detectLoop(kp, descriptors, result);
    if(result.detection() &&(result.query-result.match > 100) && cooldownTimer==0){
        cerr<<"Found Loop Closure between "<<idx<<" and "<<result.match<<endl;
        LC_FLAG = true;
        LCidx = result.match-1;
        cooldownTimer = 200;
    }
}


void visualSLAM::PyrLKtrackFrame2Frame(Mat refimg, Mat curImg, vector<Point2f>refPts, vector<Point3f>ref3dpts,
                                    vector<Point2f>&refRetpts, vector<Point3f>&ref3dretPts){
    vector<Point2f> trackPts;
    vector<uchar> Idx;
    vector<float> err;

    calcOpticalFlowPyrLK(refimg, curImg, refPts, trackPts,Idx, err);

    vector<Point2f> inlierRefPts;
    vector<Point3f> inlierRef3dPts;
    vector<Point2f> inlierTracked;
    vector<int> res;

    for(int j=0; j<refPts.size(); j++){
        if(Idx[j]==1){
            inlierRefPts.push_back(refPts[j]);
            ref3dretPts.push_back(ref3dpts[j]);
            refRetpts.push_back(trackPts[j]);
        }
    }
    //refRetpts = inlierTracked;
    //ref3dretPts = inlierRef3dPts;
    inlierReferencePyrLKPts = inlierRefPts;
}

void visualSLAM::insertKeyFrames(int start, Mat imL, Mat imR, Mat&pose4dTransform, vector<Point2f>&ftrPts, vector<Point3f>&ref3dCoords){
    vector<Point2f> new2d;
    vector<Point3f> new3d;
    
    ftrPts.clear();
    ref3dCoords.clear();

    stereoTriangulate(imL, imR, new3d, new2d);

    untransformed = new3d;

    for(int i=0; i<new3d.size(); i++){
        Point3f pt = new3d[i];
        Point3f p;

        p.x = pose4dTransform.at<double>(0,0)*pt.x + pose4dTransform.at<double>(0,1)*pt.y + pose4dTransform.at<double>(0,2)*pt.z + pose4dTransform.at<double>(0,3);
        p.y = pose4dTransform.at<double>(1,0)*pt.x + pose4dTransform.at<double>(1,1)*pt.y + pose4dTransform.at<double>(1,2)*pt.z + pose4dTransform.at<double>(1,3);
        p.z = pose4dTransform.at<double>(2,0)*pt.x + pose4dTransform.at<double>(2,1)*pt.y + pose4dTransform.at<double>(2,2)*pt.z + pose4dTransform.at<double>(2,3);

        ref3dCoords.emplace_back(p);
        ftrPts.emplace_back(new2d[i]);
    }
}

vector<Point3f> visualSLAM::update3dtransformation(vector<Point3f>& pt3d, Mat& pose4dTransform){ 
    vector<Point3f> updateref3dCoords;
    for(int i=0; i<pt3d.size(); i++){
        Point3f pt = pt3d[i];
        Point3f p;

        p.x = pose4dTransform.at<double>(0,0)*pt.x + pose4dTransform.at<double>(0,1)*pt.y + pose4dTransform.at<double>(0,2)*pt.z + pose4dTransform.at<double>(0,3);
        p.y = pose4dTransform.at<double>(1,0)*pt.x + pose4dTransform.at<double>(1,1)*pt.y + pose4dTransform.at<double>(1,2)*pt.z + pose4dTransform.at<double>(1,3);
        p.z = pose4dTransform.at<double>(2,0)*pt.x + pose4dTransform.at<double>(2,1)*pt.y + pose4dTransform.at<double>(2,2)*pt.z + pose4dTransform.at<double>(2,3);

        updateref3dCoords.emplace_back(p);
    }
    return updateref3dCoords;
}

Mat visualSLAM::loadImageL(int iter){
    char FileName[200];
    sprintf(FileName, lFptr, iter);

    Mat im = imread(FileName);
    // cvtColor(im, im, CV_BGR2GRAY);
    // cvtColor(im, im, CV_GRAY2BGR);
    if(!im.data){
        cout<<"yikes, failed to fetch frame, check the paths"<<endl;
    }
    return im;
}
Mat visualSLAM::loadImageR(int iter){
    char FileName[200];
    sprintf(FileName, rFptr, iter);

    Mat im = imread(FileName);
    // cvtColor(im, im, CV_BGR2GRAY);
    // cvtColor(im, im, CV_GRAY2BGR);
    if(!im.data){
        cout<<"yikes, failed to fetch frame, check the paths"<<endl;
    }
    return im;
}

void visualSLAM::PerspectiveNpointEstimation(Mat&prevImg, Mat&curImg, vector<Point2f>&ref2dPoints, vector<Point3f>&ref3dPoints, 
                                vector<Point2f>&tracked2dPoints, vector<Point3f>&tracked3dPoints, Mat&rvec, Mat&tvec,vector<int>&inliers){

    PyrLKtrackFrame2Frame(referenceImg, currentImage, ref2dPoints, ref3dPoints, tracked2dPoints, tracked3dPoints);
    
    Mat distCoeffs = Mat::zeros(4,1,CV_64F);

    solvePnPRansac(tracked3dPoints, tracked2dPoints, K, distCoeffs, rvec, tvec, false,100, 1.0, 0.99, inliers);
    //cerr<<"Inlier Size : "<<inliers.size()<<endl;
    if(inliers.size()<10){
        cout<<"Low inlier count at "<<inliers.size()<<", trying again with increased reprojection Threshold "<<endl;
        inliers.clear();
        solvePnPRansac(tracked3dPoints, tracked2dPoints, K, distCoeffs, rvec, tvec, false,100, 8.0, 0.98, inliers);
        if(inliers.size()<10){
            cerr<<"Man, some incredibly shitty tracking out here, gotta exit bruh"<<endl;
            SHUTDOWN_FLAG = true;
        }
    }
}

void visualSLAM::initSequence(){
    int iter = 0;
    char FileName1[200], filename2[200];
    sprintf(FileName1, lFptr, iter);
    sprintf(filename2, rFptr, iter);

    // Mat imL = imread(FileName1);
    // Mat imR = imread(filename2);

    Mat imL = loadImageL(iter);
    Mat imR = loadImageR(iter);

    referenceImg = imL;

    vector<Point2f> ref2dFeatures;
    vector<Point3f> ref3dCoords;
    stereoTriangulate(imL, imR, ref3dCoords, ref2dFeatures);
    poseGraph.initializeGraph();
    Mat R = Mat::zeros(3,3,CV_64F);
    R.at<double>(0,0) = 1.0; R.at<double>(1,1) = 1.0; R.at<double>(2,2) = 1.0;

    keyFrame kf; kf.idx = 0; kf.ref3dCoords = ref3dCoords; kf.R = R; kf.t = Mat::zeros(1,3,CV_64F);
    keyFrameHistory.reserve(4500);
    keyFrameHistory.emplace_back(kf);
    
    cerr<<"\n\n"<<endl;

    for(int iter=1; iter<3270; iter++){
        //cout<<"PROCESSING FRAME "<<iter<<endl;
        currentImage = loadImageL(iter);
        
        vector<Point3f> trked3dCoords; vector<Point2f> trked2dPts;
        Mat tvec,rvec;
        vector<int> inliers;

        PerspectiveNpointEstimation(referenceImg, currentImage, ref2dFeatures, ref3dCoords, trked2dPts, trked3dCoords,rvec, tvec, inliers);
        if(SHUTDOWN_FLAG){
            break;
        }

        checkLoopDetectorStatus(currentImage,iter);

        Mat R;
        Rodrigues(rvec, R);

        R = R.t();
        Mat t = -R*tvec;

        if(LC_FLAG){
            stageForPGO(R, t, R, t, true);
            stageForPGO(R, t, R, t, false);
            std::vector<Eigen::Isometry3d> trans = poseGraph.globalOptimize();
            Mat interT = Eigen2cvMat(trans[trans.size()-1]);
            t = interT.t();
            updateOdometry(trans);
        }
        else{
            stageForPGO(R, t, R, t, false);
        }

        Mat pose4dTransform = Mat::zeros(3,4, CV_64F);
        R.col(0).copyTo(pose4dTransform.col(0));
        R.col(1).copyTo(pose4dTransform.col(1));
        R.col(2).copyTo(pose4dTransform.col(2));
        t.copyTo(pose4dTransform.col(3));
        
        bool reloc = false;
        if(inliers.size()<200 or LC_FLAG==true){
            cerr<<"ENTERING KEYFRAME AT "<<iter<<"... "<<"\r";
            Mat i1 = loadImageL(iter); Mat i2 = loadImageR(iter);
            insertKeyFrames(0, i1, i2, pose4dTransform, ref2dFeatures, ref3dCoords);

            vector<Point3f> good3d = ref3dCoords;
            vector<Point3f> goodColors = colors;

            SORcloud(good3d, goodColors);

            mapHistory.emplace_back(good3d);
            colorHistory.emplace_back(goodColors);
            Mat tcl = t.clone();
            trajectory.emplace_back(tcl);
            reloc = true;
        }
        else{
            ref3dCoords = trked3dCoords;
            ref2dFeatures = trked2dPts;
        }

        if(cooldownTimer!=0){
            cooldownTimer--;
        }
        referenceImg = currentImage;
        LC_FLAG = false;

        SORcloud(untransformed, colors);

        keyFrame kf;
        kf.idx = iter;
        kf.R = R;
        kf.t = t;
        kf.ref3dCoords = untransformed;
        if(reloc){
            kf.retrack = true;
        }
        else{
            kf.retrack = false;
        }
        
        keyFrameHistory.emplace_back(kf);
        Mat tr = t.clone();
        Mat Rr = R.clone();

        rosPublish(mapHistory, tr,Rr);



        t.convertTo(t, CV_32F);
        Mat frame = drawDeltas(currentImage, inlierReferencePyrLKPts, trked2dPts);

        imshow("frame", frame);
        ros::spinOnce();
        int k = waitKey(1);
        if (k=='q'){
            imwrite("trajectoryUnopt.png", canvas);
            break;
        }
    }
    cerr<<"Total map size :"<<mapPts.size()<<endl;
    poseGraph.saveStructure();
    vector<Eigen::Isometry3d> res = poseGraph.globalOptimize();
    updateOdometry(res);

    SHUTDOWN_FLAG = true;
    rosPublish(mapHistory, trajectory[trajectory.size()-1], keyFrameHistory[keyFrameHistory.size()-1].R);
    imwrite("Trajectory.png",canvas);
    cerr<<"Trajectory Saved"<<endl;
}



void visualSLAM::SORcloud(vector<Point3f>&ref3d, vector<Point3f>&colorMap){
    cloudType::Ptr bufferCloud (new cloudType);
    for(size_t i=0; i<ref3d.size(); i+=1){
        if(-1*ref3d[i].z>500){
            continue;
        }
        pcl::PointXYZRGB clPt;
        clPt.x = ref3d[i].x; clPt.y = ref3d[i].y; clPt.z = ref3d[i].z;
        clPt.r = colorMap[i].z; clPt.g = colorMap[i].y; clPt.b = colorMap[i].x;
        //msg->points.emplace_back(clPt);
        bufferCloud->points.emplace_back(clPt);
    }
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(bufferCloud);
    sor.setMeanK(200);
    sor.setStddevMulThresh(0.1);
    sor.filter(*bufferCloud);

    ref3d.clear(); colorMap.clear();
    for(size_t i=0; i<bufferCloud->points.size(); i++){
        pcl::PointXYZRGB cldPt;
        Point3f p3d, c3d;

        cldPt = bufferCloud->points[i];

        p3d.x = cldPt.x; p3d.y = cldPt.y; p3d.z = cldPt.z;
        c3d.x = cldPt.b; c3d.y = cldPt.g; c3d.z = cldPt.r;

        ref3d.emplace_back(p3d);
        colorMap.emplace_back(c3d);
    }
}

void visualSLAM::rosPublish(vector<vector<Point3f>>&pt3d, Mat&trajROS, Mat&Rmat){
    //cerr<<"Publishing messages"<<endl;
    cloudType::Ptr msg (new cloudType);
    geometry_msgs::PoseStamped poseMsg;
    msg->header.frame_id = "map";
    double mulFactor = 0.1;

    int siz = 0;
    for(size_t k=0; k<pt3d.size(); k++){
        vector<Point3f> ref3dCoords = pt3d[k];
        vector<Point3f> colorMap = colorHistory[k];
        for(size_t i=0; i<ref3dCoords.size(); i+=1){
            if(-1*ref3dCoords[i].z>500){
                continue;
            }
            pcl::PointXYZRGB clPt;
            siz+=1;
            clPt.x = ref3dCoords[i].x * mulFactor; clPt.y = ref3dCoords[i].z *mulFactor; clPt.z = -1*ref3dCoords[i].y*mulFactor;
            clPt.r = colorMap[i].z; clPt.g = colorMap[i].y; clPt.b = colorMap[i].x;
            msg->points.emplace_back(clPt);
            }
        }
    if(SHUTDOWN_FLAG){
        cerr<<"SAVING POINTCLOUD as "<<plySavepath<<endl;
        pcl::io::savePLYFileBinary(plySavepath, *msg);
        cerr<<"DONE"<<endl;
    }
    mapPublisher.publish(msg);

    double Xpos, Ypos, Zpos;
    trajROS*=mulFactor;

    Xpos = trajROS.at<double>(0);
    Zpos = trajROS.at<double>(1)*-1;
    Ypos = trajROS.at<double>(2);
    
    Eigen::Quaterniond q;
    Rmat2Quat(Rmat, q);
    Eigen::Vector4d Qvector = q.coeffs();

    //cerr<<"Quat : "<<Qvector<<endl;

    poseMsg.header.frame_id = "map";
    poseMsg.pose.position.x = Xpos;
    poseMsg.pose.position.y = Ypos;
    poseMsg.pose.position.z = Zpos;

    poseMsg.pose.orientation.x = Qvector[0];
    poseMsg.pose.orientation.y = Qvector[2];
    poseMsg.pose.orientation.z = Qvector[1]*-1;
    poseMsg.pose.orientation.w = Qvector[3];

    posePublisher.publish(poseMsg);
    

    trajectoryMsg.header.frame_id = "map";
    trajectoryMsg.poses.emplace_back(poseMsg);
    trajectoryPublisher.publish(trajectoryMsg);
}


void visualSLAM::updateOdometry(vector<Eigen::Isometry3d>&T){
    cerr<<"\n\nUpdating global odometry measurements..."<<endl;
    trajectory.clear();
    trajectory.reserve(T.size());
    for(Eigen::Isometry3d &isoMatrix : T){
        Mat t = Eigen2cvMat(isoMatrix);
        trajectory.emplace_back(t.clone());
    }
    cerr<<"Updating global 3D map..."<<endl;
    mapHistory.clear();
    for(size_t j=0; j<keyFrameHistory.size(); j++){
        keyFrame kf = keyFrameHistory[j];
        Mat R,t;
        t = trajectory[j];
        R = kf.R;
        kf.t = t;
        t = t.t();

        Mat pose4dTransform = Mat::zeros(3,4, CV_64F);
        R.col(0).copyTo(pose4dTransform.col(0));
        R.col(1).copyTo(pose4dTransform.col(1));
        R.col(2).copyTo(pose4dTransform.col(2));
        t.copyTo(pose4dTransform.col(3));
        
        vector<Point3f> updatePts = update3dtransformation(kf.ref3dCoords, pose4dTransform);
        if(kf.retrack){
            mapHistory.emplace_back(updatePts);
        }
    }
    cerr<<"DONE; Trajectory size : "<<trajectory.size()<<" KeyFrame size : "<<keyFrameHistory.size()<<endl;
}
int main(int argc, char **argv){
    ros::init(argc, argv, "SLAM_node");

    const char* impathL = "/media/gautham/Seagate Backup Plus Drive/Datasets/ColorSeq/dataset/sequences/13/image_2/%0.6d.png";
    const char* impathR = "/media/gautham/Seagate Backup Plus Drive/Datasets/ColorSeq/dataset/sequences/13/image_3/%0.6d.png";
    std::string vocPath = "/home/gautham/Documents/Projects/LargeScaleMapping/orb_voc13.yml.gz";

    vector<Point2f> ref2d; vector<Point3f> ref3d;

    visualSLAM Vsl(0, impathL, impathR, vocPath);
    char FileName1[200], filename2[200];
    sprintf(FileName1, impathL, 0);
    sprintf(filename2, impathR, 0);

    Mat im1 = imread(FileName1);
    Mat im2 = imread(filename2);
    //VO.stereoTriangulate(im1, im2, ref3d, ref2d);
    //visualOdometry* VO = new visualOdometry(0, impathR, impathL);
    Vsl.initSequence();
    return 0;
}
