#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include "Eigen/Core"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/eigen.hpp>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/edge_xyz_prior.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/vertex_pointxyz.h"
#include "g2o/types/slam3d/edge_se3_pointxyz.h"
//#include "edge_se3exp_pointxyz_prior.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"


// typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> SlamBlockSolver;
// typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

typedef g2o::BlockSolver_6_3 SlamBlockSolver; 
typedef g2o::LinearSolverEigen< SlamBlockSolver::PoseMatrixType > SlamLinearSolver;

using namespace std;
using namespace cv;
using namespace g2o;

struct FrameMetedata{
    int ID = -1;
    vector<Point2f> inliers;
    vector<Point3f> projection;
    Mat K, rvec, tvec, R,t;
};

struct FrameInfo{
	std::vector<cv::Point2f> features2d;
	std::vector<int> obj_indexes;
	// std::vector<cv::Point3f> features3d;
	cv::Mat rvec;
	cv::Mat tvec;
};

vector<KeyPoint> denseKeypointExtractor(Mat img, int stepSize){
    vector<KeyPoint> out;
    for (int y=stepSize; y<img.rows-stepSize; y+=stepSize){
        for (int x=stepSize; x<img.cols-stepSize; x+=stepSize){
            out.push_back(KeyPoint(float(x), float(y), float(stepSize)));
        }
    }
    return out;
}

void appendData(vector<float> data){
    std::ofstream outfile;
    outfile.open("/home/gautham/Documents/Projects/LargeScaleMapping/visualOdometry/trajectory.txt");
    for(size_t i=0; i<data.size(); i+=7){
        for(size_t j=0; j<7; j++){
            outfile<<data[j];
            outfile<<" ";
        }
        outfile<<"\n"; 
    }
}


void pyrLKtracking(Mat refImg, Mat curImg, vector<Point2f>&refPts, vector<Point2f>&trackPts){
    vector<Point2f> trPts, inlierRefPts, inlierTracked;
    vector<uchar> Idx;
    vector<float> err;
    calcOpticalFlowPyrLK(refImg, curImg, refPts, trPts,Idx, err);

    for(int i=0; i<refPts.size(); i++){
        if(Idx[i]==1){
            inlierRefPts.emplace_back(refPts[i]);
            inlierTracked.emplace_back(trPts[i]);
        }
    }
    trackPts.clear(); refPts.clear();
    trackPts = inlierTracked; refPts = inlierRefPts;
}

void FmatThresholding(vector<Point2f>&refPts, vector<Point2f>&trkPts){
    Mat F;
    vector<uchar> mask;
    vector<Point2f>inlierRef, inlierTrk;
    F = findFundamentalMat(refPts, trkPts, CV_RANSAC, 3.0, 0.99, mask);
    for(size_t j=0; j<mask.size(); j++){
        if(mask[j]==1){
            inlierRef.emplace_back(refPts[j]);
            inlierTrk.emplace_back(trkPts[j]);
        }
    }
    refPts.clear(); trkPts.clear();
    refPts = inlierRef; trkPts = inlierTrk;
}

void FmatThresholding2(vector<Point2f>&refPts, vector<Point2f>&trkPts, vector<Point3f>&ref3d){
    Mat F;
    vector<uchar> mask;
    vector<Point2f>inlierRef, inlierTrk; vector<Point3f> inlier3d;
    F = findFundamentalMat(refPts, trkPts, CV_RANSAC, 3.0, 0.99, mask);
    for(size_t j=0; j<mask.size(); j++){
        if(mask[j]==1){
            inlierRef.emplace_back(refPts[j]);
            inlierTrk.emplace_back(trkPts[j]);
            inlier3d.emplace_back(ref3d[j]);
        }
    }
    refPts.clear(); trkPts.clear(); ref3d.clear();
    refPts = inlierRef; trkPts = inlierTrk; ref3d = inlier3d;
}

cv::Mat Merge( const cv::Mat& rvec, const cv::Mat& tvec ){

	cv::Mat R;
    cv::Rodrigues( rvec, R );
    cv::Mat T = cv::Mat::zeros(3,4, CV_64F);
    T.at<double>(0,0) = R.at<double>(0,0);
    T.at<double>(0,1) = R.at<double>(0,1);
    T.at<double>(0,2) = R.at<double>(0,2);
    T.at<double>(1,0) = R.at<double>(1,0);
    T.at<double>(1,1) = R.at<double>(1,1);
    T.at<double>(1,2) = R.at<double>(1,2);
    T.at<double>(2,0) = R.at<double>(2,0);
    T.at<double>(2,1) = R.at<double>(2,1);
    T.at<double>(2,2) = R.at<double>(2,2);

    T.at<double>(0,3) = tvec.at<double>(0);
    T.at<double>(1,3) = tvec.at<double>(1);
    T.at<double>(2,3) = tvec.at<double>(2);

    // cv::hconcat(R, tvec, T);
    // std::cout << T << std::endl;
    return T;

}

// cvMat2Eigen
Eigen::Isometry3d cvMat2Eigen( const cv::Mat& rvec, const cv::Mat& tvec ){

	cv::Mat R;
    cv::Rodrigues( rvec, R );
    Eigen::Matrix3d r;
    for ( int i=0; i<3; i++ )
        for ( int j=0; j<3; j++ ) 
            r(i,j) = R.at<double>(i,j);
  
    // 将平移向量和旋转矩阵转换成变换矩阵
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
    T = angle;
    T(0,3) = tvec.at<double>(0,0); 
    T(1,3) = tvec.at<double>(1,0); 
    T(2,3) = tvec.at<double>(2,0);

    return T;

}

//将eigen 矩阵转换为opencv矩阵
cv::Mat Eigen2cvMat(const Eigen::Isometry3d& matrix) {

	cv::Mat R = cv::Mat::zeros(3,3,CV_64F);
	cv::Mat tvec = cv::Mat::zeros(1,3,CV_64F);

    for ( int i=0; i<3; i++ )
    for ( int j=0; j<3; j++ ) 
        R.at<double>(i,j) = matrix(i,j);

    tvec.at<double>(0) = matrix(0, 3); 
    tvec.at<double>(1) = matrix(1, 3);  
    tvec.at<double>(2) = matrix(2, 3);

    tvec = -R.t()*tvec.t(); 

    return tvec;

}

//欧拉 旋转 转换为 四元角 表达
g2o::SE3Quat euler2Quaternion(const cv::Mat& rvec, const cv::Mat& tvec )
{

	// std::cout << rvec.size() << std::endl;
	// std::cout << tvec.size() << std::endl;

	cv::Mat R;
    cv::Rodrigues( rvec, R );

	double roll = rvec.at<double>(0,0);
	double pitch = rvec.at<double>(1,0);
	double yaw = rvec.at<double>(2,0);

    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd yawAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = rollAngle * yawAngle * pitchAngle;

    Eigen::Vector3d trans(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    g2o::SE3Quat pose(q,trans);
    // std::cout << pose << std::endl;
    // std::cout << trans << std::endl;

    // assert(false);

    return pose;
}

void print(const std::vector<FrameInfo>& frameinfo, g2o::SparseOptimizer& optimizer) {


	for ( size_t i=0; i < frameinfo.size(); i++ )
    {

    	// if ( i < frameinfo.size() - 1) continue;

        g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*> (optimizer.vertex(i));
        // std::cout<<"vertex id "<<i<<", pos = " << std::endl;
        Eigen::Isometry3d pose = v->estimate();
        cv::Mat tvec = Eigen2cvMat(pose);
        // std::cout<<pose.matrix()<<std::endl;
        // g2o::SE3Quat pose2 = euler2Quaternion(frameinfo[i].rvec, frameinfo[i].tvec);
        // std::cout<<pose2<<std::endl;
        std::cout << "optimized: " << std::endl << tvec << std::endl;

       	cv::Mat R;
    	cv::Rodrigues( frameinfo[i].rvec, R );
        std::cout<< "original: "<<std::endl << -R.t()*frameinfo[i].tvec <<std::endl;
    }
}

class visualOdometry{
    public:
        int seqNo;
        double baseline = 0.54;

        string absPath;
        const char* lFptr; const char* rFptr;

        double focal_x = 7.188560000000e+02;
        double cx = 6.071928000000e+02;
        double focal_y = 7.188560000000e+02;
        double cy = 1.852157000000e+02;
        
        Mat K = (Mat1d(3,3) << focal_x, 0, cx, 0, focal_y, cy, 0, 0, 1);
        
        Mat referenceImg, currentImage;
        vector<Point3f> referencePoints3D;
        vector<Point2f> referencePoints2D;
        vector<Point2f> inlierReferencePyrLKPts;
        Mat canvas = Mat::zeros(600,600, CV_8UC3);
        Mat Rba, tba;

        Mat ret;

        std::queue<FrameMetedata> BAwindowQueue;
        mutable std::vector<cv::Point3f> world_landmarks;
	    mutable std::vector<cv::Point3f> history_pose;
	    mutable std::vector<FrameInfo> framepoints;

        visualOdometry(int Seq, const char*Lfptr, const char*Rfptr){
            cout<<"\n\nINITIALIZING VISUAL ODOMETRY PIPELINE...\n\n"<<endl;
            lFptr = Lfptr;
            rFptr = Rfptr;
        }

        ~visualOdometry(){
            cout<<"\n\nDESTRUCTOR CALLED, TERMINATING PROCESS\n"<<endl;
        }

        void stereoTriangulate(Mat im1, Mat im2, 
                            vector<Point3f>&ref3dPts, 
                            vector<Point2f>&ref2dPts){
            
            Ptr<FeatureDetector> detector = xfeatures2d::SURF::create(500);
            //Ptr<FeatureDetector> detector = BRISK::create();
            if(!im1.data || !im2.data){
                cout<<"NULL IMG"<<endl;
                return;
            }

            vector<KeyPoint> kp1, kp2;
            Mat desc1, desc2;

            // kp1 = denseKeypointExtractor(im1, 10);
            // vector<Point2f> pt1;
            // for(size_t i=0; i<kp1.size(); i++){
            //     pt1.emplace_back(kp1[i].pt);
            // }

            // vector<Point2f> pt2;
            // pyrLKtracking(im1, im2, pt1, pt2);
            // FmatThresholding(pt1, pt2);

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

            Mat E, mask, R, t;
            E = findEssentialMat(pt1, pt2, K, 8, 0.99, 1, mask);
            recoverPose(E, pt1, pt2, K, R, t,mask);

            vector<Point3f> pts3d;

            Mat P1 = Mat::zeros(3,4, CV_64F);
            Mat P2 = Mat::zeros(3,4, CV_64F);
            P1.at<double>(0,0) = 1; P1.at<double>(1,1) = 1; P1.at<double>(2,2) = 1;
            P2.at<double>(0,0) = 1; P2.at<double>(1,1) = 1; P2.at<double>(2,2) = 1;
            R.col(0).copyTo(P2.col(0));
            R.col(1).copyTo(P2.col(1));
            R.col(2).copyTo(P2.col(2));
            t.copyTo(P2.col(3));

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
                reprojection.emplace_back(projection);
            }
            //cout<<reprojection.size()<<" PTSIZE "<<pt1.size()<<endl;
            ret = drawDeltas(im2, pt1, reprojection);

            ref3dPts = pts3d;
            ref2dPts = pt1;
        }

        Mat drawDeltas(Mat im, vector<Point2f> in1, vector<Point2f> in2){
            Mat frame;
            im.copyTo(frame);

            for(int i=0; i<in1.size(); i++){
                Point2f pt1 = in1[i];
                Point2f pt2 = in2[i];
                line(frame, pt1, pt2, Scalar(0,255,0),1);
                circle(frame, pt1, 5, Scalar(0,0,255));
                circle(frame, pt2, 5, Scalar(255,0,0));
            }
            return frame;
        }

        void PyrLKtrackFrame2Frame(Mat refimg, Mat curImg, vector<Point2f>refPts, vector<Point3f>ref3dpts,
                                            vector<Point2f>&refRetpts, vector<Point3f>&ref3dretPts, bool Fthresh){
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
                    inlierRefPts.emplace_back(refPts[j]);
                    ref3dretPts.emplace_back(ref3dpts[j]);
                    refRetpts.emplace_back(trackPts[j]);
                }
            }
            if(Fthresh){
                FmatThresholding2(inlierRefPts, refRetpts, ref3dretPts);
            }
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

                if(!inRange){res.emplace_back(i);}
            }
            return res;
        }

        void create_new_features(int start, const Mat& inv_transform, std::vector<Point2f>& featurePoints, std::vector<Point3f>& landmarks){

            if (featurePoints.size() != 0) {
                featurePoints.clear();
                landmarks.clear();
            }

            Mat curImage_L = loadImageL(start);
            Mat curImage_R = loadImageL(start-1);

            vector<Point3f>  landmark_3D_new;
            vector<Point2f>  reference_2D_new;

            //extract_keypoints_surf(curImage_L, curImage_R, landmark_3D_new, reference_2D_new);
            stereoTriangulate(curImage_L, curImage_R, landmark_3D_new, reference_2D_new);

            // // cout << inv_transform << endl;

            for (int k = 0; k < landmark_3D_new.size(); k++) {
            // 
                    const Point3f& pt = landmark_3D_new[k];

                    Point3f p;

                    p.x = inv_transform.at<double>(0, 0)*pt.x + inv_transform.at<double>(0, 1)*pt.y + inv_transform.at<double>(0, 2)*pt.z + inv_transform.at<double>(0, 3);
                    p.y = inv_transform.at<double>(1, 0)*pt.x + inv_transform.at<double>(1, 1)*pt.y + inv_transform.at<double>(1, 2)*pt.z + inv_transform.at<double>(1, 3);
                    p.z = inv_transform.at<double>(2, 0)*pt.x + inv_transform.at<double>(2, 1)*pt.y + inv_transform.at<double>(2, 2)*pt.z + inv_transform.at<double>(2, 3);

                    // cout << p << endl;
                    if (p.z > 0) {
                        landmarks.push_back(p);
                        featurePoints.push_back(reference_2D_new[k]);
                    }

            }

        }
        vector<int> removeDuplicate(const vector<Point2f>& baseFeaturePoints, const vector<Point2f>& newfeaturePoints, 
            const vector<int>& mask, int radius=10){	
            std::vector<int> res;
            for (int j = 0; j < newfeaturePoints.size(); j++) {
                const Point2f& p2 = newfeaturePoints[j];
                bool within_range = false;
                for (auto index : mask) {
                    const Point2f& p1 = baseFeaturePoints[index];
                    if (cv::norm(p1-p2) < radius) {
                        within_range = true;
                        break;
                    }
                }
                if (!within_range) res.push_back(j);
            }
            return res;
        }
        vector<Point2f> updateFrame(int i, const cv::Mat& inv_transform, const vector<Point2f>& featurePoints, 
                    const vector<int>&tracked, const vector<int>& inliers, const Mat& rvec, const Mat& tvec){
            vector<Point2f> new_2D;
            vector<Point3f> new_3D;

            create_new_features(i, inv_transform, new_2D, new_3D);

            std::vector<Point2f> up_featurePoints;

            const std::vector<int>& preIndexes = framepoints.back().obj_indexes;
            vector<int> res = removeDuplicate(featurePoints, new_2D, inliers);
            cout << res.size() << ": " << new_2D.size() << endl;
                std::vector<int> indexes;
                for (auto index : inliers) {
                    up_featurePoints.push_back(featurePoints[index]);
                    indexes.push_back(preIndexes[tracked[index]]);
                }

                int start = world_landmarks.size();

                for (auto index : res) {
                    up_featurePoints.push_back(new_2D[index]);
                    world_landmarks.push_back(new_3D[index]);
                    indexes.push_back(start++);
                }

                ///for check correctness 
                // for (int k = 0; k < landmarks.size(); k++) {
                // 	if (landmarks[k] != world_landmarks[indexes[k]])
                // 	throw std::invalid_argument("These two landmarks should be the same!");
                // }

                FrameInfo frameinfo;
                frameinfo.features2d = up_featurePoints;
                frameinfo.obj_indexes = indexes;
                // frameinfo.features3d = landmarks;
                frameinfo.rvec  = rvec;
                frameinfo.tvec  = tvec;
                framepoints.push_back(frameinfo);
                if (framepoints.size() > 1000) {
                    framepoints.erase(framepoints.begin());
                }
                return up_featurePoints;

            }
        vector<int> tracking(const cv::Mat& ref_img, const cv::Mat& curImg, std::vector<Point2f>& featurePoints, std::vector<Point3f>& landmarks) {
            vector<Point2f> nextPts;
            vector<uchar> status;
            vector<float> err;

            calcOpticalFlowPyrLK(ref_img, curImg, featurePoints, nextPts, status, err);

            std::vector<int> res;
            featurePoints.clear();
            // cout << featurePoints.size() << ", " << landmarks.size() << ", " << status.size() << endl;

            for (int  j = status.size()-1; j > -1; j--) {
                if (status[j] != 1) {
                    // featurePoints.erase(featurePoints.begin()+j);
                    landmarks.erase(landmarks.begin()+j);
            
                } else {
                    featurePoints.push_back(nextPts[j]);
                    res.push_back(j);
                }
            }
            std::reverse(res.begin(),res.end()); 
            std::reverse(featurePoints.begin(),featurePoints.end()); 

            return res;
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
            int iter = 1;
            char FileName1[200], filename2[200];
            sprintf(FileName1, lFptr, iter);
            sprintf(filename2, rFptr, iter);

            //PoseDatas* pose = new PoseDatas;
            vector<FrameMetedata*> window;
            Mat prevR, prevt;
            // Mat imL = imread(FileName1);
            // Mat imR = imread(filename2);

            Mat imL = loadImageL(iter);
            Mat imR = loadImageL(iter-1);

            referenceImg = imL;

            vector<Point2f> features;
            vector<Point3f> pts3d;
            stereoTriangulate(imL, imR, pts3d, features);

            for(int iter=iter+1; iter<4000; iter+=1){
                FrameMetedata* meta = new FrameMetedata;
                cout<<"PROCESSING FRAME "<<iter<<endl;
                currentImage = loadImageL(iter);

                vector<Point3f> refPts3d; vector<Point2f> refFeatures;
                PyrLKtrackFrame2Frame(referenceImg, currentImage, features, pts3d, refFeatures, refPts3d, true);
                //cout<<"     ref features "<<refPts3d.size()<<" refFeature size "<<refFeatures.size()<<endl;
                
                Mat distCoeffs = Mat::zeros(4,1,CV_64F);
                Mat rvec, tvec; vector<int> inliers;

                //cout<<refPts3d.size()<<endl;
                //cout<<refFeatures.size()<<" "<<inlierReferencePyrLKPts.size()<<" "<<refPts3d.size()<<endl;

                solvePnPRansac(refPts3d, refFeatures, K, distCoeffs, rvec, tvec, false,100, 1.0, 0.99, inliers);
                cout<<"Inlier Size : "<<inliers.size()<<endl;
                if(inliers.size()<20 or tvec.at<double>(0)>1000){
                    cout<<"\n\nEyo What the fuck, skipping RANSAC layer and retracking\n"<<endl;
                    PyrLKtrackFrame2Frame(referenceImg, currentImage, features, pts3d, refFeatures, refPts3d, false);
                    solvePnPRansac(refPts3d, refFeatures, K, distCoeffs, rvec, tvec, false,100,4.0, 0.99, inliers);
                    if(inliers.size()<10 or tvec.at<double>(0)>1000){
                        cout<<"\n\n Damn bro inliers do be less, Skipping RANSAC all together, BA recommended\n"<<endl;
                        solvePnP(refPts3d, refFeatures, K, distCoeffs, rvec, tvec);
                    }
                }

                if(inliers.size()<10){
                    cout<<"Low inlier count at "<<refFeatures.size()<<", skipping RANSAC and using PnP "<<iter<<endl;
                    solvePnP(refPts3d, refFeatures, K, distCoeffs, rvec, tvec, false);
                }
                Mat R;
                Rodrigues(rvec, R);

                Mat Rba, tba;
                R.copyTo(Rba); tvec.copyTo(tba);

                R = R.t();
                Mat t = -R*tvec;

                meta->ID = iter;
                meta->inliers = refFeatures;
                meta->K = K;
                meta->projection = refPts3d;
                meta->rvec = rvec;
                meta->R = R;
                window.emplace_back(meta);

                //BundleAdjust3d2d(features,pts3d, K, Rba, tba);
                //tba = -Rba*tba;

                //PoseDatas pose;
                // pose.x = t.at<float>(0);
                // pose.y = t.at<float>(1);
                // pose.z = t.at<float>(3);
                // pose.qa = quat.at<float>(0);
                // pose.qb = quat.at<float>(1);
                // pose.qc = quat.at<float>(2);
                // pose.qd = quat.at<float>(3);
                
                //poses.emplace_back(pose);

                prevR =  rvec; prevt = tvec;
                Rba = Rba.t();
                tba = -Rba*tba;

                Mat inv_transform = Mat::zeros(3,4, CV_64F);
                R.col(0).copyTo(inv_transform.col(0));
                R.col(1).copyTo(inv_transform.col(1));
                R.col(2).copyTo(inv_transform.col(2));
                t.copyTo(inv_transform.col(3));

                if(inliers.size()<200){
                    cerr<<"RELOCALIZING KeyFrame at "<<iter<<endl;
                    Mat i1 = loadImageL(iter); Mat i2 = loadImageR(iter);

                    relocalizeFrames(0, i1, i2, inv_transform, features, pts3d);
                }
                else{
                    features = refFeatures;
                    pts3d = refPts3d;
                }
                //create_new_features(iter, inv_transform, features, pts3d);

                vector<Point3f> test3d;

                //MonocularTriangulate(currentImage, referenceImg, refFeatures, inlierReferencePyrLKPts, test3d);

                referenceImg = currentImage;


                t.convertTo(t, CV_32F);
                tba.convertTo(tba, CV_32F);

                Mat frame = drawDeltas(currentImage, inlierReferencePyrLKPts, refFeatures);

                Point2f center = Point2f(int(t.at<float>(0)) + 300, int(t.at<float>(2)) + 100);
                Point2f centerBA = Point2f(int(tba.at<float>(0)) + 300, int(tba.at<float>(2)) + 100);
                circle(canvas, center ,1, Scalar(0,0,255), 1);
                //circle(canvas, centerBA ,1, Scalar(0,255,0), 1);
                rectangle(canvas, Point2f(10, 30), Point2f(550, 50),  Scalar(0,0,0), cv::FILLED);

                imshow("frame", frame);
                imshow("trajectory", canvas);
                int k = waitKey(100);
                if (k=='q'){
                    break;
                }
            }
            cerr<<"Trajectory Saved"<<endl;
            imwrite("Trajectory.png",canvas);
        }

/*-------------------------------------EXPERIMENTAL SHIT BEGINS HERE-------------------------------------------------------------*/
    void BundleAdjust3d2d(vector<Point2f>points_2d, vector<Point3f>points_3d, Mat&K, Mat&R, Mat&t){
        typedef BlockSolver<BlockSolverTraits<6,3>>block;   
        typedef LinearSolverDense<block::PoseMatrixType> linearSolver;

        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<block>(g2o::make_unique<linearSolver>())
        );

        SparseOptimizer optimizer;
        optimizer.initializeOptimization();
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(true);

        VertexSE3Expmap* pose = new VertexSE3Expmap();
        
        Eigen::Matrix3d Rmatrix;
        Rmatrix<<R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
               R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
               R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
        
        pose->setId(0);
        pose->setEstimate(SE3Quat(Rmatrix, Eigen::Vector3d(t.at<double>(0,0), t.at<double>(1,0), t.at<double>(2,0))));
        
        int count=1;
        optimizer.addVertex(pose);
        for(const Point3f p: points_3d){
            VertexSBAPointXYZ* point = new VertexSBAPointXYZ();
            point->setId(count);
            point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
            point->setMarginalized(true);
            optimizer.addVertex(point);
            count++;
        }

        CameraParameters* cam = new CameraParameters(
            K.at<double>(0,0), Eigen::Vector2d(K.at<double>(0,2), K.at<double>(1,2)), 0
        );

        cam->setId(0);
        optimizer.addParameter(cam);
        int edgeCount = 1;
        for(const Point2f pt: points_2d){
            EdgeProjectXYZ2UV* edge = new EdgeProjectXYZ2UV();
            edge->setId(edgeCount);
            edge->setVertex(0,  dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex(edgeCount)));
            edge->setVertex(1, pose);
            edge->setMeasurement(Eigen::Vector2d(pt.x, pt.y));
            edge->setParameterId(0,0);
            edge->setInformation(Eigen::Matrix2d::Identity());
            optimizer.addEdge(edge);
            edgeCount++;
        }
        //cout<<"T before="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
        optimizer.initializeOptimization();
        optimizer.optimize(15);

        //cout<<"T after="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
        SE3Quat postpose = pose->estimate();
        Eigen::Vector3d trans = postpose.translation();
        eigen2cv(trans, t);
        optimizer.clear();
    }
    void playSequence(){
        int startIndex = 1;

        vector<Point3f> location_history;

        Mat left_img = loadImageL(startIndex);
        Mat right_img = loadImageL(startIndex-1);
        Mat& ref_img  = left_img;

        // vector<Point3f> landmarks;
        vector<Point2f> featurePoints;
        //vector<int> landmarks;

        bool visualize = true;
        //bool bundle = true;
        // int optimized_frame = ;

        //extract_keypoints_surf(left_img, right_img, landmarks, featurePoints);
        //extract_keypoints_surf(left_img, right_img, world_landmarks, featurePoints);
        stereoTriangulate(left_img, right_img, world_landmarks,featurePoints );
        // cout << featurePoints[0] << endl;
        // cout << landmarks[0] << endl;

        // vector<FrameInfo> framepoints;
        //initilize the first frame with the below parameters
        {

            FrameInfo frameinfo;
            frameinfo.features2d = featurePoints;
            for (int i = 0; i < featurePoints.size(); i++) frameinfo.obj_indexes.push_back(i);
            // frameinfo.features3d = landmarks;
            // world_landmarks = landmarks;
            frameinfo.rvec  = Mat::zeros(3,1,CV_64F);
            frameinfo.tvec  = Mat::zeros(3,1,CV_64F);
            framepoints.push_back(frameinfo);
        }
        int start = 0;

        Mat curImg;
        bool bundle=true;

        Mat traj = Mat::zeros(600, 600, CV_8UC3);

        for(int i = startIndex + 1; i < 4000; i+=1) {

            cout << i << endl;

            curImg = loadImageL(i);

            //std::vector<Point3f>  landmarks_ref, landmarks;
            //std::vector<Point2f>  featurePoints_ref;
            std::vector<Point3f>  landmarks;

            featurePoints = framepoints.back().features2d;

            for (auto index : framepoints.back().obj_indexes) {
                landmarks.push_back(world_landmarks[index]);
            }

            vector<int> tracked = tracking(ref_img, curImg, featurePoints, landmarks);// landmarks_ref, featurePoints_ref);
            // cout << featurePoints.size() << ", " << landmarks.size() << endl;
            if (landmarks.size() < 10) continue;

            Mat dist_coeffs = Mat::zeros(4,1,CV_64F);

            Mat rvec, tvec;
            
            vector<int> inliers;
            // cout << featurePoints.size() << ", " << landmarks.size() << endl;
            solvePnPRansac(landmarks, featurePoints, K, dist_coeffs,rvec, tvec,false, 100, 8.0, 0.99, inliers);// inliers);

            if (inliers.size() < 5) continue;

            // cout << "Norm: " << normofTransform(rvec- framepoints.back().rvec, tvec - framepoints.back().tvec)  << endl;
            // if ( normofTransform(rvec- framepoints.back().rvec, tvec - framepoints.back().tvec) > 2 ) continue;

            // if (normofTransform(rvec, tvec) > 0.3) continue;

            float inliers_ratio = inliers.size()/float(landmarks.size());

            cout << "inliers ratio: " << inliers_ratio << endl;
            cerr<<i<<endl;

            Mat R_matrix;
            Rodrigues(rvec,R_matrix); 
            R_matrix = R_matrix.t();
            Mat t_vec = -R_matrix*tvec;

            // cout << t_vec << endl;
            Mat inv_transform = Mat::zeros(3,4,CV_64F);
            R_matrix.col(0).copyTo(inv_transform.col(0));
            R_matrix.col(1).copyTo(inv_transform.col(1));
            R_matrix.col(2).copyTo(inv_transform.col(2));
            t_vec.copyTo(inv_transform.col(3));


            featurePoints = updateFrame(i, inv_transform, featurePoints, tracked, inliers, rvec, tvec);

            if (featurePoints.size() == 0) continue;

            //featurePoints = up_featurePoints;
            ref_img = curImg;

            t_vec.convertTo(t_vec, CV_32F);

            if (bundle  && (framepoints.size() == 500 || i == 4000 - 1)) {
                cerr<<"BA time"<<endl;
                Point3f p3 = BundleAdjust2(framepoints, world_landmarks, location_history, K);
                framepoints.erase(framepoints.begin()+1, framepoints.end()-1);
                history_pose.push_back(p3);
            } else {
                history_pose.push_back(Point3f(t_vec.at<float>(0), t_vec.at<float>(1), t_vec.at<float>(2)));
            }
            //cout << t_vec.t() << endl;
            //cout << "truth" << endl;
            //cout << "["<<poses[i][3] << ", " << poses[i][7] << ", " << poses[i][11] <<"]"<<endl;

            if (visualize) {
                        // plot the information
                string text  = "Red color: estimated trajectory";
                string text2 = "Blue color: Groundtruth trajectory";
                // cout << framepoints.size() << endl; 
                // cout << t_vec.t() << endl;
                // cout << "["<<poses[i][3] << ", " << poses[i][7] << ", " << poses[i][11] <<"]"<<endl;
                Mat tdraw = t_vec.clone();
                tdraw*=0.4;
                Point2f center  = Point2f(int(tdraw.at<float>(0)) + 300, int(tdraw.at<float>(2)) + 100);
                //Point2f center2 = Point2f(int(poses[i][3]) + 300, int(poses[i][11]) + 100);

                circle(traj, center , 1, Scalar(0,0,255), 1);
                //circle(traj, center2, 1, Scalar(255,0,0), 1);
                rectangle(traj, Point2f(10, 30), Point2f(550, 50),  Scalar(0,0,0), cv::FILLED);
                putText(traj, text,  Point2f(10,50), cv::FONT_HERSHEY_PLAIN, 1, Scalar(0, 0,255), 1, 5);
                putText(traj, text2, Point2f(10,70), cv::FONT_HERSHEY_PLAIN, 1, Scalar(255,0,0), 1, 5);

                if (bundle) {
                        string text1 = "Green color: bundle adjusted trajectory";
                        for (const auto& p3 : location_history) {
                            int xc = int(p3.x)*0.4;
                            int zc = int(p3.z)*0.4;
                            Point2f center1 = Point2f(int(xc) + 300, int(zc) + 100);
                            circle(traj, center1, 1, Scalar(0,255,0), 1);
                        }
                        location_history.clear();
                        putText(traj, text1, Point2f(10,90), cv::FONT_HERSHEY_PLAIN, 1, Scalar(0,255,0), 1, 5);
                }

                imshow( "Trajectory", traj);
                waitKey(1);
            }

        }

        // if (bundle) {

        // 	// BundleAdjust2(framepoints, world_landmarks, history_pose, K);
        // }

        if (visualize) {

            // if (bundle) {
            // 	string text1 = "Green color: bundle adjusted trajectory";
            // 	putText(traj, text1, Point2f(10,90), cv::FONT_HERSHEY_PLAIN, 1, Scalar(0,255,0), 1, 5);
            // 	for (const auto& p3 : history_pose) {
            // 		Point2f center1 = Point2f(int(p3.x) + 300, int(p3.z) + 100);
            // 		circle(traj, center1, 1, Scalar(0,255,0), 1);
            // 	}
            // 	imshow( "Trajectory", traj);
            // }
            imwrite("map2.png", traj);
            waitKey(0);
        }

    }

    cv::Point3f BundleAdjust2(std::vector<FrameInfo>& frameinfo, std::vector<cv::Point3f>& world_points, std::vector<cv::Point3f>& history_poses, const cv::Mat& K) {
        
        // g2o::SparseOptimizer    optimizer;
    //    auto linearSolver = g2o::make_unique<SlamLinearSolver>();
    //    linearSolver->setBlockOrdering( false );
    //    // L-M 下降 
    //    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<SlamBlockSolver>(std::move(linearSolver)));
        
    //    optimizer.setAlgorithm( algorithm );
    //    optimizer.setVerbose( false );
        SlamLinearSolver* linearSolver = new SlamLinearSolver();
        linearSolver->setBlockOrdering( false );
        //SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
        
        // g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
        //     g2o::make_unique<g2o::BlockSolverX>(
        //         g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>()));
        g2o::OptimizationAlgorithm* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<SlamBlockSolver>(g2o::make_unique<SlamLinearSolver>())
        );

        g2o::SparseOptimizer optimizer;  // 最后用的就是这个东东
        optimizer.setAlgorithm( solver ); 
        // 不要输出调试信息
        optimizer.setVerbose( true );


        //camera information
        g2o::CameraParameters* camera = new g2o::CameraParameters(K.at<double>(0,0), Eigen::Vector2d(K.at<double>(0,2), K.at<double>(1,2)), 0 );
        camera->setId(0);
        optimizer.addParameter( camera);

        int size = frameinfo.size();

        //add vertext
        for ( int i=0; i< size; i++ )
        {
            g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
            v->setId(i);
            if ( i == 0) v->setFixed( true );
            // Eigen::Isometry3d T = cvMat2Eigen(frameinfo[i].rvec, frameinfo[i].tvec);
            g2o::SE3Quat pose = euler2Quaternion(frameinfo[i].rvec, frameinfo[i].tvec);
            v->setEstimate( pose );
            optimizer.addVertex( v );
        }

        int startIndex = size;

        std::cout <<"start Index: " << startIndex << std::endl;

        std::unordered_map<int, int> has_seen;
        int count = 0;

        for (int i = 0; i < size; i++) {
            
            const FrameInfo& frame = frameinfo[i];

            for (int j = 0; j < frame.obj_indexes.size(); j++) {	
                    int index = frame.obj_indexes[j];
                    int currentNodeIndex = startIndex;

                if (has_seen.find(index) == has_seen.end()) {	
                        //add the landmark
                    g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
                    v->setId(startIndex);
                    v->setMarginalized(true);
                    cv::Point3f w_point = world_points[index];
                    v->setEstimate(Eigen::Vector3d(w_point.x, w_point.y, w_point.z));
                    optimizer.addVertex( v );
                    has_seen[index] = startIndex;
                    startIndex++;
                }  else {

                    currentNodeIndex = has_seen[index];
                }

                //add the edges
                g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
                edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(currentNodeIndex)));
                edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>   (optimizer.vertex(i)));
                edge->setMeasurement( Eigen::Vector2d(frame.features2d[j].x, frame.features2d[j].y));
                edge->setInformation( Eigen::Matrix2d::Identity() );
                edge->setParameterId(0, 0);
                edge->setRobustKernel( new g2o::RobustKernelHuber() );
                optimizer.addEdge( edge );
            }
        }
    
        std::cout<<"optimizing pose graph, vertices: "<<optimizer.vertices().size()<<std::endl;
        optimizer.save("sba.g2o");
        optimizer.initializeOptimization();
        optimizer.optimize(20); //可以指定优化步数
        cerr<<"BA1"<<endl;

        // 以及所有特征点的位置
        // print(frameinfo, optimizer);
        g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*> (optimizer.vertex(size-1));
        Eigen::Isometry3d pose = v->estimate();
        cv::Mat tvec = Eigen2cvMat(pose);
        // std::cout << "tvec is : " << tvec.at<double>(0) << ", " << tvec.at<double>(1) << ", " << tvec.at<double>(2) << std::endl;
        // history_poses.push_back(cv::Point3f(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)));
        {

                                //         g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*> (optimizer.vertex(i));       
                    //         Eigen::Isometry3d pose = v->estimate();
                    //         cv::Mat tvec = Eigen2cvMat(pose);

        


                    for ( size_t i=2; i < frameinfo.size(); i++ )
                        {
                            g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*> (optimizer.vertex(i));       
                            Eigen::Isometry3d pose = v->estimate();
                            cv::Mat t_vec = Eigen2cvMat(pose);
                            //cv::Mat R;
                            //cv::Rodrigues( frameinfo[i].rvec, R );
                            //cv::Mat t_vec =  -R.t()*frameinfo[i].tvec;
                            history_poses.push_back(cv::Point3f(t_vec.at<double>(0), t_vec.at<double>(1), t_vec.at<double>(2)));
                        }

            // for (auto& item : has_seen) {
            // 	g2o::VertexSBAPointXYZ* v = dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(item.second));
            // 	Eigen::Vector3d pos = v->estimate();
            // 	world_points[item.first] = cv::Point3f(pos[0], pos[1], pos[2]);
            // }

        }
        optimizer.clear();
        std::cout<<"Optimization done."<<std::endl;

        return cv::Point3f(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
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
