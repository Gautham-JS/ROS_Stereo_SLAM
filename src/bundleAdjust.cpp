#include <iostream>
#include <vector>
#include <algorithm>
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



using namespace std;
using namespace cv;
using namespace g2o;

struct FrameMetedata{
    int ID = -1;
    vector<Point2f> inliers;
    vector<Point3f> projection;
    Mat K, rvec, tvec, R,t;
};

struct PoseDatas{
    float x = -1;
    float y = -1;
    float z = -1;
    float qa = -1;
    float qb = -1;
    float qc = -1;
    float qd = -1;
};

inline float SIGN(float x) { 
	return (x >= 0.0f) ? +1.0f : -1.0f; 
}

inline float NORM(float a, float b, float c, float d) { 
	return sqrt(a * a + b * b + c * c + d * d); 
}

// quaternion = [w, x, y, z]'
Mat mRot2Quat(const Mat& m) {
	float r11 = m.at<float>(0, 0);
	float r12 = m.at<float>(0, 1);
	float r13 = m.at<float>(0, 2);
	float r21 = m.at<float>(1, 0);
	float r22 = m.at<float>(1, 1);
	float r23 = m.at<float>(1, 2);
	float r31 = m.at<float>(2, 0);
	float r32 = m.at<float>(2, 1);
	float r33 = m.at<float>(2, 2);
	float q0 = (r11 + r22 + r33 + 1.0f) / 4.0f;
	float q1 = (r11 - r22 - r33 + 1.0f) / 4.0f;
	float q2 = (-r11 + r22 - r33 + 1.0f) / 4.0f;
	float q3 = (-r11 - r22 + r33 + 1.0f) / 4.0f;
	if (q0 < 0.0f) {
		q0 = 0.0f;
	}
	if (q1 < 0.0f) {
		q1 = 0.0f;
	}
	if (q2 < 0.0f) {
		q2 = 0.0f;
	}
	if (q3 < 0.0f) {
		q3 = 0.0f;
	}
	q0 = sqrt(q0);
	q1 = sqrt(q1);
	q2 = sqrt(q2);
	q3 = sqrt(q3);
	if (q0 >= q1 && q0 >= q2 && q0 >= q3) {
		q0 *= +1.0f;
		q1 *= SIGN(r32 - r23);
		q2 *= SIGN(r13 - r31);
		q3 *= SIGN(r21 - r12);
	}
	else if (q1 >= q0 && q1 >= q2 && q1 >= q3) {
		q0 *= SIGN(r32 - r23);
		q1 *= +1.0f;
		q2 *= SIGN(r21 + r12);
		q3 *= SIGN(r13 + r31);
	}
	else if (q2 >= q0 && q2 >= q1 && q2 >= q3) {
		q0 *= SIGN(r13 - r31);
		q1 *= SIGN(r21 + r12);
		q2 *= +1.0f;
		q3 *= SIGN(r32 + r23);
	}
	else if (q3 >= q0 && q3 >= q1 && q3 >= q2) {
		q0 *= SIGN(r21 - r12);
		q1 *= SIGN(r31 + r13);
		q2 *= SIGN(r32 + r23);
		q3 *= +1.0f;
	}
	else {
		printf("coding error\n");
	}
	float r = NORM(q0, q1, q2, q3);
	q0 /= r;
	q1 /= r;
	q2 /= r;
	q3 /= r;

	Mat res = (Mat_<float>(4, 1) << q0, q1, q2, q3);
	return res;
}

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
        vector<PoseDatas> poses;
        vector<Point2f> inlierReferencePyrLKPts;
        Mat canvas = Mat::zeros(600,600, CV_8UC3);

        Mat ret;

        std::queue<FrameMetedata> BAwindowQueue;

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

            // kp1 = denseKeypointExtractor(im1, 10);
            // vector<Point2f> pt1;
            // for(size_t i=0; i<kp1.size(); i++){
            //     pt1.emplace_back(kp1[i].pt);
            // }

            // vector<Point2f> pt2;
            // pyrLKtracking(im1, im2, pt1, pt2);
            // FmatThresholding(pt1, pt2);

            detector->detect(im1, kp1);
            detector->detect(im2, kp2);

            Mat desc1, desc2;
            detector->compute(im1, kp1, desc1);
            detector->compute(im2, kp2, desc2);

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
            int iter = 1;
            char FileName1[200], filename2[200];
            sprintf(FileName1, lFptr, iter);
            sprintf(filename2, rFptr, iter);
            //PoseDatas* pose = new PoseDatas;

            FrameMetedata* meta = new FrameMetedata;
            Mat prevR, prevt;
            // Mat imL = imread(FileName1);
            // Mat imR = imread(filename2);

            Mat imL = loadImageL(iter);
            Mat imR = loadImageR(iter);

            referenceImg = imL;

            vector<Point2f> features;
            vector<Point3f> pts3d;
            stereoTriangulate(imL, imR, pts3d, features);

            for(int iter=iter+1; iter<4000; iter++){
                cout<<"PROCESSING FRAME "<<iter<<endl;
                currentImage = loadImageL(iter);

                vector<Point3f> refPts3d; vector<Point2f> refFeatures;
                PyrLKtrackFrame2Frame(referenceImg, currentImage, features, pts3d, refFeatures, refPts3d, true);
                //cout<<"     ref features "<<refPts3d.size()<<" refFeature size "<<refFeatures.size()<<endl;
                
                Mat distCoeffs = Mat::zeros(4,1,CV_64F);
                Mat rvec, tvec; vector<int> inliers;

                //cout<<refPts3d.size()<<endl;
                //cout<<refFeatures.size()<<" "<<inlierReferencePyrLKPts.size()<<" "<<refPts3d.size()<<endl;

                solvePnPRansac(refPts3d, refFeatures, K, distCoeffs, rvec, tvec, false,100,4.0, 0.99, inliers);
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
                meta->ID = iter;
                meta->inliers = refFeatures;
                meta->K = K;
                meta->projection = refPts3d;
                meta->rvec = rvec;
                meta->R = R;
                //BundleAdjust3d2d(refFeatures, refPts3d, K, Rba, tba);

                R = R.t();
                Mat t = -R*tvec;

                Mat quat; quat = mRot2Quat(R);
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


                Mat i1 = loadImageL(iter); Mat i2 = loadImageR(iter);

                relocalizeFrames(0, i1, i2, inv_transform, features, pts3d);

                vector<Point3f> test3d;

                //MonocularTriangulate(currentImage, referenceImg, refFeatures, inlierReferencePyrLKPts, test3d);

                referenceImg = currentImage;


                t.convertTo(t, CV_32F);
                tba.convertTo(tba, CV_32F);

                Mat frame = drawDeltas(currentImage, inlierReferencePyrLKPts, refFeatures);

                Point2f center = Point2f(int(t.at<float>(0)) + 300, int(t.at<float>(2)) + 100);
                //Point2f centerBA = Point2f(int(tba.at<float>(0)) + 300, int(tba.at<float>(2)) + 100);
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
        optimizer.optimize(10);

        //cout<<"T after="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
        SE3Quat postpose = pose->estimate();
        Eigen::Vector3d trans = postpose.translation();
        eigen2cv(trans, t);
        optimizer.clear();
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
