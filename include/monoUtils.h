/*
GauthWare, LSM, 01/2021

buncha utility functions to clean up base CXX files
*/
#ifndef UTILS_H
#define UTILS_H

#include "poseGraph.h"
#include "DloopDet.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace cv;
using namespace g2o;
using namespace DLoopDetector;
using namespace DBoW2;



void appendData(vector<float> data){
    std::ofstream outfile;
    outfile.open("trajectory.csv", ios::app);
    for(size_t i=0; i<data.size(); i+=8){
        for(size_t j=i; j<i+8; j++){
            outfile<<data[j];
            outfile<<",";
        }
        outfile<<"\n"; 
    }
    outfile.close();
}


void createData(vector<float> data){
    std::ofstream outfile;
    outfile.open("trajectory.csv");
    outfile<<"Idx,Xm,Ym,Zm,Xgt,Ygt,Zgt,Const\n";
    for(size_t i=0; i<data.size(); i+=8){
        for(size_t j=i; j<i+8; j++){
            outfile<<data[j];
            outfile<<",";
        }
        outfile<<"\n"; 
    }
    outfile.close();
}

void dumpOptimized(vector<float> data){
    std::ofstream outfile;
    outfile.open("trajectoryOptimized.csv");
    outfile<<"Xo,Yo,Zo\n";
    for(size_t i=0; i<data.size(); i+=3){
        int coount = 0;
        for(size_t j=i; j<i+3; j++){
            outfile<<data[j];
            if(coount==2){
                continue;   
            }
            else{
                outfile<<",";
                coount++;
            }
        }
        outfile<<"\n"; 
    }
    outfile.close();
}

g2o::SE3Quat euler2Quaternion(const cv::Mat& R, const cv::Mat& tvec ){
	cv::Mat rvec;
    cv::Rodrigues( R, rvec );

	double roll = rvec.at<double>(0,0);
	double pitch = rvec.at<double>(1,0);
	double yaw = rvec.at<double>(2,0);

    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd yawAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(yaw, Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = rollAngle * yawAngle * pitchAngle;

    Eigen::Vector3d trans(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    g2o::SE3Quat pose(q,trans);
    return pose;
}

Eigen::Isometry3d cvMat2Eigen( const cv::Mat& R, const cv::Mat& tvec ){
    Eigen::Matrix3d r;
    for ( int i=0; i<3; i++ )
        for ( int j=0; j<3; j++ ) 
            r(i,j) = R.at<double>(i,j);
  
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
    T = angle;
    T(0,3) = tvec.at<double>(0,0); 
    T(1,3) = tvec.at<double>(1,0); 
    T(2,3) = tvec.at<double>(2,0);

    return T;
}

cv::Mat Eigen2cvMat(const Eigen::Isometry3d& matrix) {

	cv::Mat R = cv::Mat::zeros(3,3,CV_64F);
	cv::Mat tvec = cv::Mat::zeros(1,3,CV_64F);

    for ( int i=0; i<3; i++ )
    for ( int j=0; j<3; j++ ) 
        R.at<double>(i,j) = matrix(i,j);
    
    Eigen::Vector3d trans = matrix.translation();

    tvec.at<double>(0) = trans(0); 
    tvec.at<double>(1) = trans(1);  
    tvec.at<double>(2) = trans(2);

    //tvec = -R.t()*tvec.t(); //SUS af

    return tvec;
}


double getAbsoluteScale(int frame_id, double &Xpos, double &Ypos, double &Zpos){  
    string line;
    int i = 0;
    ifstream myfile ("/media/gautham/Seagate Backup Plus Drive/Datasets/dataset/poses/00.txt");
    double x =0, y=0, z = 0;
    double x_prev, y_prev, z_prev;
    if (myfile.is_open()){
        while (( getline (myfile,line) ) && (i<=frame_id)){
            z_prev = z;
            x_prev = x;
            y_prev = y;
            std::istringstream in(line);
            for (int j=0; j<12; j++)  {
                in >> z ;
                if (j==7) y=z;
                if (j==3)  x=z;
            }
            i++;
        }
        myfile.close();
    }

    else {
        cout << "Unable to open file";
        return 0;
    }
    Xpos = x_prev; Ypos = y_prev; Zpos = z_prev;
    return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;
}

Mat drawDeltas(Mat im, vector<Point2f> in1, vector<Point2f> in2){
    Mat frame;
    im.copyTo(frame);

    for(int i=0; i<in1.size(); i++){
        Point2f pt1 = in1[i];
        Point2f pt2 = in2[i];
        if(pt1.x <= 0  || pt1.y <= 0 || pt2.x <= 0 || pt2.x <= 0){
            cerr<<"Disaster averted"<<endl;
            continue;
        }
        //cerr<<pt1<<" "<<pt2<<endl;
        cv::line(frame, pt1, pt2, Scalar(0,255,0),2);
        circle(frame, pt1, 5, Scalar(0,0,255));
        circle(frame, pt2, 5, Scalar(255,0,0));
    }
    return frame;
}


void getColors(Mat& img, vector<Point2f> pts,vector<Point3f>&colorMap){
    colorMap.clear();
    colorMap.reserve(pts.size());
    for(size_t j=0; j<pts.size(); j++){
        int x = int(pts[j].x); int y = int(pts[j].y);
        Point3f color3d;
        Vec3b pixelColor = img.at<Vec3b>(y,x);

        color3d.x = pixelColor[0];
        color3d.y = pixelColor[1];
        color3d.z = pixelColor[2];
        colorMap.emplace_back(color3d);
    }
}

vector<int> removeDuplicates(vector<Point2f>&baseref2dFeatures, vector<Point2f>&newref2dFeatures,
                                    vector<int>&mask, int radius=10){
    vector<int> res;
    for(int i=0; i<newref2dFeatures.size(); i++){
        Point2f&p2 = newref2dFeatures[i];
        bool inRange=false;
        
        for(auto j:mask){
            Point2f&p1 = baseref2dFeatures[j];
            if(norm(p1-p2)<radius){
                inRange=true;
                break;
            }
        }

        if(!inRange){res.push_back(i);}
    }
    return res;
}

void Rmat2Quat(Mat&Rmat, Eigen::Quaterniond&quat){
    Mat Rvec; Rodrigues(Rmat, Rvec);
	double roll = Rvec.at<double>(0,0);
	double pitch = Rvec.at<double>(1,0);
	double yaw = Rvec.at<double>(2,0);

    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd yawAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(yaw, Eigen::Vector3d::UnitZ());
    quat = rollAngle * yawAngle * pitchAngle;

    //Eigen::Vector4d Qvec = q.coeffs();
}

#endif