/*
GauthWare, LSM, 01/2021

buncha utility functions to clean up base CXX files
*/
#ifndef UTILS_H
#define UTILS_H

#include "monoOdometry.h"
#include "poseGraph.h"
#include "DloopDet.h"

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

inline float SIGN(float x) { 
	return (x >= 0.0f) ? +1.0f : -1.0f; 
}

inline float NORM(float a, float b, float c, float d) { 
	return sqrt(a * a + b * b + c * c + d * d); 
}

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


#endif