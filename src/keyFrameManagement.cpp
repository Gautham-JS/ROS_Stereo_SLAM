/*
GAUTHAM-JS , FEB-2021;
gauthamjs56@gmail.com
PART OF ROS STERO SLAM, UNDER MIT LICENSE.
*/

#include "../include/visualSLAM.h"

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
    
    vector<Point2f> trkUntr; vector<Point3f> trk3dUntr;
    PyrLKtrackFrame2Frame(referenceImg, currentImage, ref2dPoints, ref3dPoints, tracked2dPoints, tracked3dPoints);

    //cerr<<"Ref 2d "<<ref2dPoints.size()<<" untrans "<<untransformed.size()<<endl;
    //PyrLKtrackFrame2Frame(referenceImg, currentImage, ref2dPoints, untransformed, trkUntr, trk3dUntr);
    
    Mat distCoeffs = Mat::zeros(4,1,CV_64F);

    solvePnPRansac(tracked3dPoints, tracked2dPoints, K, distCoeffs, rvec, tvec, false,100, 1.0, 0.99, inliers);
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