#include "../include/visualSLAM.h"

void visualSLAM::stageForPGO(Mat Rl, Mat tl, Mat Rg, Mat tg, bool loopClose){
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