/*
-->GauthWare, LSM, 01/2021
*/
// #include <iostream>
// #include <vector>
// #include <algorithm>

#include "../include/monoOdometry.h"
#include "../include/monoUtils.h"
#include "../include/poseGraph.h"

using namespace std;
using namespace cv;



Mat monoOdom::loadImage(int iter){
    char fileName[200];
    sprintf(fileName, lFptr, iter);
    Mat im = imread(fileName);
    if(!im.data){
        cerr<<"YIKES, failed to grab "<<iter<<" image.\nYou might wanna check that path again dawg."<<endl;
    }
    return im;
}

Mat monoOdom::drawDeltasErr(Mat im, vector<Point2f> in1, vector<Point2f> in2){
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

vector<KeyPoint> monoOdom::denseKeypointExtractor(Mat img, int stepSize){
    vector<KeyPoint> out;
    for (int y=stepSize; y<img.rows-stepSize; y+=stepSize){
        for (int x=stepSize; x<img.cols-stepSize; x+=stepSize){
            out.push_back(KeyPoint(float(x), float(y), float(stepSize)));
        }
    }
    return out;
}

void monoOdom::pyrLKtracking(Mat refImg, Mat curImg, vector<Point2f>&refPts, vector<Point2f>&trackPts){
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

void monoOdom::FmatThresholding(vector<Point2f>&refPts, vector<Point2f>&trkPts){
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

void monoOdom::monoTriangulate(Mat img1, Mat img2,vector<Point2f>&ref2dPts, vector<Point2f>&trk2dPts, vector<Point3f>&ref3dpts){
    // Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create(1000);
    // Ptr<FeatureDetector> detector = ORB::create(2000);
    vector<KeyPoint> dkps;
    dkps = denseKeypointExtractor(img1, 20);

    vector<Point2f> refPts;
    for(size_t i=0; i<dkps.size(); i++){
        refPts.emplace_back(dkps[i].pt);
    }

    vector<Point2f> trkPts;
    pyrLKtracking(img1, img2, refPts, trkPts);
    FmatThresholding(refPts, trkPts);
    // vector<KeyPoint> kp1, kp2;
    // detector->detect(img1, kp1);
    // detector->detect(img2, kp2);

    // Mat desc1, desc2;
    // detector->compute(img1, kp1, desc1);
    // detector->compute(img2, kp2, desc2);

    // desc1.convertTo(desc1, CV_32F);
    // desc2.convertTo(desc2, CV_32F);

    // BFMatcher matcher;
    // vector<vector<DMatch>> matches;
    // matcher.knnMatch(desc1, desc2, matches, 2);

    // vector<Point2f> pt1, pt2;
    // for(int i=0; i<matches.size(); i++){
    //     DMatch &m = matches[i][0]; DMatch &n = matches[i][1];
    //     if(m.distance<0.8*n.distance){
    //         pt1.emplace_back(kp1[m.queryIdx].pt);
    //         pt2.emplace_back(kp2[m.trainIdx].pt);
    //     }
    // }
    // FmatThresholding(pt1, pt2);
    // vector<Point2f> refPts, trkPts;
    // refPts = pt1; trkPts = pt2;

    Mat E, mask;
    E = findEssentialMat(refPts, trkPts, K, 8, 0.99, 1, mask);
    recoverPose(E, refPts, trkPts, K, R, t,mask);
    
    Mat pts4d;
    Mat P1 = Mat::zeros(3,4,CV_64F); Mat P2 = Mat::zeros(3,4,CV_64F);
    P1.at<double>(0,0) = 1; P1.at<double>(1,1) = 1; P1.at<double>(2,2) = 1;
    R.col(0).copyTo(P2.col(0));
    R.col(1).copyTo(P2.col(1));
    R.col(2).copyTo(P2.col(2));
    t.copyTo(P2.col(3));
    triangulatePoints(P1, P2, refPts, trkPts, pts4d);

    vector<Point3f> pts3d;

    pts3d.reserve(pts4d.cols);
    for(size_t j=0; j<pts4d.cols; j++){
        Point3f landmark;
        landmark.x = pts4d.at<double>(0,j)/pts4d.at<double>(3,j);
        landmark.y = pts4d.at<double>(1,j)/pts4d.at<double>(3,j);
        landmark.z = pts4d.at<double>(2,j)/pts4d.at<double>(3,j);
        pts3d.emplace_back(landmark);
    }

    ref2dPts.clear(); trk2dPts.clear(); ref3dpts.clear();
    ref2dPts = refPts; trk2dPts = trkPts; ref3dpts = pts3d;
}

void monoOdom::initSequence(){
    Mat ima = loadImage(iter);
    Mat imb = loadImage(iter+1);
    vector<Point2f> refPts, trkPts;
    vector<Point3f> ref3d;
    monoTriangulate(ima, imb, refPts, trkPts, ref3d);

    prevFeatures = refPts;
    prev3d = ref3d;

    imMeta.im1 = ima; imMeta.im2 = imb; imMeta.refPts = refPts; imMeta.trkPts = trkPts; imMeta.pts3d = ref3d;
    imMeta.R = R; imMeta.t = t;
    Rmono = R.clone(); tmono = t.clone();
    rvec = R.clone(); tvec = t.clone();
    poseGraph.initializeGraph();
    prevImMeta = imMeta;
    iter+=2;
}

void monoOdom::relocalize(int start, Mat imL, Mat imR, Mat&inv_transform, vector<Point2f>&ftrPts, vector<Point3f>&pts3d){
    vector<Point2f> new2d, newTrk;
    vector<Point3f> new3d;
    
    ftrPts.clear();
    pts3d.clear();

    monoTriangulate(imL, imR, new2d, newTrk, new3d);

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

void monoOdom::PyrLKtrackFrame2Frame(Mat refimg, Mat curImg, vector<Point2f>refPts, vector<Point3f>ref3dpts,
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
            inlierRefPts.emplace_back(refPts[j]);
            ref3dretPts.emplace_back(ref3dpts[j]);
            refRetpts.emplace_back(trackPts[j]);
        }
    }
}

void monoOdom::stageForPGO(Mat Rl, Mat tl, Mat Rg, Mat tg, bool loopClose){
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

void monoOdom::updateOdometry(vector<Eigen::Isometry3d>&T){
    cerr<<"\n\nUpdating global odometry measurements"<<endl;
    vector<float> data;
    for(Eigen::Isometry3d &isoMatrix : T){
        Mat t = Eigen2cvMat(isoMatrix);
        data.emplace_back(t.at<double>(0));
        data.emplace_back(t.at<double>(1));
        data.emplace_back(t.at<double>(2));
        trajectory.emplace_back(t.clone());
    }
    dumpOptimized(data);
}

void monoOdom::checkLoopDetector(Mat img, int idx){
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

void monoOdom::loopSequence(){
    vector<float> data; vector<imageMetadata> metaData;
    for(int i=iter; i<4500; i++){
        Mat ima = loadImage(i); Mat imb = loadImage(i+1);
        vector<Point2f> refPts, trkPts;
        vector<Point3f> ref3d;

        vector<Point2f> inlierRefPts;
        vector<Point3f> inlier3dPts;

        monoTriangulate(ima, imb, refPts, trkPts, ref3d);

        Mat rv, tv;
        double xgt,ygt,zgt;
        double absScale = getAbsoluteScale(i, xgt, ygt, zgt);

        Mat Rcpy, tcpy;
        Rcpy = R.clone();
        tcpy = t.clone();

        //BundleAdjust3d2d(refPts, ref3d, K, Rcpy, tcpy);

        if(absScale<0.1){
            tvec = tvec;
            rvec = rvec;
            tmono = tmono;
            Rmono = Rmono;
        }
        else{
            tvec = tvec + absScale*(rvec*t);
            rvec = rvec*R;
            tmono = tmono + absScale*(Rmono*t);
            Rmono = Rmono*R;
        }
        // rvec = R.t();
        // tvec = -rvec*t;

        Mat inv_transform = Mat::zeros(3,4, CV_64F);
        R.col(0).copyTo(inv_transform.col(0));
        R.col(1).copyTo(inv_transform.col(1));
        R.col(2).copyTo(inv_transform.col(2));
        t.copyTo(inv_transform.col(3));

        checkLoopDetector(ima, i);

        if(LC_FLAG){
            stageForPGO(R, t, Rmono, tmono, true);
            stageForPGO(R, t, Rmono, tmono, false);
            std::vector<Eigen::Isometry3d> trans = poseGraph.globalOptimize();
            Mat interT = Eigen2cvMat(trans[trans.size()-1]);
            tmono = interT.t();
        }
        else{
            stageForPGO(R, t, Rmono, tmono, false);
        }

        data.emplace_back(i);
        data.emplace_back(tvec.at<double>(0));
        data.emplace_back(tvec.at<double>(1));
        data.emplace_back(tvec.at<double>(2));
        data.emplace_back(xgt);
        data.emplace_back(ygt);
        data.emplace_back(zgt);
        data.emplace_back(-1.00);


        //cout<<int(tvec.at<double>(0))<<" "<<int(tvec.at<double>(2))<<" "<<int(xgt)<<" "<<int(zgt)<<en
        Point2f center = Point2f(int(tvec.at<double>(0)) + 750, int(-1*tvec.at<double>(2)) + 200);        
        Point2f centerGT = Point2f(xgt + 750, zgt + 200);
        circle(canvas, centerGT ,1, Scalar(0,255,0), 1);
        circle(canvas, center ,1, Scalar(0,0,255), 1);
        rectangle(canvas, Point2f(10, 30), Point2f(550, 50),  Scalar(0,0,0), cv::FILLED);

        imMeta.im1 = ima; imMeta.im2 = imb; imMeta.refPts = refPts; imMeta.trkPts = trkPts; imMeta.pts3d = ref3d;
        imMeta.R = rvec; imMeta.t = tvec;
        prevImMeta = imMeta;

        
        debug1 = drawDeltasErr(ima, refPts, trkPts);
        imshow("debug", debug1);
        imshow("Original Image",ima);
        imshow("trajectory", canvas);
        
        if(i==iter){
            createData(data);
        }
        else{
            appendData(data);
        }
        // if(LC_FLAG){
        //     if(cooldownTimer==0){
        //         vector<Eigen::Isometry3d> res = poseGraph.globalOptimize();
        //         updateOdometry(res);
        //         cerr<<"Redrawing Trajectory"<<endl;
        //         Mat tUpdated;
        //         for(Mat& position : trajectory){
        //             Point2f centerMono = Point2f(int(position.at<double>(0)) + 750, int(-1*position.at<double>(2)) + 200);
        //             circle(canvas, centerMono ,1, Scalar(255,0,0), 1);
        //             tUpdated = position.clone();
        //         }
        //         tmono = tUpdated.clone();
        //         LC_FLAG = false;
        //     }
        //     else{
        //         cooldownTimer--;
        //     }
        // }

        LC_FLAG = false;
        data.clear();

        cerr<<"Timer : "<<cooldownTimer<<"\r";
        if(cooldownTimer!=0){
            cooldownTimer-=1;
        }
        int k = waitKey(1);
        if(k=='q'){
            break;
        }
    }

    poseGraph.saveStructure();
    vector<Eigen::Isometry3d> res = poseGraph.globalOptimize();
    updateOdometry(res);
    cerr<<"\nRedrawing Trajectory\nPress any key to quit (even power key, lmao.)"<<endl;
    for(Mat& position : trajectory){
        Point2f centerMono = Point2f(int(position.at<double>(0)) + 750, int(-1*position.at<double>(2)) + 200);
        circle(canvas, centerMono ,1, Scalar(255,0,0), 1);
    }
    imshow("trajectory", canvas);
    waitKey(0);
    imwrite("Trajectory.png",canvas);
    cerr<<"Trajectory Saved"<<endl;
}

int main(){
    const char* impathL = "/media/gautham/Seagate Backup Plus Drive/Datasets/dataset/sequences/00/image_0/%0.6d.png";
    const char* impathR = "/media/gautham/Seagate Backup Plus Drive/Datasets/dataset/sequences/00/image_0/%0.6d.png";

    monoOdom od(0, impathL, impathR);
    od.initSequence();
    od.loopSequence();
    //od.pureMonocularSequence();
    return 0;
}