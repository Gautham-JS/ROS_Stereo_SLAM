#include "../include/visualSLAM.h"


Mat visualSLAM::drawDepthCMap(Mat image, vector<Point3f>&pts3d, vector<Point2f>&ref2d, vector<Point2f>&trk2d){    
    int thresh = 30;
    double bias = 15;
    vector<uchar> mapping;
    cvtColor(image, image, CV_BGR2GRAY); cvtColor(image, image, CV_GRAY2BGR);
    //cerr<<"Starting cmapping"<<endl;
    

    for(int i=0; i<ref2d.size(); i++){
        //Point2f pt = ref2d[i];
        Point3f pt3d = pts3d[i];
        double depth = pt3d.z;

        if(int(depth)<1 || int(depth)>thresh ){
            mapping.emplace_back(0);
            continue;
        }

        mapping.emplace_back(uchar(depth*10));
    }

    Mat m1d(mapping);
    Mat m2d = m1d.reshape(1,mapping.size());

    Mat mImg;

    applyColorMap(m2d, mImg, COLORMAP_JET);
    
    for(int i=0; i<ref2d.size(); i++){
        Vec3b &color = mImg.at<Vec3b>(0,i);
        Point2f pt1 = ref2d[i];
        Point2f pt2 = trk2d[i];

        uchar depth = m2d.at<uchar>(1,i);
        if(depth<1){
            continue;
        }
        Point2f sp1, sp2, sp3, sp4;
        sp1.x = pt1.x - bias/2;     
        sp1.y = pt1.y + bias/2;

        sp2.x = pt1.x + bias/2;     
        sp2.y = pt1.y + bias/2;

        sp3.x = pt1.x + bias/2;     
        sp3.y = pt1.y - bias/2;

        sp4.x = pt1.x - bias/2;     
        sp4.y = pt1.y - bias/2;

        line(image, sp1, sp2, color, 1);
        line(image, sp2, sp3, color, 1);
        line(image, sp3, sp4, color, 1);
        line(image, sp4, sp1, color, 1);

        circle(image, pt1, 3, color, CV_FILLED);
        //circle(image, pt1, 2, Scalar(0,255,0), CV_FILLED);
        //line(image, Point(int(pt1.x), int(pt1.y)), Point(int(pt2.x), int(pt2.y)), Scalar(color[0], color[1], color[2]), 1);
    }



    //cerr<<m1d<<endl;

    //imshow("mapped", image);

    return image;
}

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
        dkps = denseKeypointExtractor(im1, 30);

        //FAST(im1, dkps, 2);

        vector<Point2f> refPts;
        for(size_t i=0; i<dkps.size(); i++){
            refPts.emplace_back(dkps[i].pt);
        }

        vector<Point2f> trkPts;
        denseLKtracking(im1, im2, refPts, trkPts);
        FmatThresholding(refPts, trkPts);

        pt1 = refPts, pt2 = trkPts;
    }
    else{
        Ptr<FeatureDetector> detector = ORB::create(1000);
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
    
    //drw = drawDepthCMap(im1, ref3dCoords, pt1);
    //untransformed = ref3dCoords;
    ref3dPts = ref3dCoords;
    ref2dPts = pt1;
}