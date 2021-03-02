#include <iostream>
#include <vector>
#include <string>
#include "../include/stereoCV.h"


using namespace std; using namespace cv;


Mat StereoProcess::getImg(const char* fptr, int iter){
    char filename[200];
    sprintf(filename,fptr,iter);

    Mat im = imread(filename);
    if(!im.data){
        cout<<"\n\nYIKES Dawg, Failed to fetch frame, check the file path\n\n"<<endl;
    }
    return im;
}

Mat StereoProcess::stereoMatch(int iter){
    Mat im1 = getImg(lFptr, iter);
    Mat im2 = getImg(rFptr, iter);

    double lambda = 400; double sigma = 0.4;

    Ptr<ximgproc::DisparityWLSFilter> wls_filter;

    //cvtColor(im1, im1, CV_BGR2RGB);
    //cvtColor(im2, im2, CV_BGR2RGB);

    lImg = im1; rImg = im2;
    Mat grayIm1, grayIm2;

    cvtColor(im1, grayIm1, CV_BGR2GRAY);
    cvtColor(im2, grayIm2, CV_BGR2GRAY);

    int winSize = 1;
    Ptr<StereoSGBM> matcher = StereoSGBM::create(
        1, 
        96, 
        7, 
        8*3*winSize*winSize,
        32*3*winSize*winSize,
        0,
        60,
        0,
        3000,
        5
    );
    //wls_filter = ximgproc::createDisparityWLSFilter(matcher);
    //Ptr<StereoMatcher> right_matcher = ximgproc::createRightMatcher(matcher);
    Mat disp, rdisp, filtDisp;
    matcher->compute(grayIm1, grayIm2, disp);
    //right_matcher->compute(grayIm2, grayIm1,rdisp);
    //wls_filter->setLambda(lambda);
    //ls_filter->setSigmaColor(sigma);

    //wls_filter->filter(disp, grayIm1, filtDisp, rdisp);

    return disp;
}

void StereoProcess::stereoTriangulate(cv::Mat im1, cv::Mat im2, vector<cv::Point3f>&out3d){
    Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create(20000);
    Ptr<FeatureDetector> brief = xfeatures2d::BriefDescriptorExtractor::create();
    vector<KeyPoint> kp1, kp2;
    detector->detect(im1, kp1);
    detector->detect(im2, kp2);

    Mat desc1, desc2;
    brief->compute(im1, kp1, desc1);
    brief->compute(im2, kp2, desc2);
    desc1.convertTo(desc1, CV_32F);
    desc2.convertTo(desc2, CV_32F);

    BFMatcher matcher;
    vector<vector<DMatch>> matches;

    matcher.knnMatch(desc1, desc2, matches,2);
    vector<Point2f> pt1, pt2;
    for(size_t i=0; i<matches.size(); i++){
        DMatch &m = matches[i][0]; DMatch &n = matches[i][1];
        if(m.distance<0.8*n.distance){
            pt1.emplace_back(kp1[m.queryIdx].pt);
            pt2.emplace_back(kp2[m.trainIdx].pt);
        }
    }
    cerr<<"ckpt2"<<endl;
    Mat F; vector<uchar> mask;
    F = findFundamentalMat(pt1, pt2, CV_FM_RANSAC, 3.0, 0.99, mask);
    
    vector<Point2f> inlier1, inlier2;
    for(size_t i=0; i<mask.size(); i++){
        if(mask[i]==1){
            inlier1.emplace_back(pt1[i]);
            inlier2.emplace_back(pt2[i]);
        }
    }

    Mat P1 = Mat::zeros(3,4, CV_64F);
    Mat P2 = Mat::zeros(3,4, CV_64F);
    P1.at<double>(0,0) = 1; P1.at<double>(1,1) = 1; P1.at<double>(2,2) = 1;
    P2.at<double>(0,0) = 1; P2.at<double>(1,1) = 1; P2.at<double>(2,2) = 1;
    P2.at<double>(0,3) = -baseline;

    P1 = K*P1; P2 = K*P2;
    cerr<<"ckpt3"<<endl;
    Mat est3d;
    triangulatePoints(P1, P2, inlier1, inlier2, est3d);
    
    out3d.reserve(est3d.cols);
    for(size_t i=0; i<est3d.cols; i++){
        Point3f localpt;
        localpt.x = est3d.at<float>(0,i) / est3d.at<float>(3,i);
        localpt.y = est3d.at<float>(1,i) / est3d.at<float>(3,i);
        localpt.z = est3d.at<float>(2,i) / est3d.at<float>(3,i);
        out3d.emplace_back(localpt);
    }
    cerr<<"Completed"<<endl;
}

void StereoProcess::monocularTriangulate(Mat im1, Mat im2, vector<Point3f>&out3d){
    Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create(10000);
    
    vector<KeyPoint> kp1, kp2;
    detector->detect(im1, kp1);
    detector->detect(im2, kp2);

    Mat desc1, desc2;
    detector->compute(im1, kp1, desc1);
    detector->compute(im2, kp2, desc2);
    desc1.convertTo(desc1, CV_32F);
    desc2.convertTo(desc2, CV_32F);

    BFMatcher matcher;
    vector<vector<DMatch>> matches;

    matcher.knnMatch(desc1, desc2, matches,2);
    vector<Point2f> pt1, pt2;
    for(size_t i=0; i<matches.size(); i++){
        DMatch &m = matches[i][0]; DMatch &n = matches[i][1];
        if(m.distance<0.8*n.distance){
            pt1.emplace_back(kp1[m.queryIdx].pt);
            pt2.emplace_back(kp2[m.trainIdx].pt);
        }
    }

    Mat F; vector<uchar> mask;
    F = findFundamentalMat(pt1, pt2, CV_FM_RANSAC, 3.0, 0.99, mask);
    
    vector<Point2f> inlier1, inlier2;
    for(size_t i=0; i<mask.size(); i++){
        if(mask[i]==1){
            inlier1.emplace_back(pt1[i]);
            inlier2.emplace_back(pt2[i]);
        }
    }

    Mat E, R, t;

    E = findEssentialMat(inlier1, inlier2, K, 8, 0.99, 1, noArray());
    recoverPose(E, inlier1, inlier2, K, R, t, noArray());

    Mat P1 = Mat::zeros(3,4, CV_64F);
    Mat P2 = Mat::zeros(3,4, CV_64F);
    P1.at<double>(0,0) = 1; P1.at<double>(1,1) = 1; P1.at<double>(2,2) = 1;
    //P2.at<double>(0,0) = R.at<double>(0,0); P2.at<double>(1,1) = 1; P2.at<double>(2,2) = 1;
    //P2.at<double>(0,3) = -baseline;
    R.col(0).copyTo(P2.col(0));
    R.col(1).copyTo(P2.col(1));
    R.col(2).copyTo(P2.col(2));
    t.copyTo(P2.col(3));


    P1 = K*P1; P2 = K*P2;

    Mat est3d;
    triangulatePoints(P1, P2, inlier1, inlier2, est3d);
    
    out3d.reserve(est3d.cols);
    for(size_t i=0; i<est3d.cols; i++){
        Point3f localpt;
        localpt.x = est3d.at<float>(0,i) / est3d.at<float>(3,i);
        localpt.y = est3d.at<float>(1,i) / est3d.at<float>(3,i);
        localpt.z = est3d.at<float>(2,i) / est3d.at<float>(3,i);
        out3d.emplace_back(localpt);
    }
}

void StereoProcess::visualizeCloud(vector<cv::Point3f>pts3d, vector<cv::Point3f>colorMap){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    cloud->points.resize(pts3d.size());
    double centroidX = 0.0; double centroidY = 0.0; double centroidZ = 0.0;
    for(size_t i=0; i<pts3d.size(); i++){
        cloud->points[i].x = pts3d[i].x; centroidX += pts3d[i].x;
        cloud->points[i].y = pts3d[i].y; centroidY += pts3d[i].y;
        cloud->points[i].z = pts3d[i].z; centroidZ += pts3d[i].z;
        cloud->points[i].b = colorMap[i].x; cloud->points[i].g = colorMap[i].y; cloud->points[i].r = colorMap[i].z;
        //cloud->points[i].r = 0; cloud->points[i].g = 255; cloud->points[i].b = 0;
    }
    centroidX /= cloud->points.size();
    centroidY /= cloud->points.size();
    centroidZ /= cloud->points.size();


    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer("PointCloud Render"));
    viewer->setBackgroundColor(0,0,0);
    viewer->setCameraPosition(centroidX, centroidY, centroidZ, 0,0,0,0);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "Data");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Data");
    viewer->initCameraParameters();

    while(!viewer->wasStopped()){
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}

void StereoProcess::reprojectDisparity(cv::Mat disp, vector<cv::Point3f>&reproject3dPoints, vector<cv::Point3f>&colorMap){
    reproject3dPoints.clear(); colorMap.clear();
    Mat Q = Mat::zeros(4,4,CV_64F);
    
    Mat R = Mat::eye(3,3,CV_64F); Mat t = Mat::zeros(3,1, CV_64F);
    Mat R1, R2, P1, P2;
    t.at<double>(0,0) = baseline;

    stereoRectify(K, Mat::zeros(4,1,CV_64F), K ,Mat::zeros(4,1,CV_64F), disp.size(), R, t, R1, R2, P1, P2, Q);
    disp.convertTo(disp, CV_32F);
    Mat img3d(disp.size(), CV_32FC3);
    reprojectImageTo3D(disp,img3d,Q);

    cout<<disp.size()<<" "<<lImg.size()<<endl;


    for(int i=0; i<disp.rows; i++){
        for(int j=0; j<disp.cols; j++){
            Vec3f depth = img3d.at<Vec3f>(i,j); Vec3b colors = lImg.at<Vec3b>(i,j);
            if(depth[2]>5 or depth[2]<=0.01){
                continue;
            }
            Point3f pt; Point3f colorData;
            pt.x = depth[0]; pt.y = depth[1]*-1; pt.z = depth[2];
            colorData.x = colors[0]; colorData.y = colors[1]; colorData.z = colors[2];
            reproject3dPoints.emplace_back(pt);
            colorMap.emplace_back(colorData);
        }
    }
}

void StereoProcess::mainLoop(){
    Mat disp, normDisp;
    for(int i=0; i<4000; i+=1){
        disp = stereoMatch(i);
        normalize(disp, normDisp, 0, 255,cv::NORM_MINMAX,CV_8U);
        applyColorMap(normDisp, normDisp, COLORMAP_JET); 
        reprojectDisparity(disp, tri3dPoints, color3dMap);
        pclPublish(tri3dPoints, color3dMap);
        ros::spinOnce();
        imshow("disparity", normDisp);
        int k = waitKey(10);
        if(k=='q'){
            //stereoTriangulate(lImg, rImg, tri3dPoints);
            //monocularTriangulate(lImg, prevImg, tri3dPoints);
            visualizeCloud(tri3dPoints, color3dMap);
            cerr<<"breaking"<<endl;
            break;
        }
        prevImg = lImg; 
    }
    cerr<<"all done"<<endl;
}

void StereoProcess::pclPublish(vector<cv::Point3f>&pts3d, vector<cv::Point3f>&colorMap){
    cloudType::Ptr msg (new cloudType);
    msg->header.frame_id = "map";
    int mulFactor = 5;
    //msg->header.stamp = ros::Time::now();
    for(size_t i=0; i<pts3d.size(); i++){
        pcl::PointXYZRGB clPt;
        clPt.x = pts3d[i].x * mulFactor; clPt.y = pts3d[i].z *mulFactor; clPt.z = pts3d[i].y*mulFactor;
        clPt.r = colorMap[i].z; clPt.g = colorMap[i].y; clPt.b = colorMap[i].x;
        msg->points.emplace_back(clPt);
    }
    //std::vector<int> indices;
    //pcl::removeNaNFromPointCloud(*msg,*msg, indices);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(msg);
    sor.setMeanK(20);
    sor.setStddevMulThresh(0.8);
    sor.filter(*msg);

    pub.publish(msg);
}



int main(int argc, char **argv){
    ros::init(argc, argv, "StereoPublisher");
    const char* impathL = "/media/gautham/Seagate Backup Plus Drive/Datasets/ColorSeq/dataset/sequences/00/image_2/%0.6d.png";
    const char* impathR = "/media/gautham/Seagate Backup Plus Drive/Datasets/ColorSeq/dataset/sequences/00/image_3/%0.6d.png";

    StereoProcess *stereo = new StereoProcess(impathL, impathR);
    stereo->mainLoop();
}


