#include "../include/visualSLAM.h"


void visualSLAM::initSequence(){
    int iter = 0;
    char FileName1[200], filename2[200];
    sprintf(FileName1, lFptr, iter);
    sprintf(filename2, rFptr, iter);

    // Mat imL = imread(FileName1);
    // Mat imR = imread(filename2);

    initPangolin();

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

    Eigen::Isometry3d curPose = cvMat2Eigen(R,Mat::zeros(1,3,CV_64F));
    isoVector.emplace_back(curPose);
    
    cerr<<"\n\n"<<endl;

    mutex renderMutex;
    std::thread renderThread([&](){ 
        DrawTrajectory(isoVector, mapHistory, colorHistory);
    });

    for(int iter=1; iter<4500; iter++){
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
            isoVector = trans;
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
        if(inliers.size()<400 or LC_FLAG==true){
            cerr<<"ENTERING KEYFRAME AT "<<iter<<"... "<<"\r";
            Mat i1 = loadImageL(iter); Mat i2 = loadImageR(iter);
            insertKeyFrames(0, i1, i2, pose4dTransform, ref2dFeatures, ref3dCoords);

            vector<Point3f> good3d = ref3dCoords;
            vector<Point3f> goodColors = colors;

            SORcloud(good3d, goodColors);

            Mat Rotation = R.clone(); Mat translation = t.clone();
            Eigen::Isometry3d curPose = cvMat2Eigen(Rotation,translation);

            renderMutex.lock();
            isoVector.emplace_back(curPose);
            mapHistory.emplace_back(good3d);
            colorHistory.emplace_back(goodColors);
            renderMutex.unlock();

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
        
        resize(frame, frame, Size(), 0.7, 0.7);
        Mat reSizOG;
        resize(currentImage, reSizOG, Size(), 0.7, 0.7);

        imshow("Debug", frame);
        imshow("frame",reSizOG);
        ros::spinOnce();
        int k = waitKey(1);
        if (k=='q'){
            imwrite("trajectoryUnopt.png", canvas);
            break;
        }
    }

    cerr<<"Total map size :"<<mapPts.size()<<endl;
    poseGraph.saveStructure();
    //vector<Eigen::Isometry3d> res = poseGraph.globalOptimize();
    //updateOdometry(res);

    renderThread.join();

    SHUTDOWN_FLAG = true;
    rosPublish(mapHistory, trajectory[trajectory.size()-1], keyFrameHistory[keyFrameHistory.size()-1].R);
    imwrite("Trajectory.png",canvas);
    cerr<<"Trajectory Saved"<<endl;
    //DrawTrajectory(res,mapHistory,colorHistory);
}


int main(int argc, char **argv){
    ros::init(argc, argv, "SLAM_node");

    const char* impathL = "/media/gautham/Seagate Backup Plus Drive/Datasets/ColorSeq/dataset/sequences/00/image_2/%0.6d.png";
    const char* impathR = "/media/gautham/Seagate Backup Plus Drive/Datasets/ColorSeq/dataset/sequences/00/image_3/%0.6d.png";
    std::string vocPath = "/home/gautham/Documents/Projects/LargeScaleMapping/orb_voc00.yml.gz";

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
