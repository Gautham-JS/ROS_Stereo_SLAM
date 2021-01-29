#include "../include/visualSLAM.h"

void visualSLAM::SORcloud(vector<Point3f>&ref3d, vector<Point3f>&colorMap){
    cloudType::Ptr bufferCloud (new cloudType);
    for(size_t i=0; i<ref3d.size(); i+=1){
        if(-1*ref3d[i].z>500){
            continue;
        }
        pcl::PointXYZRGB clPt;
        clPt.x = ref3d[i].x; clPt.y = ref3d[i].y; clPt.z = ref3d[i].z;
        clPt.r = colorMap[i].z; clPt.g = colorMap[i].y; clPt.b = colorMap[i].x;
        //msg->points.emplace_back(clPt);
        bufferCloud->points.emplace_back(clPt);
    }
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(bufferCloud);
    sor.setMeanK(200);
    sor.setStddevMulThresh(0.1);
    sor.filter(*bufferCloud);

    ref3d.clear(); colorMap.clear();
    for(size_t i=0; i<bufferCloud->points.size(); i++){
        pcl::PointXYZRGB cldPt;
        Point3f p3d, c3d;

        cldPt = bufferCloud->points[i];

        p3d.x = cldPt.x; p3d.y = cldPt.y; p3d.z = cldPt.z;
        c3d.x = cldPt.b; c3d.y = cldPt.g; c3d.z = cldPt.r;

        ref3d.emplace_back(p3d);
        colorMap.emplace_back(c3d);
    }
}

void visualSLAM::rosPublish(vector<vector<Point3f>>&pt3d, Mat&trajROS, Mat&Rmat){
    //cerr<<"Publishing messages"<<endl;
    cloudType::Ptr msg (new cloudType);
    geometry_msgs::PoseStamped poseMsg;
    msg->header.frame_id = "map";
    double mulFactor = 0.1;

    int siz = 0;
    for(size_t k=0; k<pt3d.size(); k++){
        vector<Point3f> ref3dCoords = pt3d[k];
        vector<Point3f> colorMap = colorHistory[k];
        for(size_t i=0; i<ref3dCoords.size(); i+=1){
            if(-1*ref3dCoords[i].z>500){
                continue;
            }
            pcl::PointXYZRGB clPt;
            siz+=1;
            clPt.x = ref3dCoords[i].x * mulFactor; clPt.y = ref3dCoords[i].z *mulFactor; clPt.z = -1*ref3dCoords[i].y*mulFactor;
            clPt.r = colorMap[i].z; clPt.g = colorMap[i].y; clPt.b = colorMap[i].x;
            msg->points.emplace_back(clPt);
            }
        }
    if(SHUTDOWN_FLAG){
        cerr<<"SAVING POINTCLOUD as "<<plySavepath<<endl;
        pcl::io::savePLYFileBinary(plySavepath, *msg);
        cerr<<"DONE"<<endl;
    }
    mapPublisher.publish(msg);

    double Xpos, Ypos, Zpos;
    trajROS*=mulFactor;

    Xpos = trajROS.at<double>(0);
    Zpos = trajROS.at<double>(1)*-1;
    Ypos = trajROS.at<double>(2);
    
    Eigen::Quaterniond q;
    Rmat2Quat(Rmat, q);
    Eigen::Vector4d Qvector = q.coeffs();

    //cerr<<"Quat : "<<Qvector<<endl;

    poseMsg.header.frame_id = "map";
    poseMsg.pose.position.x = Xpos;
    poseMsg.pose.position.y = Ypos;
    poseMsg.pose.position.z = Zpos;

    poseMsg.pose.orientation.x = Qvector[0];
    poseMsg.pose.orientation.y = Qvector[2];
    poseMsg.pose.orientation.z = Qvector[1]*-1;
    poseMsg.pose.orientation.w = Qvector[3];

    posePublisher.publish(poseMsg);
    
    trajectoryMsg.header.frame_id = "map";
    trajectoryMsg.poses.emplace_back(poseMsg);
    trajectoryPublisher.publish(trajectoryMsg);
}
