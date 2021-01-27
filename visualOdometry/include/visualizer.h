#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>

//#include <pcl_conversions/pcl_conversions.h>
#include "pcl-1.8/pcl/conversions.h"
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

void visualizeCloud(vector<cv::Point3f>pts3d){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    cloud->points.resize(pts3d.size());
    double centroidX = 0.0; double centroidY = 0.0; double centroidZ = 0.0;
    for(size_t i=0; i<pts3d.size(); i++){
        cloud->points[i].x = pts3d[i].x; centroidX += pts3d[i].x;
        cloud->points[i].y = pts3d[i].y; centroidY += pts3d[i].y;
        cloud->points[i].z = pts3d[i].z; centroidZ += pts3d[i].z;
        //cloud->points[i].r = colorMap[i].x; cloud->points[i].g = colorMap[i].y; cloud->points[i].b = colorMap[i].z;
        cloud->points[i].r = 0; cloud->points[i].g = 255; cloud->points[i].b = 0;
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
    //viewer->addCoordinateSystem(1.0);

    while(!viewer->wasStopped()){
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}

