#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <stdio.h>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void adaptiveNonMaximalSuppresion( std::vector<cv::KeyPoint>& keypoints,
                                    const int numToKeep )
{
    if( keypoints.size() < numToKeep ) { return; }

    //
    // Sort by response
    //
    std::sort( keypoints.begin(), keypoints.end(),
                [&]( const cv::KeyPoint& lhs, const cv::KeyPoint& rhs )
                {
                return lhs.response > rhs.response;
                } );

    std::vector<cv::KeyPoint> anmsPts;

    std::vector<double> radii;
    radii.resize( keypoints.size() );
    std::vector<double> radiiSorted;
    radiiSorted.resize( keypoints.size() );

    const float robustCoeff = 1.11; // see paper

    for( int i = 0; i < keypoints.size(); ++i )
    {
    const float response = keypoints[i].response * robustCoeff;
    double radius = std::numeric_limits<double>::max();
    for( int j = 0; j < i && keypoints[j].response > response; ++j )
    {
        radius = std::min( radius, cv::norm( keypoints[i].pt - keypoints[j].pt ) );
    }
    radii[i]       = radius;
    radiiSorted[i] = radius;
    }

    std::sort( radiiSorted.begin(), radiiSorted.end(),
                [&]( const double& lhs, const double& rhs )
                {
                return lhs > rhs;
                } );

    const double decisionRadius = radiiSorted[numToKeep];
    for( int i = 0; i < radii.size(); ++i ){
        if( radii[i] >= decisionRadius ){
            anmsPts.push_back( keypoints[i] );
        }
    }

    anmsPts.swap( keypoints );
}

int main(){
    cv::Mat image = imread("/media/gautham/Seagate Backup Plus Drive/Datasets/ColorSeq/dataset/sequences/00/image_3/000000.png");
    imshow("Original : ", image);

    vector<KeyPoint> anmsKps;
    vector<KeyPoint> orbKps;

    cv::FAST(image, anmsKps, 1);

    drawKeypoints(image, anmsKps, image, Scalar(0,255,0));
    imshow("Default kps : ", image);

    int k = waitKey(0);
}