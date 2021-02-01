#include "../include/visualSLAM.h"


vector<KeyPoint> visualSLAM::denseKeypointExtractor(Mat img, int stepSize){
    vector<KeyPoint> out;
    for (int y=stepSize; y<img.rows-stepSize; y+=stepSize){
        for (int x=stepSize; x<img.cols-stepSize; x+=stepSize){
            out.push_back(KeyPoint(float(x), float(y), float(stepSize)));
        }
    }
    return out;
}

void visualSLAM::denseLKtracking(Mat refImg, Mat curImg, vector<Point2f>&refPts, vector<Point2f>&trackPts){
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

void visualSLAM::FmatThresholding(vector<Point2f>&refPts, vector<Point2f>&trkPts){
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


void visualSLAM::PyrLKtrackFrame2Frame(Mat refimg, Mat curImg, vector<Point2f>refPts, vector<Point3f>ref3dpts,
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
            inlierRefPts.push_back(refPts[j]);
            ref3dretPts.push_back(ref3dpts[j]);
            refRetpts.push_back(trackPts[j]);
        }
    }
    //refRetpts = inlierTracked;
    //ref3dretPts = inlierRef3dPts;
    inlierReferencePyrLKPts = inlierRefPts;
}