#ifndef LD_H
#define LD_H

#include "DBoW2/DBoW2.h"

#include <iostream>
#include <vector>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "poseGraph.h"
#include "DloopDet.h"
#include "TemplatedLoopDetector.h"

using namespace cv;
using namespace std;
using namespace DLoopDetector;
using namespace DBoW2;

typedef TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB> KeyFrameSelection;

class templatedLoopDetector{
    int LCidx = 0;
    int coolDown = 0;
    int LC_FLAG = false;

    std::shared_ptr<OrbLoopDetector> loopDetector;
    std::shared_ptr<OrbVocabulary> voc;
    std::shared_ptr<KeyFrameSelection> KFselector;
    std::string vocfile = "/home/gautham/Documents/Projects/LargeScaleMapping/orb_voc00.yml.gz";

    templatedLoopDetector(int nCols, int nRows, bool useNSS, double alphaThreshold, int k, int nImages){
        Params param;
        param.image_cols = nCols;
        param.image_rows = nRows;
        param.use_nss = useNSS;
        param.alpha = alphaThreshold;
        param.k = k;
        param.geom_check = GEOM_DI;
        param.di_levels = 2; 

        voc.reset(new OrbVocabulary());
        cerr<<"Loading vocabulary..."<<endl;
        voc->load(vocfile);
        cerr<<"Done"<<endl;

        loopDetector.reset(new OrbLoopDetector(*voc, param));
        loopDetector->allocate(nImages);
    }

    void restructure (cv::Mat& plain, vector<FORB::TDescriptor> &descriptors){  
        const int L = plain.rows;
        descriptors.resize(L);
        for (unsigned int i = 0; i < (unsigned int)plain.rows; i++) {
            descriptors[i] = plain.row(i);
        }
    }
    void scanLoopDetector(Mat img, int idx){
        Ptr<FeatureDetector> orb = ORB::create();
        vector<KeyPoint> kp;
        Mat desc;
        vector<FORB::TDescriptor> descriptors;

        orb->detectAndCompute(img, Mat(), kp, desc);
        restructure(desc, descriptors);
        DetectionResult result;
        loopDetector->detectLoop(kp, descriptors, result);
        if(result.detection() &&(result.query-result.match > 100) && coolDown==0){
            cerr<<"Found Loop Closure between "<<idx<<" and "<<result.match<<endl;
            LC_FLAG = true;
            LCidx = result.match-1;
            coolDown = 100;
        }
    }


};

#endif