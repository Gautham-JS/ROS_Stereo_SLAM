#include <DBoW2/FeatureVector.h>
#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedDatabase.h>
#include <DBoW2/TemplatedVocabulary.h>
#include <DBoW2/DBoW2.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <bits/stdc++.h>

using namespace std;
using namespace cv;
using namespace DBoW2;

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out){
    out.resize(plain.rows);
    for(int i = 0; i < plain.rows; ++i){
      out[i] = plain.row(i);
    }
}

void loadFeatures(vector<vector<cv::Mat > > &features, int NIMAGES, vector<String> fSysHandle){
    features.clear();
    features.reserve(NIMAGES);

    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    cout << "Extracting ORB features..." << endl;
    for(int i = 0; i < NIMAGES; i+=1){
        cout << "images/image" << i << ".png"<<endl;

        cv::Mat image = cv::imread(fSysHandle[i], 0);
        cv::Mat mask;
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        orb->detectAndCompute(image, mask, keypoints, descriptors);

        features.push_back(vector<cv::Mat >());
        changeStructure(descriptors, features.back());
    }
}

void testVocCreation(const vector<vector<cv::Mat > > &features, int NIMAGES)
{
  // branching factor and depth levels 
  const int k = 9;
  const int L = 6;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L1_NORM;

  OrbVocabulary voc(k, L, weight, scoring);

  cout << "Creating a volptous " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;

  // for(int i = 0; i < NIMAGES; i++)
  // { 
  //   voc.transform(features[i], v1);
  //   double MaxScore = 0; int bestMatch = 0;

  //   for(int j = 0; j < NIMAGES/10; j++){
  //     if(i==j) continue;
  //     voc.transform(features[j], v2);
  //     double score = voc.score(v1, v2);
  //     if(score>MaxScore){
  //       MaxScore = score;
  //       bestMatch = j;
  //     }
  //     //cout << "Image " << i << " vs Image " << j << ": " << score << endl;
  //   }
  //   if(MaxScore>0.1){
  //       cout<<"\n\nBEST MATCH BETWEEN IMAGES "<<i<<" , "<<bestMatch<<" WITH SCORE "<<MaxScore<<endl<<endl;
  //   }
  //   else{
  //       cerr<<"Skippity skip "<<i<<endl;
  //       continue;
  //   }
  //   if(std::abs(i-bestMatch)<10){
  //     cout<<"FOUND ONLY LOCAL MATCHES"<<endl;
  //   }
  // }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("orb_voc.yml.gz");
  cout << "Done" << endl;
}

void orbDetect(Mat im, vector<KeyPoint>&keys, Mat descs){
    Ptr<FeatureDetector> detector = ORB::create();
    //Ptr<FeatureDetector> detector = xfeatures2d::SIFT::create();
    detector->detect(im, keys);
    detector->compute(im, keys, descs);
}


int main(){
    vector<cv::String> indices;
    glob("/media/gautham/Seagate Backup Plus Drive/Datasets/dataset/sequences/08/image_0/*.png", indices, false);
    
    Mat prevImg;
    vector<vector<Mat>> orbFeatures;

    loadFeatures(orbFeatures, indices.size(), indices);

    testVocCreation(orbFeatures, indices.size());

    for(size_t i=0; i< indices.size(); i+=1){
        cout<<"Processing Frame "<<i<<endl;
        Mat curImg;
        curImg = imread(indices[i],0);
        vector<KeyPoint> kps; Mat descs;
        orbDetect(curImg, kps, descs);
        Mat out;
        drawKeypoints(curImg, kps, out, Scalar(0,255,0));
        
        imshow("Image", curImg); imshow("local features", out);
        int k = cv::waitKey(50);
        if(k=='q'){
            break;
        }
    }
}