/*
GAUTHAM-JS , FEB-2021;
gauthamjs56@gmail.com
PART OF ROS STERO SLAM, UNDER MIT LICENSE.
*/

#include "../include/visualSLAM.h"

#define UI_WIDTH 200
#define CHAR_LIM 16

string ack = "";
int seed = 0;
int interval = 0;

void visualSLAM::initPangolin(){
  cerr<<"GLInitStart"<<endl;
  pangolin::CreateWindowAndBind("Render",640,480);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  pangolin::GetBoundWindow()->RemoveCurrent();
  cerr<<"GLInitEnd"<<endl;
}

void loopSequence(string master){
  if(seed>master.size()-CHAR_LIM){
    seed = 0;
  }
  ack =  master.substr(seed, CHAR_LIM);
}

void visualSLAM::DrawTrajectory(vector<Eigen::Isometry3d>&poses, vector<vector<Point3f>>&pts3,vector<vector<Point3f>>&colorData){
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  //pangolin::BindToContext("Render");

  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
    pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
  );

  pangolin::View &d_cam = pangolin::CreateDisplay().
      SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -1000.0f / 600.0f).
      SetHandler(new pangolin::Handler3D(s_cam));

  pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
  pangolin::Var<bool> menuUseRGB("menu.Show RGB",true,true);
  pangolin::Var<bool> menuShowKF("menu.Show KeyFrames",true,true);
  pangolin::Var<bool> menuShowTraj("menu.Show trajectory",true,true);
  pangolin::Var<bool> menuFollowCamera("menu.Follow Frame",true,true);
  
  pangolin::Var<int> menuSparsity("menu.Sparsity", 10, 10, 30, false);
  pangolin::Var<int> menuPtSize("menu.PointSize", 1, 1, 4, false);

  pangolin::Var<double> menuVarThr("menu.Variance Th", 1, 1, 300, false);
  pangolin::Var<bool> menuResetButton("menu.Reset", false, false);
  pangolin::Var<double> menuTrackFps("menu.Tracking FPS", 0, 0, 0, false);
  //pangolin::Var<string> menuString("menu.@","", false);

  int renderIter = 0;

  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glLineWidth(2);
    int kfCount = 0;
    
    pangolin::OpenGlMatrix Twc;

    renderMutex.lock();
    Eigen::Isometry3d TwcEigen = poses[poses.size()-1];
    Eigen::Matrix<double,4,4> currentPose = TwcEigen.matrix();

    for(size_t a = 0; a<4; a++){
      for(size_t b=0; b<4; b++){
        Twc(a,b) = currentPose(a,b);
      }
    }
    
    if(menuFollowCamera){
      s_cam.Follow(Twc);
    }
    //cerr<<"Cur Pose : "<<currentPose<<endl;
    if(menuShowKF){
      for (size_t i = 1; i < poses.size(); i++) {
          
          Eigen::Vector3d Ow = poses[i].translation();
          //Eigen::Vector3d OwM1 = poses[i-1].translation();
          Eigen::Vector3d Xw = poses[i] * (0.1 * Eigen::Vector3d(1, 0, 0));
          Eigen::Vector3d Yw = poses[i] * (0.1 * Eigen::Vector3d(0, 1, 0));
          Eigen::Vector3d Zw = poses[i] * (0.1 * Eigen::Vector3d(0, 0, 1));

          float w = (Y_BOUND*0.001)/3; float h = (X_BOUND*0.001)/3; float z = (Y_BOUND*0.0005)/3;

          Eigen::Vector3d linSet00 = poses[i] * (Eigen::Vector3d(w,h,z));
          Eigen::Vector3d linSet01 = poses[i] * (Eigen::Vector3d(w,-h,z));
          Eigen::Vector3d linSet02 = poses[i] * (Eigen::Vector3d(-w,-h,z));
          Eigen::Vector3d linSet03 = poses[i] * (Eigen::Vector3d(-w,h,z));

          Eigen::Vector3d extLine0 = poses[i] * (Eigen::Vector3d(w,h,z));
          Eigen::Vector3d extLine1 = poses[i] * (Eigen::Vector3d(w,-h,z));

          Eigen::Vector3d extLine2 = poses[i] * (Eigen::Vector3d(-w,h,z));
          Eigen::Vector3d extLine3 = poses[i] * (Eigen::Vector3d(-w,-h,z));

          Eigen::Vector3d extLine4 = poses[i] * (Eigen::Vector3d(-w,h,z));
          Eigen::Vector3d extLine5 = poses[i] * (Eigen::Vector3d(w,h,z));

          Eigen::Vector3d extLine6 = poses[i] * (Eigen::Vector3d(-w,-h,z));
          Eigen::Vector3d extLine7 = poses[i] * (Eigen::Vector3d(w,-h,z));


          glBegin(GL_LINES);
          //cerr<<Ow[0]-OwM1[0]<<" "<<Ow[1]-OwM1[1]<<" "<<Ow[2]-OwM1[2]<<endl;
          if(i==poses.size()-1){
            glColor3f(1.0, 1.0, 0.0);  
          }
          else{
            glColor3f(0.0, 0.0, 1.0);
          }
          glVertex3d(Ow[0], Ow[1], Ow[2]);
          glVertex3f(linSet00[0], linSet00[1], linSet00[2]);
          glVertex3d(Ow[0], Ow[1], Ow[2]);
          glVertex3f(linSet01[0], linSet01[1], linSet01[2]);
          glVertex3d(Ow[0], Ow[1], Ow[2]);
          glVertex3f(linSet02[0], linSet02[1], linSet02[2]);
          glVertex3d(Ow[0], Ow[1], Ow[2]);
          glVertex3f(linSet03[0], linSet03[1], linSet03[2]);

          glVertex3f(extLine0[0], extLine0[1], extLine0[2]);
          glVertex3f(extLine1[0],extLine1[1],extLine1[2]);

          glVertex3f(extLine2[0], extLine2[1], extLine2[2]);
          glVertex3f(extLine3[0], extLine3[1], extLine3[2]);

          glVertex3f(extLine4[0], extLine4[1], extLine4[2]);
          glVertex3f(extLine5[0], extLine5[1], extLine5[2]);

          glVertex3f(extLine6[0], extLine6[1], extLine6[2]);
          glVertex3f(extLine7[0], extLine7[1], extLine7[2]);

          // glVertex3d(Ow[0], Ow[1], Ow[2]);
          // glVertex3d(Xw[0], Xw[1], Xw[2]);
          // glColor3f(0.0, 1.0, 0.0);
          // glVertex3d(Ow[0], Ow[1], Ow[2]);
          // glVertex3d(Yw[0], Yw[1], Yw[2]);
          // glColor3f(0.0, 0.0, 1.0);
          // glVertex3d(Ow[0], Ow[1], Ow[2]);
          // glVertex3d(Zw[0], Zw[1], Zw[2]);
          glEnd();
      }
    }
    else{
        int i = poses.size()-1;
        Eigen::Vector3d Ow = poses[i].translation();
        //Eigen::Vector3d OwM1 = poses[i-1].translation();
        Eigen::Vector3d Xw = poses[i] * (0.1 * Eigen::Vector3d(1, 0, 0));
        Eigen::Vector3d Yw = poses[i] * (0.1 * Eigen::Vector3d(0, 1, 0));
        Eigen::Vector3d Zw = poses[i] * (0.1 * Eigen::Vector3d(0, 0, 1));

        float w = (Y_BOUND*0.001)/3; float h = (X_BOUND*0.001)/3; float z = (Y_BOUND*0.0005)/3;

        Eigen::Vector3d linSet00 = poses[i] * (Eigen::Vector3d(w,h,z));
        Eigen::Vector3d linSet01 = poses[i] * (Eigen::Vector3d(w,-h,z));
        Eigen::Vector3d linSet02 = poses[i] * (Eigen::Vector3d(-w,-h,z));
        Eigen::Vector3d linSet03 = poses[i] * (Eigen::Vector3d(-w,h,z));

        Eigen::Vector3d extLine0 = poses[i] * (Eigen::Vector3d(w,h,z));
        Eigen::Vector3d extLine1 = poses[i] * (Eigen::Vector3d(w,-h,z));

        Eigen::Vector3d extLine2 = poses[i] * (Eigen::Vector3d(-w,h,z));
        Eigen::Vector3d extLine3 = poses[i] * (Eigen::Vector3d(-w,-h,z));

        Eigen::Vector3d extLine4 = poses[i] * (Eigen::Vector3d(-w,h,z));
        Eigen::Vector3d extLine5 = poses[i] * (Eigen::Vector3d(w,h,z));

        Eigen::Vector3d extLine6 = poses[i] * (Eigen::Vector3d(-w,-h,z));
        Eigen::Vector3d extLine7 = poses[i] * (Eigen::Vector3d(w,-h,z));


        glBegin(GL_LINES);
        //cerr<<Ow[0]-OwM1[0]<<" "<<Ow[1]-OwM1[1]<<" "<<Ow[2]-OwM1[2]<<endl;
        if(i==poses.size()-1){
          glColor3f(1.0, 1.0, 0.0);  
        }
        else{
          glColor3f(0.0, 0.0, 1.0);
        }
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glVertex3f(linSet00[0], linSet00[1], linSet00[2]);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glVertex3f(linSet01[0], linSet01[1], linSet01[2]);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glVertex3f(linSet02[0], linSet02[1], linSet02[2]);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glVertex3f(linSet03[0], linSet03[1], linSet03[2]);

        glVertex3f(extLine0[0], extLine0[1], extLine0[2]);
        glVertex3f(extLine1[0],extLine1[1],extLine1[2]);

        glVertex3f(extLine2[0], extLine2[1], extLine2[2]);
        glVertex3f(extLine3[0], extLine3[1], extLine3[2]);

        glVertex3f(extLine4[0], extLine4[1], extLine4[2]);
        glVertex3f(extLine5[0], extLine5[1], extLine5[2]);

        glVertex3f(extLine6[0], extLine6[1], extLine6[2]);
        glVertex3f(extLine7[0], extLine7[1], extLine7[2]);

        // glVertex3d(Ow[0], Ow[1], Ow[2]);
        // glVertex3d(Xw[0], Xw[1], Xw[2]);
        // glColor3f(0.0, 1.0, 0.0);
        // glVertex3d(Ow[0], Ow[1], Ow[2]);
        // glVertex3d(Yw[0], Yw[1], Yw[2]);
        // glColor3f(0.0, 0.0, 1.0);
        // glVertex3d(Ow[0], Ow[1], Ow[2]);
        // glVertex3d(Zw[0], Zw[1], Zw[2]);
        glEnd();
    }
    
    //cerr<<"KFs : "<<kfCount<<endl;
    if(menuShowTraj){
      for (size_t i = 1; i < poses.size()-1; i++) {
        glColor3f(1.0, 0.0, 0.0);
        glLineWidth(2.0);
        glBegin(GL_LINES);
        auto p1 = poses[i], p2 = poses[i + 1];
        glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
        glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
        glEnd();
      }
    } 

    for(size_t j=0; j<pts3.size(); j++){
        vector<Point3f> localPts = pts3[j];
        vector<Point3f> localColors = colorData[j];

        if(!menuUseRGB && j==pts3.size()-1){
          glPointSize(menuPtSize);
          glColor3f(1.0, 0.0, 0.0);
          glBegin(GL_LINES);
          float edgeSize = 0.5;
          for (size_t l = 0; l < localPts.size(); l++){
            Point3f pt = localPts[l];
            glVertex3d(pt.x + edgeSize, pt.y, pt.z);
            glVertex3d(pt.x - edgeSize, pt.y, pt.z);

            glVertex3d(pt.x, pt.y+edgeSize, pt.z);
            glVertex3d(pt.x, pt.y-edgeSize, pt.z);

            glVertex3d(pt.x, pt.y, pt.z+edgeSize);
            glVertex3d(pt.x, pt.y, pt.z-edgeSize);

            // glVertex3d(pt.x + edgeSize, pt.y, pt.z);
            // glVertex3d(pt.x - edgeSize, pt.y, pt.z); 
          }
          glEnd();
        }
        else{
          glPointSize(menuPtSize);
        }
        glBegin(GL_POINTS);
        for(size_t k=0; k<localPts.size(); k++){
            Point3f p = localPts[k];
            Point3f c = localColors[k];
            double x = p.x; double r = c.z;
            double y = p.y; double g = c.y;
            double z = p.z; double b = c.x;
            if(menuUseRGB){
                glColor3f(r/255 ,g/255, b/255);
            }
            else{
              if(j==pts3.size()-1){
                glColor3f(1.0, 0.0, 0.0);
              }
              else{
                glColor3f(1.0, 1.0, 1.0);
              }
            }
            glVertex3d(x,y,z);
        }
        glEnd();
    }

    menuTrackFps = trackFPS;
    renderMutex.unlock();

    // loopSequence("      GauthamJ.S ; AkashSharma ; SuryankKumar");
    // if(interval==0){
    //   seed++;
    //   interval = 25;
    // }
    // else{
    //   interval--;
    // }
    renderIter++;

    // menuString = ack;

    pangolin::FinishFrame();
    usleep(5000); 
  }
  cerr<<"\n\nRENDERING THREAD REVOKED!\n\nSHUTTING DOWN MAIN THREAD TOO...\n"<<endl;
  SHUTDOWN_FLAG = true;
  cerr<<"GLloopALLdone"<<endl;
  pangolin::GetBoundWindow()->RemoveCurrent();
}