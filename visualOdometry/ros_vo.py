#!/usr/bin/env python
'''
-->Gautham J.S, 26/12/2020
A rosified implementation of
Visual Odometry + Stereo Reconstruction + P2P-ICP transformation
'''
import numpy as np
import cv2
import os
import math
import open3d as o3d
import matplotlib.pyplot as plt

import rospy
from math import pow, atan2, sqrt, sin, cos
from std_msgs.msg import Float64, Float64MultiArray, Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from sensor_msgs import point_cloud2 as pcl2
from geometry_msgs.msg import Pose, PoseStamped
from cv_bridge import CvBridge

from scipy.spatial.transform import Rotation


Trmat = [4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02, 
        -7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02, 
        9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01]


class PoseGraphOptimizer:
    def __init__(self,downsamplefactor=0.0):
        self.Pcd = o3d.geometry.PointCloud()
        self.Points = None
        self.pPcd = o3d.geometry.PointCloud()

        self.pcdBuffer = []
        self.downSampleFactor = downsamplefactor

        self.max_correspondence_distance_coarse = self.downSampleFactor * 15
        self.max_correspondence_distance_fine = self.downSampleFactor * 1.5

        self.pPoints = None
        self.pR_ = None
        self.pt_ = None

        self.count = 0
        self.ckpt = 1

        self.xpts = []
        self.ypts = []
        self.zpts = []

        self.X = None
        self.Y = None
        self.Z = None
    
    def loadBuffer(self, pcl):
        self.pcdBuffer.clear()
        self.pcdBuffer.append(self.pPcd)
        #pcl.voxel_down_sample(voxel_size = self.downSampleFactor)
        self.Pcd = pcl
        self.pcdBuffer.append(self.Pcd)
    
    def pairwiseRegistration(self, source, target):
        max_correspondence_distance_coarse = self.max_correspondence_distance_coarse
        max_correspondence_distance_fine = self.max_correspondence_distance_fine

        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_coarse, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        icp_fine = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_fine,
            icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        transformation_icp = icp_fine.transformation
        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance_fine,
            icp_fine.transformation)
        return transformation_icp, information_icp

    def fullRegistration(self,pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine):
        max_correspondence_distance_coarse = self.max_correspondence_distance_coarse
        max_correspondence_distance_fine = self.max_correspondence_distance_fine
        pose_graph = o3d.pipelines.registration.PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
        n_pcds = len(pcds)
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, n_pcds):
                transformation_icp, information_icp = self.pairwiseRegistration(
                    pcds[source_id], pcds[target_id])
                print("Build o3d.pipelines.registration.PoseGraph")
                if target_id == source_id + 1:  # odometry case
                    odometry = np.dot(transformation_icp, odometry)
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(
                            np.linalg.inv(odometry)))
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation_icp,
                                                                information_icp,
                                                                uncertain=False))
                else:  # loop closure case
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation_icp,
                                                                information_icp,
                                                                uncertain=True))
        return pose_graph
    
    def processAndUpdate(self,pcl):
        if self.count==0:
            #self.pPcd = pcl.voxel_down_sample(voxel_size=self.downSampleFactor)
            self.pPcd = pcl
            self.pPcd.estimate_normals()
            self.count+=1
            print("initializing ICP")
            return None, None
        
        self.Pcd = pcl
        self.Pcd.estimate_normals()
        max_correspondence_distance_coarse = self.downSampleFactor * 15
        max_correspondence_distance_fine = self.downSampleFactor * 1.5

        transformation, info = self.pairwiseRegistration(self.pPcd, self.Pcd, max_correspondence_distance_coarse, max_correspondence_distance_fine)
        R,t = transformation[:3, :3], transformation[:3, -1:]

        if self.count==1:
            print("initializing R,t")
            self.pR_ = R
            self.pt_ = t
            self.count+=1
            return R,t
        self.pt_ = self.pt_ + 3*(self.pR_.dot(t))
        self.pR_ = self.pR_.dot(R)

        tvec = self.pt_.T[0]
        self.X = tvec[0]
        self.Y = tvec[1]
        self.Z = tvec[2]
        self.xpts.append(tvec[0])
        self.ypts.append(tvec[2])
        self.zpts.append(tvec[1])


        self.pPcd = self.Pcd
        return self.pR_, self.pt_
        

    


class VOpipeline:
    def __init__(self, Focal, pp, impath):
        self.impath = impath

        fsys_handle = os.listdir(self.impath)
        n_ims = len(fsys_handle)

        self.count = 0
        self.ckpt = 1

        self.pKps = None
        self.pDescs = None
        self.pPts = None
        self.pR = None
        self.pt = None
        self.pProj = None

        self.inlier1, self.inlier2 = [], [] 

        self.FOCAL = Focal
        self.PP = pp
        self.K = np.array([self.FOCAL, 0, self.PP[0], 0, self.FOCAL, self.PP[1], 0, 0, 1]).reshape(3, 3)

        self.xpts = []
        self.ypts = []
        self.zpts = []
        self.X = 0
        self.Y = 0
        self.Z = 0
        
        self.kpmask = None
        self.traj = np.zeros((600,600,3), dtype=np.uint8)
    
    def drawDeltas(self, im, in1, in2):
        dist = []
        for i in range(len(in1)):
            pt1 = (int(in1[i][0]) , int(in1[i][1]))
            pt2 = (int(in2[i][0]) , int(in2[i][1]))
            cv2.line(im, pt1, pt2, color=(0,255,0),thickness=2)
            cv2.circle(im, pt1, 5, color = (0,0,255))
            cv2.circle(im, pt2, 5, color = (255,0,0))
            delta = math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
            dist.append(delta)
        mean = sum(dist)/len(dist)
        org = (int(im.shape[1]-(im.shape[1]*0.3)) , int(im.shape[0]-(im.shape[0]*0.1)))
        im = cv2.putText(im,"Mean delta :{0:.2f} Pixels".format(mean),org,cv2.FONT_HERSHEY_SIMPLEX,0.7,color=(0,0,255),thickness=2) 
        return im, mean 
    
    def isRotationMatrix(self,R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self,R) :
        assert(self.isRotationMatrix(R))  
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])

    def execute(self,ID):
        fptr = self.impath + str(self.count).zfill(6) + ".png"
        frame = cv2.imread(fptr,0)

        detector = cv2.xfeatures2d.SIFT_create(2000)
        kpf, descf = detector.detectAndCompute(frame,None)
        if self.count==0:
            self.pKps = kpf
            self.pDescs = descf
            print(f"CKPT {self.ckpt}")
            self.ckpt+=1
            self.count+=1
            return None, None
        
        bf = cv2.BFMatcher()
        knnmatch = bf.knnMatch(descf, self.pDescs,k=2)

        good = []
        pts1 = []
        pts2 = []
        for m,n in knnmatch:
            if m.distance < 0.8*n.distance:
                good.append(m)
                pts1.append(kpf[m.queryIdx].pt)
                pts2.append(self.pKps[m.trainIdx].pt)

        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        
        F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=0.1, confidence=0.99)
        inlier1 = pts1[mask.ravel()==1]
        inlier2 = pts2[mask.ravel()==1]
        
        self.inlier1 = inlier1
        self.inlier2 = inlier2

        E, emask = cv2.findEssentialMat(inlier1, inlier2, focal=self.FOCAL, pp=self.PP)
        _,R,t,pmask = cv2.recoverPose(E, inlier1, inlier2, focal = self.FOCAL, pp = self.PP)

        if self.count==1:
            print(f"CKPT : {self.ckpt}")
            self.pR = R
            self.pt = t
            self.pProj = np.concatenate((np.dot(self.K,R),np.dot(self.K,t)), axis = 1)
            self.count+=1
            self.ckpt+=1
            return R,t
        
        kpmask = frame.copy()
        kpmask = cv2.cvtColor(kpmask,cv2.COLOR_GRAY2BGR)
        self.kpmask, meanDelta = self.drawDeltas(kpmask,inlier1,inlier2)

        if meanDelta<3:
            print("No motion")
            self.count+=1
            return self.pR, self.pt

        self.pt = self.pt + 0.3*(self.pR.dot(t))
        self.pR = self.pR.dot(R)


        Proj = np.concatenate((np.dot(self.K, self.pR), np.dot(self.K,self.pt)), axis = 1)
        
        trans = self.pt.T[0]
        x, y, z = trans[0], trans[1], trans[2]

        self.X = z
        self.Y = x
        self.Z = -y
        self.xpts.append(x)
        self.ypts.append(z)
        self.zpts.append(-y)

        draw_x, draw_y = int(x)+290, int(z)+90
        cv2.circle(self.traj, (draw_x, draw_y), 1, (self.count*255/4540,255-self.count*255/4540,0), 1)
        cv2.rectangle(self.traj, (10, 20), (600, 60), (0,0,0), -1)

        self.pKps = kpf
        self.pDescs = descf
        self.count+=1
        return self.pR, self.pt
    
 


class Stereo_Driver:
    def __init__(self, seqNo):
        rospy.init_node('camera_driver', anonymous=True)

        self.frame_id = 0
        self.seqNo = seqNo
        self.impathL = f"/home/gautham/Documents/Datasets/dataset/sequences/{str(seqNo).zfill(2)}/image_0/" 
        self.impathR = f"/home/gautham/Documents/Datasets/dataset/sequences/{str(seqNo).zfill(2)}/image_1/" 

        self.pcpub = rospy.Publisher("/Stereo/PointCloud",PointCloud2,queue_size=10)
        self.posepub = rospy.Publisher("/VO/PoseSt",PoseStamped,queue_size=10)

        self.n_frames = 0
        self.outpath = "/home/gautham/ros_env/src/sjtu-drone/data/"

        self.bridge = CvBridge()
        self.rate = rospy.Rate(5)
        self.break_flg = False

        self.frame0 = None
        self.frame1 = None
        self.points = None
        self.colors = None
        self.R = None
        self.t = None

        self.focal = 718.8560
        self.pp  = (607.1928, 185.2157)
        self.baseline = 0.5707/15
        self.clip_threshold = 10
        self.wlsFilterFlag = 0

        self.disparity = None
        self.pointCloud = None

        self.pPcd = o3d.geometry.PointCloud()
        self.pcd = None

        self.VO = VOpipeline(self.focal, self.pp, self.impathL)
        self.ICP = PoseGraphOptimizer(downsamplefactor=0.001)

    def wlsFilterCalc(self,imgL,imgR, min_disp, num_disp, p1, p2):
        sigma = 2.0
        lmbda = 2000.0

        lstereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                numDisparities = num_disp,
                blockSize = 5,
                P1 = p1,
                P2 = p2,
                preFilterCap= 60,
                speckleWindowSize = 3000,
                speckleRange = 1,
                mode=cv2.StereoSGBM_MODE_SGBM_3WAY
            )
        rstereo = cv2.ximgproc.createRightMatcher(lstereo)
        left_disp = lstereo.compute(imgL, imgR)
        right_disp = rstereo.compute(imgR, imgL)

        wls = cv2.ximgproc.createDisparityWLSFilter(lstereo)
        wls.setSigmaColor(sigma)
        filtered_disp = wls.filter(left_disp, imgL, disparity_map_right=right_disp)
        return filtered_disp

    def stereo_core(self):
        if self.frame0 is not None:
            imgL = self.frame0
            imgR = self.frame1
            window_size = 1
            min_disp = 2
            num_disp = 80
            p1 = (8*3*window_size**2)//(2**3)
            p2 = (32*3*window_size**2)//(2**3)
            
            if self.wlsFilterFlag==0:
                stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                    numDisparities = num_disp,
                    blockSize = 7,
                    P1 = p1,
                    P2 = p2,
                    preFilterCap= 100,
                    speckleWindowSize = 3000,
                    speckleRange = 1,
                    mode=cv2.StereoSGBM_MODE_SGBM_3WAY
                )

                disparity = stereo.compute(imgL,imgR)
            else:
                disparity = self.wlsFilterCalc(imgL, imgR, min_disp, num_disp, p1, p2)
        
            norm_disp = cv2.normalize(disparity,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            self.disparity = norm_disp

            h, w = imgL.shape[:2]
            f = self.focal 
            rev_proj_matrix = np.zeros((4,4))
            Q = np.float32([[1, 0, 0, -0.5*w],
                            [0, 1, 0,  -0.5*h], 
                            [0, 0, 0, f], 
                            [0, 0, -1/self.baseline,  0]])


            points = cv2.reprojectImageTo3D(disparity, Q)

            # reflect_matrix = np.identity(3)
            # reflect_matrix[0] *= -1
            # points = np.matmul(points,reflect_matrix)

            colors = cv2.cvtColor(self.frame0, cv2.COLOR_GRAY2RGB)

            mask = self.disparity > ((self.disparity.max() - self.disparity.min()) * (self.clip_threshold/100))
            out_colors = colors[mask]
            out_colors = out_colors.reshape(-1, 3)
            out_points = points[mask]
            idx = np.fabs(out_points[:,0]) < 0.3
            out_points = out_points[idx]
            out_colors = out_colors[idx]

            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(np.array(out_points))
            self.pcd = self.pcd.voxel_down_sample(voxel_size=0.001)
            filtered_points = self.StatisticOutlierRemoval(self.pcd, 300, 1.0)
            #filtered_points = np.array(self.pcd.points)
            self.pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points))

            points_refined = np.zeros_like(filtered_points)
            points_refined[:,2] = filtered_points[:,1]
            points_refined[:,1] = filtered_points[:,2]
            points_refined[:,0] = filtered_points[:,0]
            colors_refined = np.zeros_like(out_points)
            colors_refined[:,2] = out_colors[:,1]
            colors_refined[:,1] = out_colors[:,2]
            colors_refined[:,0] = out_colors[:,0]
            self.points = self.orthogonalTransform(filtered_points,0,0,0)
            self.colors = colors_refined
        else:
            print("NULL Disparity")
    
    def StatisticOutlierRemoval(self,cloud, n_neigh, std_ratio):
        cl, ind = cloud.remove_statistical_outlier(nb_neighbors=n_neigh, std_ratio=std_ratio)
        inlier_cloud = cloud.select_by_index(ind)
        return np.array(inlier_cloud.points)

    def transformCloud(self, cloud, R, t):
        cloud.translate((t[0], t[1], t[2]))
        cloud.rotate(R, center=(0, 0, 0))
        self.pcd = cloud
    
    def orthogonalTransform(self, points, rotx, roty, rotz):
        points_refined = np.zeros_like(points)
        points_refined[:,0] = points[:,0]
        points_refined[:,2] = points[:,1]
        points_refined[:,1] = points[:,2]
        return points_refined

    def run(self):
        while True:
            self.frame0 = cv2.imread(self.impathL + str(self.frame_id).zfill(6) + ".png" ,0)
            self.frame1 = cv2.imread(self.impathR + str(self.frame_id).zfill(6) + ".png",0)
            
            if self.frame0 is not None:
                R,t = self.VO.execute(self.frame_id)
                if self.VO.kpmask is not None:
                    cv2.imshow("Frame1",self.VO.kpmask)

                pose_msg = PoseStamped()

                if (R is not None):
                    rots = self.VO.rotationMatrixToEulerAngles(R)
                    rot = Rotation.from_matrix(R)
                    rot_q = rot.as_quat()

                    pose_msg.pose.position.x = self.VO.Y
                    pose_msg.pose.position.y = self.VO.X
                    pose_msg.pose.position.z = self.VO.Z

                    pose_msg.pose.orientation.x = rot_q[2]
                    pose_msg.pose.orientation.y = rot_q[0]
                    pose_msg.pose.orientation.z = -rot_q[1]
                    pose_msg.pose.orientation.w = rot_q[3]
                    
                    Rtx = Rotation.from_quat([rot_q[2], rot_q[0], rot_q[1], rot_q[3]])
                    Tmat = np.array([self.VO.Y, self.VO.X, self.VO.Z])
                    Rmat = Rtx.as_matrix()
                    self.R, self.t = Rmat, Tmat
                else:
                    pose_msg.pose.position.x = 0
                    pose_msg.pose.position.y = 0
                    pose_msg.pose.position.z = 0

                    pose_msg.pose.orientation.x = 0
                    pose_msg.pose.orientation.y = 0
                    pose_msg.pose.orientation.z = 0
                    pose_msg.pose.orientation.w = 1
                    
                    Rtx = Rotation.from_quat([0, 0, 0, 1])
                    Tmat = np.array([0, 0, 0])
                    Rmat = Rtx.as_matrix()
                    self.R, self.t = Rmat, Tmat
                
                self.stereo_core()
                Ricp, Ticp = self.ICP.processAndUpdate(self.pcd)

                print("VO tvec :\n{}\nICP tvec:\n{}\n".format(t, Ticp))

                cv2.imshow("disparity",self.disparity)
                cv2.imshow("trajectory",self.VO.traj)
                h = Header()
                h.stamp = rospy.Time.now()
                h.frame_id = "map"
                pose_msg.header = h
                scaled_points = pcl2.create_cloud_xyz32(h,self.points*100)
                self.pcpub.publish(scaled_points)
                self.posepub.publish(pose_msg)

                k = cv2.waitKey(100)
                if k%256 == 27:
                    print("Escape hit, closing...")
                    break

                elif k%256==32:
                    print("Writing frames {}".format(self.n_frames))
                self.n_frames+=1
                self.frame_id+=1
            else:
                print("NULL FRAME, {}.png".format(self.impathL + str(self.frame_id).zfill(6)))
                break
            if self.break_flg:
                break
            self.rate.sleep()
        self.plotter(self.VO.xpts, self.VO.ypts, self.VO.zpts)
        self.plotter(self.ICP.xpts, self.ICP.ypts, self.ICP.zpts)
        cv2.destroyAllWindows()
    
    def plotter(self, xpts, ypts, zpts):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(xpts, ypts, zpts, c="b")
        ax.scatter3D(xpts, ypts, zpts, c= zpts, cmap='plasma')
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        plt.show()

if __name__ == "__main__":
    rig1 = Stereo_Driver(7)
    rig1.run()
