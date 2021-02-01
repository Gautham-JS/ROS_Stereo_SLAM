#!/usr/bin/env python
import numpy as np
import cv2
from numpy.linalg import inv, pinv
import math
import matplotlib.pyplot as plt
from matplotlib import style
import open3d as o3d
from scipy.spatial.transform import Rotation

import rospy
from math import pow, atan2, sqrt, sin, cos
from std_msgs.msg import Float64, Float64MultiArray, Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from sensor_msgs import point_cloud2 as pcl2
from geometry_msgs.msg import Pose, PoseStamped
from cv_bridge import CvBridge

#style.use("ggplot")


K = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
              [0, 7.188560000000e+02, 1.852157000000e+02],
              [0, 0, 1]])


def getTruePose():
    file = '/home/gautham/Documents/Datasets/dataset/poses/00.txt'
    return np.genfromtxt(file, delimiter=' ',dtype=None)

class PoseGraphOptimize:
    def __init__(self):
        self.pcds = []
        self.downSampleFactor = 1.0

        self.max_correspondence_distance_coarse = self.downSampleFactor *15
        self.max_correspondence_distance_fine = self.downSampleFactor *1.5

        self.poseGraph = o3d.pipelines.registration.PoseGraph()

        self.odometry = np.identity(4)
        self.poseGraph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    def addPoseToGraph(self,source, target, R, t, fromNodeID, toNodeID, loopClosure=False, loopClosureNode=None):
        odometry = np.identity(4)
        odometry[:3,:3] = R
        odometry[:3,-1:] = t
        infoMatrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source,target,self.max_correspondence_distance_coarse, odometry)

        if not loopClosure:
            self.poseGraph.nodes.append(o3d.pipelines.registration.PoseGraphNode( np.linalg.inv(odometry) ))
            self.poseGraph.edges.append(o3d.pipelines.registration.PoseGraphEdge(fromNodeID, toNodeID, odometry, infoMatrix, uncertain=False))
        else:
            self.poseGraph.edges.append(o3d.pipelines.registration.PoseGraphEdge(fromNodeID, toNodeID, odometry, infoMatrix, uncertain=True))
        
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


class FeatureVocabulary:
    def __init__(self):
        self.index = -1
        self.kps = None
        self.descs = None



class VisOdometry:
    def __init__(self, K, absPath, seqNo, baseline, scaleFactor = 0, stereoFlag=False):
        rospy.init_node('camera_driver', anonymous=True)
        self.K = K

        self.absPath = absPath
        self.seqNo = seqNo
        self.baseline = baseline

        self.lFptr = absPath + str(seqNo).zfill(2) + "/image_0/"
        self.rFptr = absPath + str(seqNo).zfill(2) + "/image_1/"

        self.scaleFactor = scaleFactor
        self.stereoFlag = False

        self.bridge = CvBridge()
        self.rate = rospy.Rate(5)
        self.break_flg = False
        self.pcpub = rospy.Publisher("/Stereo/PointCloud",PointCloud2,queue_size=10)
        self.posepub = rospy.Publisher("/VO/PoseSt",PoseStamped,queue_size=10)
        self.pathpub = rospy.Publisher("/VO/Path", Path, queue_size=10)

        self.path = []

        self.referenceImage = None
        self.referencePoints2D = None
        self.referencePoints3D = None

        self.currentImage = None
        self.p3ds = np.array([])

        self.objectPointsPnP = None
        self.framePointsPnP = None
        self.trigger = False

        self.canvas = np.zeros((600, 600, 3), dtype=np.uint8)
        self.ply = o3d.geometry.PointCloud()

        self.iteration, self.relocIter, self.pointDensity , self.relocalizations = (list() for l in range(4))
        self.xpos, self.ypos, self.zpos = (list() for l in range(3))
        self.xtrue, self.ytrue, self.ztrue = (list() for l in range(3))
        self.x3d, self.y3d, self.z3d = (list() for l in range(3))


    def drawDeltas(self, im, in1, in2):
        dist = []
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
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

    def trackPointsFrame2Frame(self,RefIm, CurIm, refPts, ref3dPts):
        lkparams = dict(winSize=(21,21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        trackPts , Idx, err = cv2.calcOpticalFlowPyrLK(RefIm, CurIm, refPts, None, **lkparams)
        Idx = Idx.reshape(Idx.shape[0])

        refInliers = refPts[Idx==1]
        trackInliers = trackPts[Idx==1]
        pts3dInliers = ref3dPts[Idx==1]

        return pts3dInliers, refInliers, trackInliers

    def removeDuplicate(self,queryPoints, refPoints, radius=5):
        for i in range(len(queryPoints)):
            query = queryPoints[i]
            xliml, xlimh = query[0]-radius, query[0]+radius
            yliml, ylimh = query[1]-radius, query[1]+radius
            inside_x_lim_mask = (refPoints[:,0] > xliml) & (refPoints[:,0] < xlimh)
            curr_kps_in_x_lim = refPoints[inside_x_lim_mask]

            if curr_kps_in_x_lim.shape[0] != 0:
                inside_y_lim_mask = (curr_kps_in_x_lim[:,1] > yliml) & (curr_kps_in_x_lim[:,1] < ylimh)
                curr_kps_in_x_lim_and_y_lim = curr_kps_in_x_lim[inside_y_lim_mask,:]
                if curr_kps_in_x_lim_and_y_lim.shape[0] != 0:
                    queryPoints[i] =  np.array([0,0])
        return (queryPoints[:, 0]  != 0 )

    def sparseStereoTriangulate(self, lImg, rImg, refPts=None):
        detector = cv2.xfeatures2d.SIFT_create(2000)
        #detector = cv2.AKAZE_create(1000)
        kpL, descL = detector.detectAndCompute(lImg, None)
        kpR, descR = detector.detectAndCompute(rImg, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict()   
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        bf = cv2.BFMatcher()
        matches = flann.knnMatch(descL,descR,k=2)

        pts1, pts2 = [], []

        match_points1, match_points2 = [], []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                pts1.append(kpL[m.queryIdx].pt)
                pts2.append(kpR[m.trainIdx].pt)
        
        pts1 = np.array(pts1).astype(float)
        pts2 = np.array(pts2).astype(float)

        P1 = self.K.dot( np.hstack((np.eye(3), np.zeros((3,1)) )) )
        P2 = self.K.dot(np.hstack((np.eye(3), np.array([[-self.baseline,0, 0]]).T)))

        if refPts is not None:
            mask = self.removeDuplicate(pts1, refPts)
            pts1 = pts1[mask,:]
            pts2 = pts2[mask,:] 

        pt1_flip = np.vstack((pts1.T,np.ones((1,pts1.shape[0]))))
        pt2_flip = np.vstack((pts2.T,np.ones((1,pts2.shape[0]))))

        pts4D = cv2.triangulatePoints(P1, P2, pt1_flip[:2], pt2_flip[:2])
        pts3D = pts4D/pts4D[3]
        pts3D = pts3D[:3]
        return  pts3D.T ,pts1
    
    def StatisticOutlierRemoval(self, cloud, n_neigh, std_ratio):
        cl, ind = cloud.remove_statistical_outlier(nb_neighbors=n_neigh, std_ratio=std_ratio)
        inlier_cloud = cloud.select_by_index(ind)
        return inlier_cloud
    
    def initializeSequence(self, lImg, rImg):
        pts3d, lpts = self.sparseStereoTriangulate(lImg, rImg)
        lpts = lpts.astype("float32")

        self.objectPointsPnP = np.expand_dims(pts3d, axis = 2)
        self.framePointsPnP = np.expand_dims(lpts, axis=2).astype(float)

        self.referenceImage = lImg
        self.referencePoints2D = lpts
        self.referencePoints3D = pts3d

        self.p3ds = np.zeros_like(pts3d)

    def trackSequence(self):
        curIm = cv2.imread(self.lFptr + str(0).zfill(6) + ".png", 0)
        curImR = cv2.imread(self.rFptr + str(0).zfill(6) + ".png", 0)

        self.initializeSequence(curIm, curImR)

        truePose = getTruePose()

        for i in range(1,4000):
            curIm = cv2.imread(self.lFptr + str(i).zfill(6) + ".png", 0)

            print(self.referencePoints2D.shape, self.referencePoints3D.shape)
            
            self.referencePoints3D, self.referencePoints2D, tracked2Dpts = self.trackPointsFrame2Frame(self.referenceImage, curIm, self.referencePoints2D, self.referencePoints3D)
            frame, _ = self.drawDeltas(curIm, tracked2Dpts, self.referencePoints2D)
            self.objectPointsPnP = np.expand_dims(self.referencePoints3D, axis=2)
            self.framePointsPnP = np.expand_dims(tracked2Dpts, axis=2).astype(float)

            _, rvec, tvec, inlierMap = cv2.solvePnPRansac(self.objectPointsPnP, self.framePointsPnP, self.K,  None)
            self.referencePoints2D = tracked2Dpts[inlierMap[:,0],:]
            self.referencePoints3D = self.referencePoints3D[inlierMap[:,0],:]

            R,_ = cv2.Rodrigues(rvec)
            tvec = -R.T.dot(tvec)

            inv_transform = np.hstack((R.T, tvec))

            inlierThresh = len(inlierMap)/len(self.objectPointsPnP)
            #print(inlierThresh)

            #if inlierThresh<0.99 or len(self.referencePoints2D)<50:
            print(f"REINITIALIZING AT INDEX {i}")
            curImR = cv2.imread(self.rFptr + str(i).zfill(6) + ".png", 0)
            pts3d_updated, lpts_updated = self.sparseStereoTriangulate(curIm, curImR, refPts = self.referencePoints2D)
            lpts_updated = lpts_updated.astype("float32")

            pts3d_transformed = inv_transform.dot(np.vstack((pts3d_updated.T, np.ones((1,pts3d_updated.shape[0])))))

            pts3d_valid = pts3d_transformed[2,:] > 0
            pts3d_updated = pts3d_transformed[:, pts3d_valid]

            self.referencePoints2D = np.vstack( (self.referencePoints2D, lpts_updated[pts3d_valid,:]) )
            self.referencePoints3D = np.vstack( (self.referencePoints3D, pts3d_updated.T) )
            self.trigger = True
            self.referenceImage = curIm
        
            self.xpos.append(tvec[0])
            self.ypos.append(tvec[2])
            self.zpos.append(tvec[1])

            pose_msg = PoseStamped()


            rot = Rotation.from_matrix(R)
            rot_q = rot.as_quat()

            #rot_q /= rot_q[3]

            pose_msg.pose.position.x = tvec[0]*self.scaleFactor
            pose_msg.pose.position.y = tvec[2]*self.scaleFactor
            pose_msg.pose.position.z = -1*tvec[1]*self.scaleFactor
            pose_msg.pose.orientation.x = rot_q[0]
            pose_msg.pose.orientation.y = rot_q[2]
            pose_msg.pose.orientation.z = rot_q[1]
            pose_msg.pose.orientation.w = rot_q[3]

            h = Header()
            h.stamp = rospy.Time.now()
            h.frame_id = "map"
            pose_msg.header = h
            self.path.append(pose_msg)
            pathMsg = Path()
            pathMsg.header = h
            pathMsg.poses = self.path
            
            self.pathpub.publish(pathMsg)
            self.posepub.publish(pose_msg)

            self.xtrue.append(truePose[i][3])
            self.ytrue.append(truePose[i][11])
            self.ztrue.append(truePose[i][7])

            self.iteration.append(i)
            self.pointDensity.append(self.referencePoints3D.shape[0])

            p3d = np.zeros_like(self.referencePoints3D)
            p3d[:,0] = self.referencePoints3D[:,0]
            p3d[:,1] = self.referencePoints3D[:,2]
            p3d[:,2] = self.referencePoints3D[:,1]*-1



            self.ply.points = o3d.utility.Vector3dVector(p3d)

            self.ply = self.StatisticOutlierRemoval(self.ply, 200, 1.0)
            p3d = np.array(self.ply.points)
            
            if self.trigger:
                self.relocalizations.append(self.referencePoints3D.shape[0])
                self.relocIter.append(i)
                Idx = np.random.randint(0,len(p3d), size=3)
                p3d_inlier = p3d[Idx]
                rosIdx = np.random.randint(0,len(p3d), size=3)
                rosP3d = p3d[rosIdx]

                #print(len(rosP3d))

                #self.p3ds = np.append(self.p3ds, p3d, 0)

                self.x3d.extend(p3d_inlier[:,0])
                self.y3d.extend(p3d_inlier[:,1])
                self.z3d.extend(p3d_inlier[:,2])
            
            scaled_points = pcl2.create_cloud_xyz32(h, self.p3ds*self.scaleFactor)
            self.pcpub.publish(scaled_points)
        
            

            #print(tvec.T)

            draw_x, draw_y = int(tvec[0]) + 300, int(tvec[2]) + 100;
            cv2.circle(self.canvas, (draw_x, draw_y) ,1, (0,0,255), 2);
            cv2.rectangle(self.canvas, (10, 30), (550, 50), (0,0,0), cv2.FILLED);

            self.rate.sleep()
            cv2.imshow( "Trajectory", self.canvas );
            cv2.imshow("f", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                self.visualization()
                break
        
    def visualization(self):
        fig = plt.figure()
        ax = plt.axes()

        ax = fig.add_subplot(1,2,1, projection="3d")
        ax.plot3D(np.array(self.xpos).flatten(), np.array(self.ypos).flatten(), np.array(self.zpos).flatten(), c='r', label="Precicted Trajectory", linewidth=2)
        ax.plot3D(np.array(self.xtrue).flatten(), np.array(self.ytrue).flatten(), np.array(self.ztrue).flatten(), c='g', label="True Trajectory", linewidth=2)

        ax.scatter3D(np.array(self.x3d).flatten(), np.array(self.y3d).flatten(), np.array(self.z3d).flatten(), c=self.z3d, cmap="gnuplot2", label="3D estimates")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")

        ax.set_zlim(max(self.z3d)*4, max(self.z3d)*-4)
        ax.legend()
        
        ax = fig.add_subplot(1,2,2)
        ax.plot(np.array(self.iteration).flatten(), np.array(self.pointDensity).flatten(), label='Point Density')
        ax.scatter(np.array(self.relocIter).flatten(), np.array(self.relocalizations).flatten(), marker="x", c='r', label="Relocalizations")
        #ax.set_ylim(max(self.pointDensity)*4,max(self.pointDensity)*-4)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":
    path = "/home/gautham/Documents/Datasets/dataset/sequences/"
    VO = VisOdometry(K, absPath=path, seqNo=0, baseline=0.54, scaleFactor=0.1)
    VO.trackSequence()









