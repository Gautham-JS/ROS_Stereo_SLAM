#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/edge_xyz_prior.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/vertex_pointxyz.h"
#include "g2o/types/slam3d/edge_se3_pointxyz.h"
//#include "edge_se3exp_pointxyz_prior.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"

using namespace std;
using namespace  Eigen;


static double uniform_rand(double lowerBndr, double upperBndr){
    return lowerBndr + ((double) std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}

static double gauss_rand(double mean, double sigma){
    double x, y, r2;
    do {
        x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
        y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
        r2 = x * x + y * y;
    } while (r2 > 1.0 || r2 == 0.0);
    return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
}

double uniform(){
    return uniform_rand(0., 1.);
}

struct CamExtrinsic{
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
};

vector<CamExtrinsic> genPose(int siz){
    vector<CamExtrinsic> poses;
    for(int i=0; i<siz; i++){
        poses.emplace_back(CamExtrinsic{Eigen::Matrix3d::Identity(3,3), Eigen::Vector3d(0,0,i)});
    }
    return poses;
}

int main(){
    
    vector<CamExtrinsic> poses = genPose(10);

    g2o::SparseOptimizer optimizer;
    auto linearSolverType = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolverPL<6,1>::PoseMatrixType>>();

    auto solver = g2o::make_unique<g2o::BlockSolverPL<6,1>>(std::move(linearSolverType));

    g2o::OptimizationAlgorithmLevenberg* LMAlgo = new g2o::OptimizationAlgorithmLevenberg(std::move(solver));
    optimizer.setAlgorithm(LMAlgo);

    vector<g2o::VertexSE3Expmap*> se3Vertices;
    int ID=0;
    for(auto pose : poses){
        g2o::SE3Quat curPose(pose.R, pose.t);
        g2o::VertexSE3Expmap* vertex;

        vertex->setId(ID);
        vertex->setFixed(false);

        curPose.setTranslation(curPose.translation()+2.0f*Vector3d(uniform(), uniform(), uniform()));

        vertex->setEstimate(curPose);
        se3Vertices.emplace_back(vertex);
        optimizer.addVertex(vertex);
        ID++;

        g2o::EdgeSE3Expmap* constraint = new g2o::EdgeSE3Expmap();
    }   
}

