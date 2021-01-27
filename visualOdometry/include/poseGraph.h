#ifndef POSE_GRAPH_H
#define POSE_GRAPH_H

#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>

#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/edge_se3.h"
#include "g2o/stuff/sampler.h"
#include "g2o/stuff/command_args.h"
#include "g2o/core/factory.h"


#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>

typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;

using namespace std;
using namespace g2o;



class globalPoseGraph{
    public:
        int globalNodeID = 0;
        bool loopClosureFlag = false;

        VertexSE3* prevVertex;
        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();

        vector<VertexSE3*> vertices;
        vector<EdgeSE3*> odometryEdges;
        vector<EdgeSE3*> edges;
        
        // SlamLinearSolver* linearSolver;
        // OptimizationAlgorithmLevenberg* solver;
        // SlamBlockSolver* solverPtr;
        SparseOptimizer optimizer;
        OptimizationAlgorithm* solver = new g2o::OptimizationAlgorithmGaussNewton(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
        );
        string outFileName = "poseGraph.g2o";
    
    globalPoseGraph(){  
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(true); 
    }   

    void initializeGraph();
    void augmentNode(Eigen::Isometry3d localT, Eigen::Isometry3d globalT);
    void addLoopClosure(Eigen::Isometry3d T, int fromID);
    vector<Eigen::Isometry3d> globalOptimize();
    void saveStructure();
};

void globalPoseGraph::initializeGraph(){
    VertexSE3* v = new VertexSE3;
    v->setId(globalNodeID);

    Eigen::Isometry3d t = Eigen::Isometry3d::Identity();

    cerr<<"Initial transformation : "<<t.matrix()<<endl;
    v->setFixed(true);

    v->setEstimate(t);
    vertices.emplace_back(v);
    optimizer.addVertex(v);

    prevVertex = v; 
    globalNodeID++;
}


void globalPoseGraph::augmentNode(Eigen::Isometry3d localT, Eigen::Isometry3d globalT){
    VertexSE3* prev = prevVertex;
    EdgeSE3* e = new EdgeSE3();
    VertexSE3* cur = new VertexSE3();

    cur->setId(globalNodeID);
    cur->setEstimate(globalT);
    cur->setMarginalized(false);
    cur->setFixed(false);
    Eigen::Isometry3d t = prev->estimate().inverse() * cur->estimate();
    //cerr<<"break1"<<endl;
    e->setVertex(0, prev);
    //cerr<<"break2"<<endl;
    e->setVertex(1, cur);
    //cerr<<"break3"<<endl;
    
    e->setMeasurement(t);
    //cerr<<"break4"<<endl;
    //e->setInformation(information);
    optimizer.addVertex(cur);
    optimizer.addEdge(e);

    odometryEdges.emplace_back(e);
    //edges.emplace_back(e);
    prevVertex = cur;
    vertices.emplace_back(cur);
    globalNodeID++;
}

void globalPoseGraph::addLoopClosure(Eigen::Isometry3d T, int fromID){
    VertexSE3* cur = vertices[fromID];
    EdgeSE3* e = new EdgeSE3;
    VertexSE3* prev = prevVertex;
    //Eigen::Isometry3d t = prev->estimate().inverse() * cur->estimate();
    Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
    e->setVertex(0, prev);
    e->setVertex(1, cur);
    e->setMeasurement(t);
    //e->setInformation(information);
    odometryEdges.emplace_back(e);
    edges.emplace_back(e);
    optimizer.addEdge(e);
    //prevVertex = cur;
    //vertices.emplace_back(cur);
    //globalNodeID++;
}

vector<Eigen::Isometry3d> globalPoseGraph::globalOptimize(){
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    optimizer.save("result.g2o");

    vector<Eigen::Isometry3d> out;
    for(VertexSE3* v: vertices){
        out.emplace_back(v->estimate());
    }
    return out;
}

void globalPoseGraph::saveStructure(){
    std::ofstream fileOutputStream;
    if (outFileName != "-") {
        cerr << "Writing into " << outFileName << endl;
        fileOutputStream.open(outFileName.c_str());
    } else {
        cerr << "writing to stdout" << endl;
    }
    cerr<<"b0"<<endl;
    string vertexTag = Factory::instance()->tag(vertices[0]);
    string edgeTag = Factory::instance()->tag(odometryEdges[0]);
    cerr<<"b1"<<endl;
    ostream& fout = outFileName != "-" ? fileOutputStream : cout;
    for (size_t i = 0; i < vertices.size(); ++i) {
        VertexSE3* v = vertices[i];
        fout << vertexTag << " " << v->id() << " ";
        v->write(fout);
        fout << endl;
    }
    cerr<<"b2"<<endl;

    for (size_t i = 0; i < odometryEdges.size(); ++i) {
        EdgeSE3* e = odometryEdges[i];
        VertexSE3* from = static_cast<VertexSE3*>(e->vertex(0));
        VertexSE3* to = static_cast<VertexSE3*>(e->vertex(1));
        fout << edgeTag << " " << from->id() << " " << to->id() << " ";
        e->write(fout);
        fout << endl;
    }
    cerr<<"b3"<<endl;
    // for (size_t i = 0; i < edges.size(); ++i) {
    //     EdgeSE3* e = edges[i];
    //     VertexSE3* from = static_cast<VertexSE3*>(e->vertex(0));
    //     VertexSE3* to = static_cast<VertexSE3*>(e->vertex(1));
    //     fout << edgeTag << " " << from->id() << " " << to->id() << " ";
    //     e->write(fout);
    //     fout << endl;
    // }
    cerr<<"SAVED "<<outFileName<<endl;
}


#endif