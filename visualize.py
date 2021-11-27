import h5py
import torch
from utils.pc_viz import show_pointclouds
from architecture import CustomDenseDeepGCN

import vtk
from numpy import random
from gcn_lib.dense.torch_edge import DenseKnnGraph

class VtkPointCloud:

    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()

        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetNumberOfComponents(3)
        self.colors.SetName("Colors")

        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)

    def addPoint(self, point, color):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.colors.InsertNextTuple(color)
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
            self.vtkActor.GetProperty().SetPointSize(4)
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkPolyData.GetPointData().SetScalars(self.colors)
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def addLine(self, id1, id2):
        self.vtkLines.InsertNextCell(2, [id1, id2])

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkLines = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.SetLines(self.vtkLines)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')

def dense_knn_to_set(knn_index):
    edges = set()
    for i in range(knn_index.shape[2]):
        for j in range(knn_index.shape[3]):
            l = list(knn_index[:,:,i,j])
            tuple = (int(l[0]),int(l[1]))
            edges.add(tuple)
    return edges

if __name__ == '__main__':
    filename = "data/ply_data_all_0.h5"

    checkpoint2 = torch.load('weights/mlp2_best.pth',map_location='cpu')
    checkpoint3 = torch.load('weights/mlp3_best.pth',map_location='cpu')

    mlp3 = torch.nn.Linear(9,9)
    mlp3.weight.data = checkpoint3['state_dict']['module.graph_mlp.0.weight']
    mlp3.bias.data = checkpoint3['state_dict']['module.graph_mlp.0.bias']

    mlp2 = torch.nn.Sequential(torch.nn.Linear(9,32),
                              torch.nn.ReLU(),
                              torch.nn.Linear(32,9))
    mlp2[0].weight.data = checkpoint2['state_dict']['module.graph_mlp.0.weight']
    mlp2[0].bias.data = checkpoint2['state_dict']['module.graph_mlp.0.bias']
    mlp2[2].weight.data = checkpoint2['state_dict']['module.graph_mlp.2.weight']
    mlp2[2].bias.data = checkpoint2['state_dict']['module.graph_mlp.2.bias']

    knn = DenseKnnGraph(16)

    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = torch.Tensor(f[a_group_key])
        pointCloud = VtkPointCloud()

        #Add points
        for j in range(data.shape[1]):
            pointCloud.addPoint(list(data[1,j,0:3]),list(255*data[1,j,3:6]))
        #Add graph lines
        #(batch_size, num_dims, num_points, 1)
        _, knn_index = knn(data[1,:,:].unsqueeze(0).unsqueeze(-1).transpose(2,1))
        _, knn_index_mlp2 = knn(mlp2(data[1,:,:]).unsqueeze(0).unsqueeze(-1).transpose(2,1))
        _, knn_index_mlp3 = knn(mlp3(data[1,:,:]).unsqueeze(0).unsqueeze(-1).transpose(2,1))
        edges = dense_knn_to_set(knn_index)
        edges_mlp2 = dense_knn_to_set(knn_index_mlp2)
        edges_mlp3 = dense_knn_to_set(knn_index_mlp3)
        diff1 = edges.difference(edges_mlp2)
        diff2 = edges_mlp2.difference(edges)
        diff3 = edges_mlp2.difference(edges_mlp3)
        for edge in diff3:
            pointCloud.addLine(edge[0],edge[1])

        # Renderer
        renderer = vtk.vtkRenderer()
        renderer.AddActor(pointCloud.vtkActor)
        renderer.SetBackground(.2, .3, .4)
        renderer.ResetCamera()

        # Render Window
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)

        # Interactor
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        # Begin Interaction
        renderWindow.Render()
        renderWindowInteractor.Start()
        #o3d.draw_geometries([data[i,:,:3].numpy() for i in range(data.shape[0])])
        #show_pointclouds([data[i,:,:3].numpy() for i in range(data.shape[0])], [255*data[i,:,3:6].numpy() for i in range(data.shape[0])],text = [str(i) for i in range(data.shape[0])], interactive=True)
