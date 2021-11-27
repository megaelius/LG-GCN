import h5py
import torch
from utils.pc_viz import show_pointclouds

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

        self.lines = vtk.vtkCellArray()

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
        else:
            r = random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkPolyData.GetPointData().SetScalars(self.colors)
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()


    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')

if __name__ == '__main__':
    filename = "data/ply_data_all_0.h5"

    model = torch.load('mlp3_best.pth',map_location='cpu')
    print(model['state_dict']['module.graph_mlp.0.weight'], model['state_dict']['module.graph_mlp.0.bias'])

    mlp = torch.nn.Linear(9,9)
    mlp.weight.data = model['state_dict']['module.graph_mlp.0.weight']
    mlp.bias.data = model['state_dict']['module.graph_mlp.0.bias']

    knn = DenseKnnGraph(16)

    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = torch.Tensor(f[a_group_key])
        print(data.shape)
        print(data[0,0,:])
        print(mlp(data[0,0,:]))
        print(data[0,:,:3].numpy().shape)
        print(data[0,:,:3].unsqueeze(0).unsqueeze(-1).transpose(2,1).shape)

        #data2, knn_index = knn(data[0,:,:3].unsqueeze(-1).transpose(2,1))
        #print(knn_index.shape)
        pointCloud = VtkPointCloud()
        for j in range(data.shape[1]):
            pointCloud.addPoint(list(data[0,j,0:3]),list(255*data[0,j,3:6]))
            #pointCloud.addPoint(list(data[999,j,0:3]),list(255*data[999,j,3:6]))
        pointCloud.addPoint([0,0,0],[0,0,0])
        pointCloud.addPoint([0,0,0],[0,0,0])
        pointCloud.addPoint([0,0,0],[0,0,0])
        pointCloud.addPoint([0,0,0],[0,0,0])

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
