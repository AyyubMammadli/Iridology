# Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np
import cv2
import torch
from skimage import io
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, ChebConv, GATv2Conv, SAGEConv, global_max_pool, global_mean_pool
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__, template_folder='templates',static_url_path='/static')

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, k_size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=k_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=k_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, k_size):
        super().__init__()

        self.conv = conv_block(in_c, out_c, k_size)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p
    
class GCNBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.gcn1 = SAGEConv(in_channels, hidden_channels)
        self.gcn2 = SAGEConv(hidden_channels, 128)


    def forward(self, p4):
        batch_size, num_channels, height, width = p4.size()

        # Reshape p4 to form node features for the graph
        x = p4.view(batch_size, num_channels, -1).permute(0, 2, 1)  # [batch_size, height*width, num_channels]
        # Create k-nearest neighbor graph
        graphs = []
        
        for g in x:
            edge_index = knn_graph(g, k=8, loop=False)

            # Pass the node features through the GCN layer
            out = self.gcn1(g, edge_index)
            out = F.relu(out)
            out = self.gcn2(out, edge_index)
            
            # Reshape back to the original shape

            graphs.append(out)
            
        graphs = torch.stack(graphs, dim=0)
        return graphs
    

class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 32, 7)
        self.e2 = encoder_block(32, 64, 5)
        self.e3 = encoder_block(64, 128, 5)
        self.e4 = encoder_block(128, 256, 3)
        self.e5 = encoder_block(256, 512, 3)
        
        
        """ Bottleneck """
#         self.b = conv_block(512, 1024)
        self.gcn_bottleneck = GCNBottleneck(512, 256)


        """ Classifier """
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=0.2)
        self.outputs = nn.Linear(64, 1)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        s5, p5 = self.e5(p4)
        

        """ Bottleneck """
#         b = self.b(p4)
#         print("P4: ",p4.shape)
        gcn_output = self.gcn_bottleneck(p5)
#         print("GCN OUTPUT",gcn_output.shape)
        node_size = gcn_output.shape[1]
        batch_tensor = torch.zeros(node_size, dtype=torch.int64)
        batch_tensor = batch_tensor.to(inputs.device)
        
        pooled = []
        for g in gcn_output:
            pooled_output = global_mean_pool(g, batch_tensor)
            pooled.append(pooled_output)
            
        pooled = torch.stack(pooled, dim=0).reshape(-1, 128)
#         print("pooled",pooled.shape)
        outputs = self.fc1(pooled)
        outputs = self.dropout(outputs)
#         print("FC1 ",outputs.shape)
        outputs = F.relu(outputs)
        outputs = self.outputs(outputs)
    

        return outputs
    



# Define function for resizing image
def resize_image(image_array, target_size=(1200, 986)):
    image = Image.fromarray(image_array)
    resized_image = image.resize(target_size)
    return np.array(resized_image)

# Define function for prediction using Pytorch model
def predict_image(image_array):

    # Preprocess the image
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image_array = image_array.astype(np.float32) / 255.0  # Normalize
    image_tensor = torch.tensor(image_array)
    # Make prediction

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet()
    
    model.load_state_dict(torch.load(f"checkpoint_13.pth", map_location=device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        output = torch.sigmoid(model(image_tensor))

        print(output)
        prediction = torch.round(output)  # Round to 0 or 1
    
    # Here, you would return the class label or any other relevant information based on your model's output
    return prediction.item()


# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for image processing
@app.route('/process_image', methods=['POST'])
def process_image():
    # Get image from request
    image_file = request.files['image']
    image_bytes = image_file.read()
    image_array = np.array(Image.open(BytesIO(image_bytes)))
    
    # Resize the image
    resized_image = resize_image(image_array)
    
    # Perform prediction on resized image
    prediction = predict_image(resized_image)
    
    # Display the result on the website
    return render_template('result.html', prediction=prediction)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
