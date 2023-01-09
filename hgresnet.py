import networkx as nx
from lib.ssn.ssn import ssn_iter, sparse_ssn_iter
import timm
import torch
from torch import nn
import torch.nn.functional as F

def make_adj_matrix(num_row, num_col):
    adj_matrixs = []
    directions = [[i,j] for i in [-1,0,1] for j in [-1,0,1]]
    for direction in directions:
        G = nx.DiGraph()
        G.add_nodes_from([(i,j) for i in range(num_row) for j in range(num_col)])
        for i in range(num_row):
            for j in range(num_col):
                if (i,j) in G and (i+direction[0], j+direction[1]) in G:
                    G.add_edge((i,j),(i + direction[0], j+direction[1]))
        adj_matrixs.append(torch.tensor(nx.to_numpy_array(G), requires_grad=False)[None])
    return torch.concat(adj_matrixs).float()

class HGResidualBlock(nn.Module):
    def __init__(self, input_dims, num_groups):
        super().__init__()
        self.input_dims = input_dims
        self.num_groups = num_groups

        self.activation = nn.ReLU()
        self.dense1 = nn.Linear(input_dims, input_dims, bias=False)
        self.graph_conv_1 = nn.Linear(9*input_dims, 9*input_dims, bias=False)
        self.dense2 = nn.Linear(input_dims, input_dims, bias=False)

        self.bn1 = nn.BatchNorm1d(num_groups)
        self.bn2 = nn.BatchNorm1d(num_groups)
        self.bn3 = nn.BatchNorm1d(num_groups)

    def graph_conv(self, x, coef_matrix):
        # coef_matrix * x * W
        expand_x = torch.tile(x, (1,1,9))
        out = self.graph_conv_1(expand_x)
        out = out.reshape(-1, 9, self.num_groups, self.input_dims)
        out = coef_matrix @ out
        out = out.sum(dim=1)
        return out

    def forward(self, inp, coef_matrix):
        x = self.dense1(inp)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.graph_conv(x, coef_matrix)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.dense2(x)
        x = self.bn3(x)
        x = x + inp
        x = self.activation(x)
        return x

class HGBlock(nn.Module):
    def __init__(self, input_shape, num_groups, num_iter, adj_mats):
        super().__init__()
        self.num_groups = num_groups
        self.num_iter = num_iter
        self.input_shape = input_shape
        self.adj_mats = adj_mats
        self.group_matrix = None

        self.hgblock_1 = HGResidualBlock(self.input_shape[1], num_groups)
        self.hgblock_2 = HGResidualBlock(self.input_shape[1], num_groups)
        self.hgblock_3 = HGResidualBlock(self.input_shape[1], num_groups)
        self.output_conv = nn.Conv2d(2*self.input_shape[1], self.input_shape[1], 1, padding="same")
        self.eps = 0.0000001 # 0除算を回避するための定数

    def grouping(self, x):
        # if self.training:
        _, group_label, _ = ssn_iter(x, self.num_groups, self.num_iter)
        # else:
            # _, group_label, _ = sparse_ssn_iter(x, self.num_groups, self.num_iter)
        group_matrix = torch.eye(self.num_groups, device=group_label.device, requires_grad=False)[group_label]
        return group_matrix

    def pooling(self, x, group_matrix):
        group_matrix_bar = group_matrix / (group_matrix.sum(dim=2, keepdim=True) + self.eps)
        x = x.reshape(self.input_shape[0], self.input_shape[1], -1)
        x = torch.bmm(x, group_matrix_bar).transpose(1,2)
        return x

    def unpooling(self, x, group_matrix):
        group_matrix_tilde = group_matrix / (group_matrix.sum(dim=1, keepdim=True) + self.eps)
        x = torch.bmm(group_matrix_tilde, x)
        return x

    def make_coef_matrix(self, group_matrix):
        # A_hatを計算
        adj_mats = self.adj_mats[None] # バッチ軸を追加
        group_matrix = group_matrix[:,None] # 方向軸を追加
        group_adj_matrix = (adj_mats @ group_matrix).transpose(2,3) @ group_matrix
        # Noise Reduction
        denoise_group_adj_matrix = nn.functional.relu(group_adj_matrix - torch.flip(group_adj_matrix, dims=[0]))
        denoise_group_adj_matrix[:, 3] = group_adj_matrix[:, 3]

        scale_matrix = 1/(group_adj_matrix.sum(dim=3, keepdims=True) + 1)
        coef_matrix = denoise_group_adj_matrix * scale_matrix
        return coef_matrix

    def forward(self, inp):
        self.group_matrix = self.grouping(inp)
        coef_matrix = self.make_coef_matrix(self.group_matrix)
        x = self.pooling(inp, self.group_matrix)
        x = self.hgblock_1(x, coef_matrix)
        x = self.hgblock_2(x, coef_matrix)
        x = self.hgblock_3(x, coef_matrix)
        x = self.unpooling(x, self.group_matrix)
        x = x.reshape(self.input_shape)
        x = torch.concat([x, inp], dim=1)
        x = self.output_conv(x)
        return x

class DillationConvBlock(nn.Module):
    def __init__(self, input_shape, dillation_ratio=1):
        super().__init__()
        self.conv_1 = nn.Conv2d(input_shape[1], input_shape[1], kernel_size=3, dilation=dillation_ratio, padding="same", bias=False)
        self.conv_2 = nn.Conv2d(input_shape[1], input_shape[1], kernel_size=3, dilation=dillation_ratio, padding="same", bias=False)
        self.conv_3 = nn.Conv2d(input_shape[1], input_shape[1], kernel_size=3, dilation=dillation_ratio, padding="same", bias=False)

        self.bn1 = nn.BatchNorm2d(input_shape[1])
        self.bn2 = nn.BatchNorm2d(input_shape[1])
        self.bn3 = nn.BatchNorm2d(input_shape[1])
        self.act = nn.ReLU()

    def forward(self,x):
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv_2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.conv_3(x)
        x = self.bn3(x)
        x = self.act(x)

        return x

class HGResNet(nn.Module):
    def __init__(self, input_hw, class_num, batch_size):
        super().__init__()
        self.input_hw = input_hw
        self.backbone = timm.create_model("resnet18", pretrained=False, features_only=True, out_indices=[2])
        embed_shape = self.backbone(torch.randn(batch_size, 3, input_hw[0], input_hw[1]))[0].shape
        self.dilation_conv = DillationConvBlock(embed_shape, dillation_ratio=3)
        adj_mats = make_adj_matrix(embed_shape[2], embed_shape[3]).cuda()
        self.hgconv_block = HGBlock(embed_shape, int(embed_shape[2]*embed_shape[3]/64), 3, adj_mats)
        self.segmentation_head_aux = nn.Sequential(nn.Conv2d(embed_shape[1], embed_shape[1], 3, padding="same"),
                                                   nn.BatchNorm2d(embed_shape[1]),
                                                   nn.ReLU(),
                                                   nn.Dropout2d(),
                                                   nn.Conv2d(embed_shape[1], class_num, 1, padding="same"),
                                                   )
        self.segmentation_head = nn.Sequential(nn.Conv2d(embed_shape[1], embed_shape[1], 3, padding="same"),
                                               nn.BatchNorm2d(embed_shape[1]),
                                               nn.ReLU(),
                                               nn.Dropout2d(),
                                               nn.Conv2d(embed_shape[1], class_num, 1, padding="same"),
                                               )
    def forward(self, x):
        x = self.backbone(x)[0]
        x = self.dilation_conv(x)
        out1 = self.segmentation_head_aux(x)
        out1 = F.interpolate(out1, self.input_hw, mode="bilinear", align_corners=False)
        x = self.hgconv_block(x)
        out2 = self.segmentation_head(x)
        out2 = F.interpolate(out2, self.input_hw, mode="bilinear", align_corners=False)
        return out1, out2