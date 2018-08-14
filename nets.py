import torch
import torch.nn as nn
import math
import copy

class ClusterlingLayer(nn.Module):
    def __init__(self, in_features=10, out_features=10, alpha=1.0):
        super(ClusterlingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha +1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, alpha={}'.format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)


class CAE_3(nn.Module):
    def __init__(self, input_shape=[128,128,3], num_clusters=10, filters=[32, 64, 128]):
        super(CAE_3, self).__init__()
        bias = True
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        self.conv1 = nn.Conv2d(input_shape[2], filters[0], 5, stride=2, padding=2, bias=bias)
        self.relu = nn.ReLU(inplace=False)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.conv3 = nn.Conv2d(filters[1], filters[2], 3, stride=2, padding=0, bias=bias)
        lin_features_len = ((input_shape[0]//2//2-1) // 2) * ((input_shape[0]//2//2-1) // 2) * filters[2]
        # print(lin_features_len)
        self.embedding = nn.Linear(lin_features_len, num_clusters, bias=bias)
        self.deembedding = nn.Linear(num_clusters, lin_features_len, bias=bias)
        out_pad = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=0, output_padding=out_pad, bias=bias)
        out_pad = 1 if input_shape[0] // 2 % 2 == 0 else 0
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        out_pad = 1 if input_shape[0] % 2 == 0 else 0
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[2], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        self.clustering = ClusterlingLayer(num_clusters, num_clusters)
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        x = self.relu1_1(x)
        # print(x.size())
        x = self.conv2(x)
        x = self.relu2_1(x)
        # print(x.size())
        x = self.conv3(x)
        x = self.relu3_1(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.embedding(x)
        # print(x.size())
        extra_out = x
        clustering_out = self.clustering(x)
        # print(clustering_out.size())
        x = self.deembedding(x)
        x = self.relu1_2(x)
        x = x.view(x.size(0), self.filters[2], ((self.input_shape[0]//2//2-1) // 2), ((self.input_shape[0]//2//2-1) // 2))
        # print(x.size())
        x = self.deconv3(x)
        x = self.relu2_2(x)
        # print(x.size())
        x = self.deconv2(x)
        x = self.relu3_2(x)
        # print(x.size())
        x = self.deconv1(x)
        # print(x.size())
        return x, clustering_out, extra_out


class CAE_3bn(nn.Module):
    def __init__(self, input_shape=[128,128,3], num_clusters=10, filters=[32, 64, 128]):
        super(CAE_3bn, self).__init__()
        bias = True
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        self.conv1 = nn.Conv2d(input_shape[2], filters[0], 5, stride=2, padding=2, bias=bias)
        self.bn1_1 = nn.BatchNorm2d(filters[0])
        self.relu = nn.ReLU(inplace=False)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.bn2_1 = nn.BatchNorm2d(filters[1])
        self.conv3 = nn.Conv2d(filters[1], filters[2], 3, stride=2, padding=0, bias=bias)
        # self.bn3_1 = nn.BatchNorm2d(filters[2])
        lin_features_len = ((input_shape[0]//2//2-1) // 2) * ((input_shape[0]//2//2-1) // 2) * filters[2]
        # print(lin_features_len)
        self.embedding = nn.Linear(lin_features_len, num_clusters, bias=bias)
        self.deembedding = nn.Linear(num_clusters, lin_features_len, bias=bias)
        out_pad = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=0, output_padding=out_pad, bias=bias)
        out_pad = 1 if input_shape[0] // 2 % 2 == 0 else 0
        self.bn3_2 = nn.BatchNorm2d(filters[1])
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        out_pad = 1 if input_shape[0] % 2 == 0 else 0
        self.bn2_2 = nn.BatchNorm2d(filters[0])
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[2], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        self.clustering = ClusterlingLayer(num_clusters, num_clusters)
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.bn1_1(x)
        # print(x.size())
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.bn2_1(x)
        # print(x.size())
        x = self.conv3(x)
        x = self.relu3_1(x)
        # x = self.bn3_1(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.embedding(x)
        # print(x.size())
        extra_out = x
        clustering_out = self.clustering(x)
        # print(clustering_out.size())
        x = self.deembedding(x)
        x = self.relu1_2(x)
        x = x.view(x.size(0), self.filters[2], ((self.input_shape[0]//2//2-1) // 2), ((self.input_shape[0]//2//2-1) // 2))
        # print(x.size())
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.bn3_2(x)
        # print(x.size())
        x = self.deconv2(x)
        x = self.relu3_2(x)
        x = self.bn2_2(x)
        # print(x.size())
        x = self.deconv1(x)
        # print(x.size())
        return x, clustering_out, extra_out


class CAE_4(nn.Module):
    def __init__(self, input_shape=[128,128,3], num_clusters=10, filters=[32, 64, 128, 256]):
        super(CAE_4, self).__init__()
        bias = True
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        self.relu = nn.ReLU(inplace=False)
        self.relu = nn.LeakyReLU(negative_slope=0.01)

        self.conv1 = nn.Conv2d(input_shape[2], filters[0], 5, stride=2, padding=2, bias=bias)
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.conv3 = nn.Conv2d(filters[1], filters[2], 5, stride=2, padding=2, bias=bias)
        self.conv4 = nn.Conv2d(filters[2], filters[3], 3, stride=2, padding=0, bias=bias)

        lin_features_len = ((input_shape[0] // 2 // 2 // 2 - 1) // 2) * ((input_shape[0] // 2 // 2 // 2 - 1) // 2) * \
                           filters[3]
        # print(lin_features_len)
        self.embedding = nn.Linear(lin_features_len, num_clusters, bias=bias)
        self.deembedding = nn.Linear(num_clusters, lin_features_len, bias=bias)
        out_pad = 1 if input_shape[0] // 2 // 2 // 2 % 2 == 0 else 0
        self.deconv4 = nn.ConvTranspose2d(filters[3], filters[2], 3, stride=2, padding=0, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[0] // 2 // 2 % 2 == 0 else 0
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[0] // 2 % 2 == 0 else 0
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = 1 if input_shape[0] % 2 == 0 else 0
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[2], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.clustering = ClusterlingLayer(num_clusters, num_clusters)
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)

    def forward(self, x):
        print(x.size())
        x = self.conv1(x)
        x = self.relu1_1(x)
        print(x.size())
        x = self.conv2(x)
        x = self.relu2_1(x)
        print(x.size())
        x = self.conv3(x)
        x = self.relu3_1(x)
        print(x.size())
        x = self.conv4(x)
        x = self.relu4_1(x)
        print(x.size())
        x = x.view(x.size(0), -1)
        print(x.size())
        x = self.embedding(x)
        print(x.size())
        extra_out = x
        clustering_out = self.clustering(x)
        print(clustering_out.size())
        x = self.deembedding(x)
        x = self.relu4_2(x)
        x = x.view(x.size(0), self.filters[2], ((self.input_shape[0]//2//2-1) // 2), ((self.input_shape[0]//2//2-1) // 2))
        print(x.size())
        x = self.deconv4(x)
        x = self.relu3_2(x)
        print(x.size())
        x = self.deconv3(x)
        x = self.relu2_2(x)
        print(x.size())
        x = self.deconv2(x)
        x = self.relu1_2(x)
        print(x.size())
        x = self.deconv1(x)
        print(x.size())
        return x, clustering_out, extra_out


