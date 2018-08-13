import torch
import torch.nn as nn
import math


class DCEC(nn.Module):
    def __init__(self, input_shape=[128,128,3], num_clusters=10, filters=[32, 64, 128]):
        super(DCEC, self).__init__()
        bias = True
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        self.conv1 = nn.Conv2d(input_shape[2], filters[0], 5, stride=2, padding=2, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(negative_slope=0.1)
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

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        x = self.relu(x)
        # print(x.size())
        x = self.conv2(x)
        x = self.relu(x)
        # print(x.size())
        x = self.conv3(x)
        x = self.relu(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.embedding(x)
        # print(x.size())
        extra_out = x
        clustering_out = self.clustering(x)
        # print(clustering_out.size())
        x = self.deembedding(x)
        x = self.relu(x)
        x = x.view(x.size(0), self.filters[2], ((self.input_shape[0]//2//2-1) // 2), ((self.input_shape[0]//2//2-1) // 2))
        # print(x.size())
        x = self.deconv3(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv2(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv1(x)
        # print(x.size())
        return x, clustering_out, extra_out


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
        # q = 1.0 / (1.0 + (K.sum(K.square(x), axis=2) / self.alpha))
        # q **= (self.alpha + 1.0) / 2.0
        # q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        # return q
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, alpha={}'.format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)
