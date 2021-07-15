import torch.nn.functional as F
from torch import nn

import torch


class Perciever(nn.Module):
    def __init__(self, n_query=10, query_size=15, latent_size=44, num_repeats=2):
        super().__init__()
        self.num_repeats = num_repeats
        self.kernel_h = 4
        self.kernel_w = 4
        self.step = 4
        self.n_channels = 1
        self.img_h = 28
        self.img_w = 28
        self.unfold = nn.Unfold(kernel_size=(4, 4), stride=4)
        x_coord = torch.stack([torch.arange(0, self.img_w) for x in range(self.img_h)])
        y_coord = torch.stack([torch.arange(0, self.img_h) for x in range(self.img_w)])
        coord_array = torch.stack([x_coord, y_coord.T], )
        coord_unf = self.unfold(coord_array.unsqueeze(1).float())

        self.latent_array = nn.Parameter(torch.normal(torch.zeros(n_query, latent_size)),
                                         requires_grad=True)
        # normalize to range [-1, 1]
        coord_ave = (coord_unf.permute(0, 2,1).mean(-1).T / torch.as_tensor((self.img_h, self.img_w)) - 0.5) * 2
        self.pos_embed_matrix = nn.Parameter(torch.normal(torch.zeros(2, 10)), requires_grad=False)
        proj = coord_ave.to(self.pos_embed_matrix) @ self.pos_embed_matrix

        self.coord_embed = nn.Parameter(torch.cat([torch.sin(proj), torch.cos(proj)], -1))


        n_neurons = 256
        self.query = nn.Sequential(nn.Linear(latent_size, n_neurons),
                                    nn.LeakyReLU(),
                                    nn.Linear(n_neurons, query_size))
        self.bias = nn.Parameter(torch.normal(torch.zeros(latent_size)),
                                 requires_grad=True)
        self.item_size = self.kernel_h * self.kernel_w * self.n_channels
        position_size = 20
        self.key = nn.Sequential(nn.Linear(self.item_size + position_size, n_neurons),
                                    nn.LeakyReLU(),
                                    nn.Linear(n_neurons, query_size))

        self.value = nn.Sequential(nn.Linear(self.item_size + position_size, n_neurons),
                                    nn.LeakyReLU(),
                                    nn.Linear(n_neurons, latent_size))

        self.query1 = nn.Sequential(nn.Linear(latent_size, n_neurons),
                                    nn.LeakyReLU(),
                                    nn.Linear(n_neurons, query_size))

        self.key1 = nn.Sequential(nn.Linear(latent_size, n_neurons),
                                    nn.LeakyReLU(),
                                    nn.Linear(n_neurons, query_size))

        self.value1 = nn.Sequential(nn.Linear(latent_size, n_neurons),
                                    nn.LeakyReLU(),
                                    nn.Linear(n_neurons, latent_size))

        self.linear = nn.Linear(latent_size, 512)
        self.linear1 = nn.Linear(n_query * 512, 10)

    def forward(self, img):
        """
        iteratively:
            build query
            apply cross-attention between query and input data
            apply cross + self-attention to the result
        """
        data = self.unfold(img).permute(0, 2, 1)
        data = torch.cat([data, self.coord_embed.repeat(data.shape[0], 1, 1)], -1)

        # cross-attention
        Q = self.query(self.latent_array)
        K1 = self.key(data)
        V1 = self.value(data)

        code = torch.softmax(Q@K1.permute(0, 2, 1), dim=-1) @ V1

        for i in range(self.num_repeats):
            # Latent transformer
            Q = self.query1(code)
            K = self.key1(code)
            V = self.value1(code)
            code_back = code
            code = torch.softmax(Q@K.permute(0, 2, 1), dim=-1) @ V

            # cross-attention
            Q = self.query(code)
            K = self.key(data)
            V = self.value(data)
            # residual
            code = torch.softmax(Q@K.permute(0, 2, 1), dim=-1) @ V + code_back - self.bias

        result = self.final(code)
        return result

    def final(self, code):
        return F.log_softmax(self.linear1(F.leaky_relu(self.linear(code).flatten(1))), dim=1)

