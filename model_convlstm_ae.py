import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        * input_dim: int
            Number of channels of input tensor.
        * hidden_dim: int
            Number of channels of hidden state.
        * kernel_size: (int, int)
            Size of the convolutional kernel.
        * bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Parameters:
        * input_dim: Number of channels in input
        * hidden_dim: Number of hidden channels
        * kernel_size: Size of kernel in convolutions
        * num_layers: Number of LSTM layers stacked on each other
        * batch_first: Whether or not dimension 0 is the batch or not
        * bias: Bias or no bias in Convolution
        * return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
        
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        * input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        * hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ST_AutoEncoder(nn.Module):
    """
    Sequential Model for the Spatio Temporal Autoencoder (ST_AutoEncoder)
    """

    def __init__(self, in_channel, n_outliers):
        super(ST_AutoEncoder, self).__init__()

        self.in_channel = in_channel
        self.n_outliers = n_outliers

        # Spatial Encoder
        self.spatial_encoder = nn.Sequential(
            nn.BatchNorm3d(self.in_channel),
            nn.Conv3d(in_channels=self.in_channel, out_channels=128, kernel_size=(1, 13, 13), stride=(1, 2, 2)),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(5, 5, 5), stride=(3, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(5, 11, 11), stride=(2, 2, 2)),
            nn.ReLU()
        )

        # Temporal Encoder & Decoder
        self.temporal_encoder_decoder = Temporal_EncDec(self.n_outliers)

        # Spatial Decoder
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(5, 11, 11), stride=(2, 2, 2)),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=128, kernel_size=(5, 5, 5), stride=(4, 2, 2)),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=self.in_channel, kernel_size=(1, 13, 13),  stride=(1, 2, 2)),  # out_channels=3 - original
            nn.ReLU()
        )

    def forward(self, x):
        # x : (N, C, T, H, W) / T : sequence length(or D)
        h_hat = self.spatial_encoder(x)
        #print(h_hat.shape)
        h_hat = h_hat.permute(0, 2, 1, 3, 4)  # (N, C, T, H, W) -> (N, T, C, H, W)
        h_hat, loss_contrastive  = self.temporal_encoder_decoder(h_hat)
        h_hat = h_hat.permute(0, 2, 1, 3, 4)  # (N, T, C, H, W) -> (N, C, T, H, W)
        #print(h_hat.shape)
        output = self.spatial_decoder(h_hat)
        return output, loss_contrastive


class Temporal_EncDec(nn.Module):
    def __init__(self, n_outliers):
        super(Temporal_EncDec, self).__init__()

        self.n_outliers = n_outliers
        
        self.convlstm_1 = ConvLSTM(input_dim=64, hidden_dim=64, kernel_size=(3, 3), num_layers=1, bias=True,
                                   return_all_layers=True)
        self.convlstm_2 = ConvLSTM(input_dim=64, hidden_dim=32, kernel_size=(3, 3), num_layers=1, bias=True,
                                   return_all_layers=True)
        self.convlstm_3 = ConvLSTM(input_dim=32, hidden_dim=64, kernel_size=(3, 3), num_layers=1, bias=True)

    def forward(self, inp):
        layer_output_list, _ = self.convlstm_1(inp)
        layer_output_list, _ = self.convlstm_2(layer_output_list[0])

        loss_contrastive = torch.tensor(0)
        if self.n_outliers>0:
            x = layer_output_list[0][:-self.n_outliers]
            #print('main:',x.shape)
            y = layer_output_list[0][-self.n_outliers:]
            #print('outliers:',y.shape)
            x = torch.flatten(x, start_dim=1)
            #print('main flattened:',x.shape)        
            y = torch.flatten(y, start_dim=1)
            #print('outliers flattened:',y.shape)
            sum_dist_xy = torch.cdist(x,y,p=2).sum() / (x.shape[0]*y.shape[0])
            #print('sum distances xy :',sum_dist_xy)
            sum_dist_xx = torch.cdist(x,x,p=2).sum() / (x.shape[0]*(x.shape[0]-1)/2)
            #print('sum distances xx:',sum_dist_xx)
            #sum_dist_yy = torch.cdist(y,y,p=2).sum()
            #print('sum distances yy:',sum_dist_yy)            
            loss_contrastive = sum_dist_xx - sum_dist_xy
                
        layer_output_list, _ = self.convlstm_3(layer_output_list[0])
        
        return layer_output_list[0], loss_contrastive


if __name__ == "__main__":
    device = torch.device("cuda")

    model = ST_AutoEncoder(1,2)
    model.to(device)
    x = torch.randn(16, 1, 21, 101, 101).to(device)
    print(x.shape)
    xp, loss_contrastive = model(x)
    print(xp.shape)
    print('contrastive_loss:',loss_contrastive)
    
