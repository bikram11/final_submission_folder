import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.Wxi = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=self.padding)
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=self.padding, bias=False)
        self.Wxf = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=self.padding)
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=self.padding, bias=False)
        self.Wxc = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=self.padding)
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=self.padding, bias=False)
        self.Wxo = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=self.padding)
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
        else:
            assert shape[0] == self.Wci.size(2), 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size(3), 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])))


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super().__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = f'cell{i}'
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                name = f'cell{i}'
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)


class ConvLSTMNet(nn.Module):
    def __init__(self, gridheight, gridwidth, seqlen):
        super().__init__()
        self.conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.pool = nn.AdaptiveAvgPool2d((6, 10))
        self.convlstm = ConvLSTM(input_channels=128, hidden_channels=[16], kernel_size=3, step=seqlen,
                                 effective_step=[seqlen - 1])
        self.fc3 = nn.Linear(960, gridheight * gridwidth)

    def forward(self, x):
        x = torch.squeeze(x).float()
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.pool(self.conv1(c_in))
        output_convlstm, _ = self.convlstm(c_out)
        x = output_convlstm[0]
        x = x.view(batch_size, timesteps, -1)
        x = self.fc3(x[:, -1, :])
        return x