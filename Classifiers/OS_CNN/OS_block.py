import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def calculate_mask_index(kernel_length_now,largest_kernel_lenght):
    right_zero_mast_length = math.ceil((largest_kernel_lenght-1)/2)-math.ceil((kernel_length_now-1)/2)
    left_zero_mask_length = largest_kernel_lenght - kernel_length_now - right_zero_mast_length
    return left_zero_mask_length, left_zero_mask_length+ kernel_length_now

def creat_mask(number_of_input_channel,number_of_output_channel, kernel_length_now, largest_kernel_lenght):
    ind_left, ind_right= calculate_mask_index(kernel_length_now,largest_kernel_lenght)
    mask = np.ones((number_of_input_channel,number_of_output_channel,largest_kernel_lenght))
    mask[:,:,0:ind_left]=0
    mask[:,:,ind_right:]=0
    return mask


def creak_layer_mask(layer_parameter_list):
    largest_kernel_lenght = layer_parameter_list[-1][-1]
    mask_list = []
    init_weight_list = []
    bias_list = []
    for i in layer_parameter_list:
        conv = torch.nn.Conv1d(in_channels=i[0], out_channels=i[1], kernel_size=i[2])
        ind_l,ind_r= calculate_mask_index(i[2],largest_kernel_lenght)
        big_weight = np.zeros((i[1],i[0],largest_kernel_lenght))
        big_weight[:,:,ind_l:ind_r]= conv.weight.detach().numpy()
        
        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)
        
        mask = creat_mask(i[1],i[0],i[2], largest_kernel_lenght)
        mask_list.append(mask)
        
    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.concatenate(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)

    
class build_layer_with_layer_parameter(nn.Module):
    def __init__(self,layer_parameters, relu_or_not_at_last_layer = True):
        super(build_layer_with_layer_parameter, self).__init__()
        self.relu_or_not_at_last_layer = relu_or_not_at_last_layer

        os_mask, init_weight, init_bias= creak_layer_mask(layer_parameters)
        
        
        in_channels = os_mask.shape[1] 
        out_channels = os_mask.shape[0] 
        max_kernel_size = os_mask.shape[-1]

        self.weight_mask = nn.Parameter(torch.from_numpy(os_mask),requires_grad=False)
        
        self.padding = nn.ConstantPad1d((int((max_kernel_size-1)/2), int(max_kernel_size/2)), 0)
         
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=max_kernel_size)
        self.conv1d.weight = nn.Parameter(torch.from_numpy(init_weight),requires_grad=True)
        self.conv1d.bias =  nn.Parameter(torch.from_numpy(init_bias),requires_grad=True)

        self.bn = nn.BatchNorm1d(num_features=out_channels)
    
    def forward(self, X):
        self.conv1d.weight.data = self.conv1d.weight*self.weight_mask
        #self.conv1d.weight.data.mul_(self.weight_mask)
        result_1 = self.padding(X)
        result_2 = self.conv1d(result_1)
        result_3 = self.bn(result_2)
        if self.relu_or_not_at_last_layer:
            result = F.relu(result_3)
            return result
        else:
            return result_3
    
class OS_block(nn.Module):
    def __init__(self,layer_parameter_list, relu_or_not_at_last_layer = True):
        super(OS_block, self).__init__()
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []
        self.relu_or_not_at_last_layer = relu_or_not_at_last_layer
        
        
        for i in range(len(layer_parameter_list)):
            if i!= len(layer_parameter_list)-1:
                using_relu = True
            else:
                using_relu = self.relu_or_not_at_last_layer
                
            layer = build_layer_with_layer_parameter(layer_parameter_list[i],using_relu)
            self.layer_list.append(layer)
        
        self.net = nn.Sequential(*self.layer_list)
            

    def forward(self, X):
        
        X = self.net(X)

        return X
    
# class SampaddingConv1D_BN(nn.Module):
#     def __init__(self,in_channels,out_channels,kernel_size):
#         super(SampaddingConv1D_BN, self).__init__()
#         self.padding = nn.ConstantPad1d((int((kernel_size-1)/2), int(kernel_size/2)), 0)
#         self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
#         self.bn = nn.BatchNorm1d(num_features=out_channels)
        
#     def forward(self, X):
#         X = self.padding(X)
#         X = self.conv1d(X)
#         X = self.bn(X)
#         return X
    
# class build_layer_with_layer_parameter(nn.Module):
#     def __init__(self,layer_parameters): 
#         super(build_layer_with_layer_parameter, self).__init__()
#         self.conv_list = nn.ModuleList()
        
#         for i in layer_parameters:
#             conv = SampaddingConv1D_BN(i[0],i[1],i[2])
#             self.conv_list.append(conv)
    
#     def forward(self, X):
        
#         conv_result_list = []
#         for conv in self.conv_list:
#             conv_result = conv(X)
#             conv_result_list.append(conv_result)
            
#         result = F.relu(torch.cat(tuple(conv_result_list), 1))
#         return result
