import torch
from pytracking import optimization, TensorList, operation
import math


class FactorizedConvProblem(optimization.L2Problem):
    def __init__(self, training_samples: TensorList, y:TensorList, filter_reg: torch.Tensor, projection_reg, params, sample_weights: TensorList,
                 projection_activation, response_activation):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.params = params
        self.projection_reg = projection_reg
        self.projection_activation = projection_activation
        self.response_activation = response_activation
        
        #self.attention1_reg = projection_reg
        #self.attention2_reg = projection_reg
        #self.attention3_reg = projection_reg
        #self.diag_M = (((self.filter_reg.concat(self.attention3_reg)).concat(self.attention2_reg)).concat(self.attention1_reg)).concat(projection_reg)
        
        self.diag_M = self.filter_reg.concat(projection_reg)

    def __call__(self, x: TensorList):
        """
        Compute residuals
        :param x: [filters, projection_matrices]
        :return: [data_terms, filter_regularizations, proj_mat_regularizations]
        """
        #import pdb
        #pdb.set_trace()
        filter = x[:len(x)//2]  # w2 in paper
        P = x[len(x)//2:]       # w1 in paper
        
        #filter = x[:1]
        #attention3 = x[1:2]
        #attention2 = x[2:3]
        #attention1 = x[3:4]
        #P = x[4:]
        #import pdb
        #pdb.set_trace()
        # Do first convolution
        compressed_samples = operation.conv1x1(self.training_samples, P).apply(self.projection_activation)
        #compressed_samples = operation.conv2d(self.training_samples, P, mode='same').apply(self.projection_activation)
        #compressed_samples = operation.channel_attention(compressed_samples, attention1, attention2, attention3)
        #import pdb
        #pdb.set_trace()
        # Do second convolution
        residuals = operation.conv2d(compressed_samples, filter, mode=None).apply(self.response_activation)
        #residuals = operation.conv2d(residuals, filter2, mode=None).apply(self.response_activation)
        #residuals = operation.conv2d(residuals, filter3, mode=None).apply(self.response_activation)
        # Compute data residuals
        residuals = residuals - self.y
        
        #a=10
        #c=0.2
        #shrinkage_parameter=torch.exp(self.y[0])/(1+torch.exp(a*(c-torch.abs(residuals[0]))))
        #shrinkage_parameter=shrinkage_parameter.sqrt()
        #residuals=TensorList([shrinkage_parameter])*residuals
        
        residuals = self.sample_weights.sqrt().view(-1, 1, 1, 1) * residuals

        # Add regularization for projection matrix
        residuals.extend(self.filter_reg.apply(math.sqrt) * filter)
        
        #residuals.extend(self.attention3_reg.apply(math.sqrt) * attention3)
        
        #residuals.extend(self.attention2_reg.apply(math.sqrt) * attention2)
        
        #residuals.extend(self.attention1_reg.apply(math.sqrt) * attention1)
        
        # Add regularization for projection matrix
        residuals.extend(self.projection_reg.apply(math.sqrt) * P)

        return residuals


    def ip_input(self, a: TensorList, b: TensorList):
        #import pdb
        #pdb.set_trace()
        num = len(a) // 2       # Number of filters
        a_filter = a[:num]
        b_filter = b[:num]
        a_P = a[num:]
        b_P = b[num:]
        

        #a_filter = a[:1]
        #b_filter = b[:1]
        #a_P = a[4:]
        #b_P = b[4:]
        #a_attention3 = a[1:2]
        #b_attention3 = b[1:2]
        #a_attention2 = a[2:3]
        #b_attention2 = b[2:3]
        #a_attention1 = a[3:4]
        #b_attention1 = b[3:4]
        

        # Filter inner product
        # ip_out = a_filter.reshape(-1) @ b_filter.reshape(-1)
        ip_out = operation.conv2d(a_filter, b_filter).view(-1)
        
        #ip_out += operation.conv2d(a_filter2.view(1, -1, 1, 1), b_filter2.view(1, -1, 1, 1)).view(-1)
        
        #ip_out += operation.conv2d(a_filter1.view(1, -1, 1, 1), b_filter1.view(1, -1, 1, 1)).view(-1)
        
        #ip_out += operation.conv2d(a_attention3, b_attention3).view(-1)
        
        #ip_out += a_attention2.reshape(-1) @ b_attention2.reshape(-1)
        
        #ip_out += a_attention1.reshape(-1) @ b_attention1.reshape(-1)

        # Add projection matrix part
        # ip_out += a_P.reshape(-1) @ b_P.reshape(-1)
        ip_out += operation.conv2d(a_P.view(1,-1,1,1), b_P.view(1,-1,1,1)).view(-1)

        # Have independent inner products for each filter
        return ip_out.concat(ip_out.clone())
        #return (((ip_out.concat(ip_out.clone())).concat(ip_out.clone())).concat(ip_out.clone())).concat(ip_out.clone())

    def M1(self, x: TensorList):
        return x / self.diag_M


class ConvProblem(optimization.L2Problem):
    def __init__(self, training_samples: TensorList, y:TensorList, filter_reg: torch.Tensor, sample_weights: TensorList, response_activation):
        self.training_samples = training_samples
        self.y = y
        self.filter_reg = filter_reg
        self.sample_weights = sample_weights
        self.response_activation = response_activation

    def __call__(self, x: TensorList):
        """
        Compute residuals
        :param x: [filters]
        :return: [data_terms, filter_regularizations]
        """
        # Do convolution and compute residuals
        #import pdb
        #pdb.set_trace()
        residuals = operation.conv2d(self.training_samples, x, mode=None).apply(self.response_activation)
        residuals = residuals - self.y
        
        #a=10
        #c=0.2
        #shrinkage_parameter=torch.exp(self.y[0])/(1+torch.exp(a*(c-torch.abs(residuals[0]))))
        #shrinkage_parameter=shrinkage_parameter.sqrt()
        #residuals=TensorList([shrinkage_parameter])*residuals
        
        residuals = self.sample_weights.sqrt().view(-1, 1, 1, 1) * residuals

        # Add regularization for projection matrix
        residuals.extend(self.filter_reg.apply(math.sqrt) * x)

        return residuals

    def ip_input(self, a: TensorList, b: TensorList):
        # return a.reshape(-1) @ b.reshape(-1)
        # return (a * b).sum()
        return operation.conv2d(a, b).view(-1)
