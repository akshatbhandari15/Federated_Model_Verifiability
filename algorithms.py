import functools
from collections import OrderedDict
import torch

#averages all of the given state dicts
class fedavg():

    def __init__(self, config):
        self.algorithm = "FedAvg"

    def aggregate(self,server_state_dict,state_dicts):
        #server_state_dict is of no use in FedAvg,
        # to maintain consistency with other algorithms; it is provided as an argument
        result_state_dict = OrderedDict()
        for key in state_dicts[0].keys():
            current_key_tensors = [state_dict[key] for state_dict in state_dicts]
            current_key_sum = functools.reduce( lambda accumulator, tensor: accumulator + tensor, current_key_tensors )
            current_key_average = current_key_sum / len(state_dicts)
            result_state_dict[key] = current_key_average

        return result_state_dict


class fedadagrad():

    def __init__(self, config):
        self.algorithm = "FedAdagrad"
        self.lr = 0.01
        self.epsilon = 1e-6
        self.state = None

    def aggregate(self,server_state_dict,state_dicts):

        keys = server_state_dict.keys() #List of keys in a state_dict

        #Averages the differences that we got by subtracting the server_model from client_model (delta_y)
        avg_delta_y = OrderedDict()
        for key in keys:
            current_key_tensors = [state_dict[key] for state_dict in state_dicts]
            current_key_sum = functools.reduce( lambda accumulator, tensor: accumulator + tensor, current_key_tensors )
            current_key_average = current_key_sum / len(state_dicts)
            avg_delta_y[key] = current_key_average

        if not self.state: #If state = None, then the following line will execute.
            #So only at first round, it'll execute
            self.state = [torch.zeros_like(server_state_dict[key]) for key in server_state_dict.keys()]

        #Updates the server_state_dict
        for key, state in zip(keys, self.state):
            state.data += torch.square(avg_delta_y[key])
            server_state_dict[key] += self.lr * avg_delta_y[key] / torch.sqrt(state.data + self.epsilon)

        return server_state_dict

class fedadam():

    def __init__(self, config):
        self.algorithm = "FedAdam"
        self.lr = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6
        self.timestep = 1

        self.m = None #1st moment vectpr
        self.v = None #2nd moment vector

    def aggregate(self,server_state_dict,state_dicts):

        keys = server_state_dict.keys() #List of keys in a state_dict

        #Averages the differences that we got by subtracting the server_model from client_model (delta_y)
        avg_delta_y = OrderedDict()
        for key in keys:
            current_key_tensors = [state_dict[key] for state_dict in state_dicts]
            current_key_sum = functools.reduce( lambda accumulator, tensor: accumulator + tensor, current_key_tensors )
            current_key_average = current_key_sum / len(state_dicts)
            avg_delta_y[key] = current_key_average

        if not self.m: #If self.m = None, then the following line will execute. So only at first round, it'll execute
            self.m = [torch.zeros_like(server_state_dict[key]) for key in server_state_dict.keys()]
            self.v = [torch.zeros_like(server_state_dict[key]) for key in server_state_dict.keys()]

        #Updates the server_state_dict
        for key, m, v in zip(keys, self.m, self.v):
            m.data = self.beta1 * m.data + (1 - self.beta1) * avg_delta_y[key].data
            v.data = self.beta2 * v.data + (1 - self.beta2) * torch.square(avg_delta_y[key].data)
            m_bias_corr = m / (1 - self.beta1**self.timestep)
            v_bias_corr = v / (1 - self.beta2**self.timestep)
            server_state_dict[key].data += self.lr * m_bias_corr / (torch.sqrt(v_bias_corr) + self.epsilon)


        self.timestep += 1 #After each aggregation, timestep will increment by 1

        return server_state_dict

class fedavgm():

    def __init__(self, config):
        self.algorithm = "FedAvgM"
        self.momentum = 0.9
        self.lr = 1
        self.velocity = None

    def aggregate(self,server_state_dict,state_dicts):

        keys = server_state_dict.keys() #List of keys in a state_dict

        #Averages the differences that we got by subtracting the server_model from client_model (delta_y)
        avg_delta_y = OrderedDict()
        for key in keys:
            current_key_tensors = [state_dict[key] for state_dict in state_dicts]
            current_key_sum = functools.reduce( lambda accumulator, tensor: accumulator + tensor, current_key_tensors )
            current_key_average = current_key_sum / len(state_dicts)
            avg_delta_y[key] = current_key_average

        #Updates the velocity
        if self.velocity: #This will be False at the first round
            for key in keys:
                self.velocity[key] = self.momentum * self.velocity[key] + avg_delta_y[key]
        else:
            self.velocity = avg_delta_y

        #Uses Nesterov gradient
        for key in keys:
            avg_delta_y[key] += self.momentum * self.velocity[key]

        #Updates server_state_dict
        for key in keys:
            server_state_dict[key] += self.lr * avg_delta_y[key]

        return server_state_dict


class feddyn():

    def __init__(self, config):
        self.algorithm = "Mime"
        self.lr = 1.0
        self.momentum = 0.9
        self.h = None
        self.alpha = 0.01

    def aggregate(self, server_model_state_dict, state_dicts):

        keys = server_model_state_dict.keys() #List of keys in a state_dict

        if not self.h: #If self.h = None, then the following line will execute.
            #So only at first round, it'll execute
            self.h = [torch.zeros_like(server_model_state_dict[key]) for key in server_model_state_dict.keys()]

        sum_y = OrderedDict() #This will be our new server_model_state_dict
        for key in keys:
            current_key_tensors = [state_dict[key] for state_dict in state_dicts]
            current_key_sum = functools.reduce( lambda accumulator, tensor: accumulator + tensor, current_key_tensors )
            sum_y[key] = current_key_sum

        delta_x = [torch.zeros_like(server_model_state_dict[key]) for key in server_model_state_dict.keys()]
        for d_x, key in zip(delta_x, keys):
            d_x.data = sum_y[key] - server_model_state_dict[key]

        #Update h
        for h, d_x in zip(self.h, delta_x):
            h.data -= (self.alpha/len(state_dicts)) * d_x.data

        #Update x
        for key, h in zip(keys, self.h):
            server_model_state_dict[key] = (sum_y[key]/len(state_dicts)) - (h.data/self.alpha)

        return server_model_state_dict

class fedyogi():

    def __init__(self, config):
        self.algorithm = "FedYogi"
        self.lr = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6
        self.timestep = 1

        self.m = None #1st moment vectpr
        self.v = None #2nd moment vector

    def aggregate(self,server_state_dict,state_dicts):

        keys = server_state_dict.keys()  #List of keys in a state_dict

        #Averages the differences that we got by subtracting the server_model from client_model (delta_y)
        avg_delta_y = OrderedDict()
        for key in keys:
            current_key_tensors = [state_dict[key] for state_dict in state_dicts]
            current_key_sum = functools.reduce( lambda accumulator, tensor: accumulator + tensor, current_key_tensors )
            current_key_average = current_key_sum / len(state_dicts)
            avg_delta_y[key] = current_key_average

        if not self.m: #If self.m = None, then the following line will execute.
            #So only at first round, it'll execute
            self.m = [torch.zeros_like(server_state_dict[key]) for key in server_state_dict.keys()]
            self.v = [torch.zeros_like(server_state_dict[key]) for key in server_state_dict.keys()]

        #Updates the server_state_dict
        for key, m, v in zip(keys, self.m, self.v):
            m.data = self.beta1 * m.data + (1 - self.beta1) * avg_delta_y[key].data
            v.data = v.data + (1 - self.beta2) * torch.sign(
                                    torch.square(avg_delta_y[key].data) - v.data
                                ) * torch.square(avg_delta_y[key].data)

            m_bias_corr = m / (1 - self.beta1**self.timestep)
            v_bias_corr = v / (1 - self.beta2**self.timestep)
            server_state_dict[key].data += self.lr * m_bias_corr / (torch.sqrt(v_bias_corr) + self.epsilon)


        self.timestep += 1 #After each aggregation, timestep will increment by 1

        return server_state_dict
