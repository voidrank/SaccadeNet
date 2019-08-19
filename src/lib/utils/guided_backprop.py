"""
Created on Thu Oct 26 11:23:47 2017
@author: Utku Ozbulak - github.com/utkuozbulak

Modified on Wed Jul 24 17:37:43 2019
@author: Shiyi Lan - github.com/voidrank
"""

from IPython import embed

import torch
from torch.nn import ReLU, Module

from .misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = {}
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()
        self.gradients = []

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients.append(grad_in[0])
        # Register hook to the first layer
        for layer in self.model.hook_layers:
            layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[id(module)]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs[id(module)] = ten_out

        def search_and_update_relus(nn_model):
            if isinstance(nn_model, ReLU):
                nn_model.register_backward_hook(relu_backward_hook_function)
                nn_model.register_forward_hook(relu_forward_hook_function)
            elif isinstance(nn_model, Module):
                for key, sub_module in nn_model._modules.items():
                    search_and_update_relus(sub_module)


        # Loop through layers, hook up ReLUs
        search_and_update_relus(self.model)

    def generate_wh_gradients(self, wh, topk=1):
        wh_one_hot = torch.FloatTensor(*wh.shape).zero_()
        # Target for backprop
        wh_one_hot[:, :topk, :] = 1
        wh_one_hot = wh_one_hot.cuda()
        wh.backward(gradient=wh_one_hot)
        gradients_as_arr = [gradient.cpu().data.numpy()[0] for gradient in self.gradients]
        return gradients_as_arr

    def generate_class_gradients(self, target_class, model_output):
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr


if __name__ == '__main__':
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    # Save colored gradients
    save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
    save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
    print('Guided backprop completed')