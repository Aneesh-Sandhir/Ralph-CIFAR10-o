#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 13:27:18 2026

@author: asandhir
"""

import os
from pathlib import Path
import requests
import io
from collections.abc import Iterable
from PIL import Image
from tqdm.notebook import tqdm, trange

import torch
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F
from scipy.optimize import differential_evolution
from Tracking import Tracking

import numpy as np
import matplotlib.pyplot as plt

class RalphCIFAR10o:
    def __init__(self, model_name, maxiter = 50, population_size = 20):
        # load model and its utilities
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'    
        self.inputs_dir = Path('inputs')
        self.outputs_dir = Path('animations')
        self.model_name = model_name
        self.model = self.load_model(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.means, self.stdevs = self.get_normalization_parameters(self.model_name)
        self.input_size = self.get_input_size(self.model_name)
        self.classes = self.get_classes(self.model_name)
        
        # image processing pipelines
        self.downsample = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()          # Converts to tensor and scales pixels to [0.0, 1.0]
            ])
        self.model_normalize = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),          # Converts to tensor and scales pixels to [0.0, 1.0]
            transforms.Normalize(mean = self.means, 
                                 std = self.stdevs)
            ])
        
        # plot image as the model "sees" it
        self.target = Image.open(self.select_target())
        self.plot_figure(self.target)
        self.plot_figure(self.downsample(self.target))
        self.plot_figure(self.model_normalize(self.target))
        
        
        self.target_class = self.get_target_class(self.model_name)
        self.pixel_budget = self.get_pixel_budget(self.model_name)
        self.maxiter = maxiter
        self.population_size = population_size
        self.mutations = self.n_pixel_attack(self.target, self.target_class, 
                                              self.pixel_budget, maxiter = self.maxiter, 
                                              population_size = self.population_size)[-1]
        self.perturbed_image = self.perturb_image(self.mutations, self.target)
        self.plot_figure(self.perturbed_image)
        
                
    def load_model(self, model_name):
        """
        Loads a given model to be attacked

        Parameters
        ----------
        model_name : string
            name of the model being attacked.

        Returns
        -------
        model : TYPE
            computer vision model to be attacked.

        """
        if (model_name == 'CIFAR-10'): 
            return torch.hub.load("chenyaofo/pytorch-cifar-models", 
                                   "cifar10_resnet20", pretrained=True)
        elif (model_name == 'CIFAR-100'): 
            return torch.hub.load("chenyaofo/pytorch-cifar-models", 
                                   "cifar100_mobilenetv2_x1_4", pretrained=True)
        elif (model_name == 'Imagenet'):
            return torch.hub.load('pytorch/vision:v0.10.0', 
                                   'mobilenet_v2', weights=True)
    
        return None
    def get_normalization_parameters(self, model_name):
        """
        Returns the means and standard deviations of the red, green and blue 
        color channels of the image data a given model was trained on. This is 
        needed to normalize incoming images before performing inference.

        Parameters
        ----------
        model_name : string
            name of the model being attacked.

        Returns
        -------
        means : torch.Tensor
            means of the red, green and blue color channels of the 
            image data used to train the given model.
        stdevs : torch.Tensor
            standard deviations of the red, green and blue color channels of 
            the image data used to train the given model.

        """
        if (model_name == 'CIFAR-10'):
            return ([0.4914, 0.4822, 0.4465], 
                    [0.2023, 0.1994, 0.2010])
            
        elif (model_name == 'CIFAR-100'):
            return ([0.5071, 0.4867, 0.4408], 
                    [0.2673, 0.2564, 0.2762])

        elif (model_name == 'Imagenet'):
            return ([0.485, 0.456, 0.406],  
                    [0.229, 0.224, 0.225])
        
        return None, None
    
    def get_input_size(self, model_name):
        """
        Returns the hight and width of an image accepted by a given model 

        Parameters
        ----------
        model_name : string
            name of the model being attacked.

        Returns
        -------
        input_size : tuple
            hight and width of the image the model accepted as an input.

        """
        if (model_name == 'CIFAR-10') or (model_name == 'CIFAR-100'):
            input_size = (32, 32)
        elif (model_name == 'Imagenet'):
            input_size = (224, 224)
            
        return input_size
    
    def get_classes(self, model_name):
        """
        Returns the classes predicted by a given model 

        Parameters
        ----------
        model_name : string
            name of the model being attacked.

        Returns
        -------
        classes : list
            list of classes predicted by the model.

        """
        if (model_name == 'CIFAR-10'):
            return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        elif (model_name == 'CIFAR-100'): 
            return ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 
                    'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 
                    'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 
                    'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 
                    'clock', 'cloud', 'cockroach', 'couch', 'crab', 
                    'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 
                    'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                    'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard',
                    'lion', 'lizard', 'lobster', 'man', 'maple_tree', 
                    'motorcycle', 'mountain', 'mouse', 'mushroom', 
                    'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 
                    'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 
                    'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 
                    'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
                    'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 
                    'spider', 'squirrel', 'streetcar', 'sunflower', 
                    'sweet_pepper', 'table', 'tank', 'telephone', 'television',
                    'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 
                    'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        elif (model_name == 'Imagenet'):
            imagenet_classes_url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
            return requests.get(imagenet_classes_url).text.split('\n')

        return None
        
    def get_target_class(self, model_name):
        """
        Returns the index of the traget class intended by the attack

        Parameters
        ----------
        model_name : string
            name of the model being attacked.
            
        Returns
        -------
        target_class : integer
            index of the intended target class.

        """
        if (model_name == 'CIFAR-10'):
            target_class = 7 # Horse
        elif (model_name == 'CIFAR-100'):
            target_class = 15 # Camel
        elif (model_name == 'Imagenet'):
            target_class = 339 # Sorrel
            
        return target_class # She was a beautiful innocent creature!
        
    def get_pixel_budget(self, model_name):
        """
        Returns the starting point for the number of pixels 
        to be manipulated during an attack

        Parameters
        ----------
        model_name : string
            name of the model being attacked.

        Returns
        -------
        pixel_budget : integer
            number of pixels to be manipulated in an attack.

        """
        if (model_name == 'CIFAR-10') or (model_name == 'CIFAR-100'):
            return 3
        elif (model_name == 'Imagenet'):
            return 10
            
        return None
    
    def select_target(self):
        """
        Randomly selects an image from the inputs directory 

        Returns
        -------
        image_path : pathlib.Path
            path to a random image in the inputs directory.

        """

        images = [image for image in sorted(list(self.inputs_dir.iterdir())) if image.suffixes in [['.jpeg'], ['.jpg']]]
        random_number_generator = np.random.default_rng()
        image_path = random_number_generator.choice(images, size=1)[0]
        
        return image_path

    def plot_figure(self, image):
        """
        Plots a given image tensor

        Parameters
        ----------
        image : torch.Tensor
            image tensor to be plotted.

        Returns
        -------
        None.

        """

        if (type(image) == torch.Tensor):
            shape = torch.tensor(image.shape)
            dimensions = len(shape)
            hw_indicies = torch.where(shape == self.input_size[0])[0]
            if (0 not in hw_indicies) and (dimensions == 3):
                permutation = torch.cat((hw_indicies, torch.tensor([0])))
                image = image.permute(permutation.tolist())
    
        plt.figure(figsize = (4.8, 4.8))
        plt.imshow(image)
        plt.axis('off')
        plt.show();
        
    def make_prediction(self, image_tensor):
        """
        Performs inference on a given image tensor

        Parameters
        ----------
        image_tensor : torch.Tensor
            image tensor to be inferred upon.

        Returns
        -------
        prediction : string
            name of the most probable class output by the model.

        """

        # check normalization
        batch = image_tensor.permute(0, 2, 3, 1).flatten(0, 2)
        min_value, max_value = torch.aminmax(batch, dim = 0)    
        for channels in range(len(min_value)):
            min_possible = (0.0 - self.means[channels])/self.stdevs[channels]
            max_possible = (1.0 - self.means[channels])/self.stdevs[channels]
            if (min_value[channels] < min_possible) and (max_value[channels] > max_possible):
                print('Input not normalized appropriately')
                return torch.zeroslike(self.classes)
    
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            confidence, predicted_class = torch.max(outputs, 1)
        
        return self.classes[predicted_class.item()]
    
    def perturb_image(self, pixels, image):
        """
        Manipulates a given image given a list of pixel coordinates along with 
        how they are to be pertrubed in the red, green and blue channels

        Parameters
        ----------
        pixels : list
            x and y coordinates of the values in the image to be manipuated 
            as well as their red green and blue channel values.
        image : torch.Tensor
            image tensor to be manipulated.

        Returns
        -------
        image : torch.Tensor
            manipulated image

        """
        #image processing pipelines
        discrete_downsample = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),          # Converts to tensor and scales pixels to [0.0, 1.0]
            transforms.Lambda(lambda x: (x * 255)),
            transforms.ToDtype(torch.uint8, scale = False)
        ])
        continuous = transforms.Compose([
            transforms.ToDtype(torch.float32, scale=True)
            ])
        image = discrete_downsample(image)
            
        # At pixel's x,y position, assign its rgb value
        pixels = pixels.reshape(-1, 5)
        for pixel in pixels:
            x_pos, y_pos, *rgb = pixel
            x_pos, y_pos = [torch.clamp(torch.tensor(p), min = 0, max = self.input_size[0] - 1).long() for p in (x_pos, y_pos)]
            rgb = torch.Tensor(rgb)
            rgb = torch.clamp(rgb, min=0, max=255)
            image[:, x_pos, y_pos] = rgb
            
        image = continuous(image)
        
        return image
    
    def objective_function(self, pixels, image, target):
        """
        The function scipy's differential evolution 
        algorithm works to optimize

        Parameters
        ----------
        ppixels : list
            x and y coordinates of the values in the image to be manipuated 
            as well as their red green and blue channel values.
        image : torch.Tensor
            image tensor to be manipulated.
        target : integer
            index of the intended target class.

        Returns
        -------
        target_probability : float
            negative of the target classes probability.

        """
    
        perturbed_image = self.perturb_image(pixels, image)
        normalize = transforms.Compose([
            transforms.Normalize(mean=self.means, 
                                 std=self.stdevs)
        ])
        perturbed_image = normalize(perturbed_image)
        perturbed_image = perturbed_image.unsqueeze(0)
        perturbed_image = perturbed_image.to(self.device)
        
        with torch.no_grad():
            logits = self.model(perturbed_image)
            probabilities = F.softmax(logits, dim = 1)
        
        return -probabilities[0, target].cpu().item()
    
    def n_pixel_attack(self, image, target_class, pixel_budget, 
                       maxiter = 50, population_size = 20):
        """
        Performs differential evolution to find the optimal pixel manipulations 
        to induce the desired missclassification for a given pixel budget

        Parameters
        ----------
        image : torch.Tensor
            image tensor to be manipulated..
        target_class : TYPE
            index of the intended target class.
        pixel_budget : integer
            the number of pixels to be manipulated in the attack.
        maxiter : TYPE, optional
            the number of generations the differential evolution 
            algorithm is run for. The default is 50.
        population_size : TYPE, optional
            the number of points comprising a generation. 
            The default is 20.

        Returns
        -------
        evolutionary_line : TYPE
            the most successful mutation of every generation.

        """
    
        evolutionary_line = []
        generation_callback = Tracking(itterations = maxiter)
        bounds = [(0,self.input_size[0] - 1), (0,self.input_size[1] - 1), (0,255), (0,255), (0,255)] * pixel_budget
        try:
            result = differential_evolution(
                self.objective_function,
                bounds,
                args=(image, target_class),
                maxiter = maxiter,
                popsize = population_size,
                #strategy = 'rand2bin',
                callback=generation_callback,
            )
            evolutionary_line = generation_callback.parameter_history
        finally:
            generation_callback.close()
            
        return evolutionary_line
    
    def tune_attack(self, image, target_class, 
                    search_steps = 3, pixel_budget = 10):
        """
        Performs binary search to minimize the number of pixels 
        to manipulate to induce the target misclassififcation 

        Parameters
        ----------
        image : torch.Tensor
            image tensor to be manipulated.
        target_class : integer
            index of the intended target class..
        search_steps : integer, optional
            the number of times binary search is performed.
            The default is 3.
        pixel_budget : integer optional
            starting number of pixels to be manipulated in 
            an attack. The default is 10.

        Returns
        -------
        successful_history : list
            list of each generation's the most successful mutation..

        """

        yet_to_converge = True
        final_perturbation = None
        successful_history = None
    
        lower_bound = 1
        upper_bound = pixel_budget
        
        for step in trange(search_steps):
            evolutionary_line = self.n_pixel_attack(image, target_class, pixel_budget = pixel_budget)
            
            # apply perturbation
            final_mutation = evolutionary_line[-1]
            final_perturbation = self.perturb_image(final_mutation, target)
            normalize_tensor = self.model_normalize(final_perturbation)
            perturbed_image = normalize_tensor.unsqueeze(0)
            perturbed_image = perturbed_image.to(self.device)
    
            # check if attack was successful
            with torch.no_grad():
                logits = self.model(perturbed_image)
                probabilities = F.softmax(logits, dim = 1)
            confidence, predicted_class = torch.max(probabilities, 1)
            converged = (predicted_class == target_class)
    
            # tune pixel buget
            if (converged == True):
                yet_to_converge = False
                upper_bound = pixel_budget
                successful_history = evolutionary_line
            elif (converged == False) and (yet_to_converge == False):
                lower_bound = pixel_budget
            elif (converged == False) and (yet_to_converge == True):
                lower_bound = pixel_budget
                upper_bound *= 4
    
            if (lower_bound == upper_bound):
                break
            pixel_budget = int((upper_bound + lower_bound)/2)
    
        return successful_history
    
    def animate_evolution(self, target_path, history):
        """
        Animates the attack 

        Parameters
        ----------
        target_path : pathlib.Path
            path to the image ,amipulated as part of the attack
        history : list
            list of each generation's the most successful mutation.

        Returns
        -------
        filepath : pathlib.Path
            path to the animation of the attack.

        """
        evolution_gif = []
    
        target = Image.open(target_path)
        target = self.downsample(target)
    
        for generation, mutation in enumerate(history):
            if np.NaN in mutation:
                npa = self.downsample(target)
            else:
                npa = self.perturb_image(mutation, target)
            
            normalize_tensor = self.model_normalize(npa)
            perturbed_image = normalize_tensor.unsqueeze(0)
            perturbed_image = perturbed_image.to(self.device)
            with torch.no_grad():
                logits = self.model(perturbed_image)
                probabilities = F.softmax(logits, dim = 1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            title = f"Generation: {generation}\nPredicted Class: {self.classes[predicted_class.item()]} with {confidence.item(): .3f} confidence\nTarget Class: {self.classes[self.target_class]} with {probabilities[0, self.target_class]: .3f} confidence"
            
            shape = torch.tensor(npa.shape)
            dimensions = len(shape)
            hw_indicies = torch.where(shape == self.input_size[0])[0]
            if (0 not in hw_indicies) and (dimensions == 3):
                permutation = torch.cat((hw_indicies, torch.tensor([0])))
                npa = npa.permute(permutation.tolist())
            
            plt.figure(figsize = (4.8, 4.8))
            plt.title(title, loc="left")
            plt.imshow(npa)
            plt.axis('off')
        
            # Convert the plot into an Image object
            buf = io.BytesIO()
            plt.savefig(buf, format='jpg', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            cam = Image.open(buf)
            evolution_gif.append(cam)
        
        pixel_count = int(len(mutation)//5)
        filename = f"{target_path.name.split('.')[0]}_{self.model_name}_{pixel_count}_pixel_attack.gif"
        filepath = Path(self.outputs_dir)/ filename
        evolution_gif[0].save(filepath, save_all = True, 
                          append_images = evolution_gif[1:], duration=100, loop=0)
    
        return filepath
    
if __name__ == "__main__":
    ralph_cifar10 = RalphCIFAR10o('CIFAR-10')
    ralph_cifar100 = RalphCIFAR10o('CIFAR-100')
    ralph_imagenet = RalphCIFAR10o('Imagenet')
    
    models = [ralph_cifar10, ralph_cifar100, ralph_imagenet]
    for model in models:
        images = [image for image in sorted(list(model.inputs_dir.iterdir())) if image.suffixes in [['.jpeg'], ['.jpg']]]
        
        gif_paths = []
        for image_path in tqdm(images):
            target = Image.open(image_path)
            history = model.tune_attack(target, model.target_class, pixel_budget = model.pixel_budget)
            gif_paths.append(model.animate_evolution(image_path, history))
            
        