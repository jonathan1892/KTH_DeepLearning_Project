# Presentation

This repository contains the source code and pdf report for project of the Deep Learning course at KTH. This project was done in collaboration with Sarah Berenji and Dennis Malmgren, and consists of image style transfer.

# Code structure

The two main source files are as follows :    
- The code for the style transfer algorithm can be found in `src/styleTransfer.py`.      
- `ScipyOptimizer.py` simply acts as an interface between Keras/Tensorflow and scipy. It is a dependency used by `styleTransfer.py`.

Moreover :
- The file `batchTransfer.py` allows us to batch run the style transfer algorithm.  
- `denoise.py` performs total variation denoising on an input image.   
- `transform_caffe_vgg_normalized_gatys_weights.py` loads gatys original normalized weights into a caffe model and transforms them to tensorflow format.
- `weight_normalizer.py` samples the activation across a directory of images and normalizes VGG19 weights to have unit mean activation across all images, all positions.
- `vgg19_loader.py` loads a VGG19 model with max pool layers replaced by average pool layers.


# Running the code

First, make sure that Tensorflow and Keras are properly installed. Install all the requirements in the `requirements.txt` file with pip.

In order to perform style transfer on a singleimage, simply run the following in the terminal : `python3 styleTransfer.py -s <pathToStyleImage> -c <pathToContentImage>`.    
Some other options are also available, type `python3 styleTransfer.py --help` for details.   

If you wish to automate the style transfer process and schedule several runs of the algorithm, modify and run `batchTransfer.py` accordingly. Details are provided in the source file.

Some sample images used in the report have been provided in the `images` folder.

# Results

This section only showcases some of the results obtained with our implementation. For more details regarding the theory, implementation choices and the obtained results please refer to the pdf report.     
The following images where generated with "Stockholm" as content picture (`/images/input/stockholm.jpg`) and various style images (see report for details).

![example of style transfer](https://raw.githubusercontent.com/jojo38000/KTH_DL_Proj/master/reports/style_transfer_introduction.jpg) 

![example of style transfer](https://raw.githubusercontent.com/jojo38000/KTH_DL_Proj/master/reports/Image_Results/best_gatys/b6_stockholm_femme_iter_200_conv2_2_r10.0_s1000.0_c5.0.png)

![example of style transfer](https://raw.githubusercontent.com/jojo38000/KTH_DL_Proj/master/reports/Image_Results/best_gatys/b7_stockholm_composition_iter_200_conv2_2_r10.0_s1000.0_c5.0.png)

![example of style transfer](https://raw.githubusercontent.com/jojo38000/KTH_DL_Proj/master/reports/Image_Results/based_on_tutorial/b7_stockholm_chaos_iter_200_r0.1_s5.0_c0.025.png)

![example of style transfer](https://raw.githubusercontent.com/jojo38000/KTH_DL_Proj/master/reports/Image_Results/based_on_tutorial/b6_stockholm_waves_iter_200_r0.1_s5.0_c0.025.png)

# Report

The report can be found at `/report.pdf`. It provides a theoretical background regarding style transfer, details the implemention choices and discusses the various results obtained.
