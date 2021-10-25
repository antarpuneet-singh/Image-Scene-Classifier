# Image-Scene-Classifier
One of the core problems in computer vision is classification. In this project, I have tried to classify the scenes in various categories of images using the "Bag of words" technique.
It's based on the work done in this research paper :
S. Lazebnik, C. Schmid, and J. Ponce, “Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories”. CVPR, 2006.

# Data
The data set contains 1491 images from the SUN image database, divided into 8 categories. The aim of the project is to build a classifier that can detect which category an image belongs to.

# Approach
I have used the "bag of words" approach to solve this problem. "Bag of words" is a very popular technique used in Natural Language Processing. To use it in the image domain, I have converted every pixel in the image into a high dimensional representation by applying a set of image convolution filters on the pixels. Then these high dimensional features are clustered by using K-means algorithm. The group of cluster centres gives us our dictionary of visual words. We can then use the nearest neighbour classifier to categorize the pixels.

