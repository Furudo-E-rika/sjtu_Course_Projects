#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from PIL import Image

def detect_edges(image):
  """Find edge points in a grayscale image.

  Args:
  - image (2D uint8 array): A grayscale image.

  Return:
  - edge_image (2D float array): A heat map where the intensity at each point
      is proportional to the edge magnitude.
  """
  #TODO
  np.pad(image, ((1,1), (1,1)),'constant', constant_values=(0,0))
  hei, wid = image.shape[0], image.shape[1]
  edge_image = np.zeros((hei,wid))
  for y in range(1,hei-1):
    for x in range(1,wid-1):
        Gx = -1*image[y-1][x-1] + image[y+1][x+1] - 2*image[y][x-1] + 2*image[y][x+1] - image[y+1][x-1] + image[y+1][x+1]
        Gy = image[y-1][x-1] + 2*image[y-1][x] + image[y-1][x+1] - image[y+1][x-1] - 2*image[y+1][x] - image[y+1][x+1]
        magnitude = abs(Gx) + abs(Gy)
        edge_image[y][x] = magnitude
  return edge_image



def hough_circles(edge_image, edge_thresh, radius_values):
  """Threshold edge image and calculate the Hough transform accumulator array.

  Args:
  - edge_image (2D float array): An H x W heat map where the intensity at each
      point is proportional to the edge magnitude.
  - edge_thresh (float): A threshold on the edge magnitude values.
  - radius_values (1D int array): An array of R possible radius values.

  Return:
  - thresh_edge_image (2D bool array): Thresholded edge image indicating
      whether each pixel is an edge point or not.
  - accum_array (3D int array): Hough transform accumulator array. Should have
      shape R x H x W.
  """
  #TODO
  hei, wid = edge_image.shape
  tresh_edge_image = np.zeros((hei,wid))
  tresh_edge_image[edge_image > edge_thresh] = True
  tresh_edge_image[edge_image <= edge_thresh] = False
  
  accum_array = np.zeros((len(radius_values), hei, wid))
  thetas = np.deg2rad(np.arange(0, 360))
  cos_thetas = np.cos(thetas)
  sin_thetas = np.sin(thetas)

  for i in range(len(radius_values)):
    for y in range(hei):
      for x in range(wid):
        if tresh_edge_image[y][x] :
          for theta in range(len(thetas)):
            x0 = int(x + radius_values[i]*cos_thetas[theta])
            y0 = int(y + radius_values[i]*sin_thetas[theta])
          
            if (x0 >=0 and x0 < wid and y0 >= 0 and y0 < hei):
              accum_array[i][y0][x0] += 1

  return tresh_edge_image, accum_array


def find_circles(image, accum_array, radius_values, hough_thresh):
  """Find circles in an image using output from Hough transform.

  Args:
  - image (3D uint8 array): An H x W x 3 BGR color image. Here we use the
      original color image instead of its grayscale version so the circles
      can be drawn in color.
  - accum_array (3D int array): Hough transform accumulator array having shape
      R x H x W.
  - radius_values (1D int array): An array of R radius values.
  - hough_thresh (int): A threshold of votes in the accumulator array.

  Return:
  - circles (list of 3-tuples): A list of circle parameters. Each element
      (r, y, x) represents the radius and the center coordinates of a circle
      found by the program.
  - circle_image (3D uint8 array): A copy of the original image with detected
      circles drawn in color.
  """
  #TODO
  circles = []
  circle_image = image.copy()
  thresh_ryx = np.argwhere(accum_array > hough_thresh)
  for r_y_x in thresh_ryx:
    y, x = r_y_x[1], r_y_x[2]
    r = radius_values[r_y_x[0]]
    ryx_tuple = (r, y, x)
    circles.append(ryx_tuple)
    cv2.circle(circle_image ,(x, y), r, color=(0,255,0), thickness=2)
  return circles, circle_image


if __name__ == '__main__':
  #TODO
  img_name = 'coins'
  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edge_thresh = 400
  hough_thresh = 220
  # Plot heat map
  edge_image = detect_edges(gray_image)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  im = ax.imshow(edge_image, cmap=plt.cm.hot_r)
  plt.colorbar(im)
  plt.title('Sobel heat map')
  plt.savefig('./output/' + img_name + '_edge_magnitude.png')
  radius_values = np.arange(20,50)
  tresh_edge_image, accum_array = hough_circles(edge_image,edge_thresh=edge_thresh,radius_values=radius_values)
  circles, circle_image = find_circles(image = img, accum_array = accum_array, radius_values=radius_values, hough_thresh=hough_thresh)
  tresh_edge_image[tresh_edge_image == 1] = 255
  cv2.imwrite('output/' + img_name + "_edges.png", tresh_edge_image)
  cv2.imwrite('output/' + img_name + "_circles.png", circle_image)

  print("-"*20+"circles:"+"-"*20)
  print(circles)


