#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import math
from utils.p1_b import Union_Find, first_pass, second_pass

def binarize(gray_image, thresh_val):
  # TODO: 255 if intensity >= thresh_val else 0
  wid, hei = gray_image.shape[0], gray_image.shape[1]
  binary_image = np.empty([wid, hei])
  binary_image[gray_image >= thresh_val] = 255
  binary_image[gray_image < thresh_val] = 0

  return binary_image

def label(binary_image):
  # TODO
  uf_set = Union_Find(30000)
  binary_image = binary_image.astype('int')
  binary_image[binary_image == 255] = 1
  first_image = first_pass(binary_image, uf_set)
  labeled_image = second_pass(first_image, uf_set)
  unique = np.unique(labeled_image)
  for i in range(len(unique)):
    labeled_image[labeled_image == unique[i]] = i
  max_value = labeled_image.max()
  labeled_image = labeled_image * 255 // max_value
  return labeled_image


def get_attribute(labeled_image):
  # TODO
  hei, wid = labeled_image.shape[0], labeled_image.shape[1]
  attribute_list = []
  components = np.unique(labeled_image)
  
  
  for component in components:
    if component == 0:
      continue
    attribute = {}
    x_list = []
    y_list = []
    # position
    for y in range(hei):
      for x in range(wid):
        if labeled_image[y][x] == component :
          x_list.append(x)
          y_list.append(y)
    x_list = np.array(x_list, dtype='float')
    y_list = np.array(y_list, dtype='float')
    x_avg = x_list.mean()
    y_avg = y_list.mean() 
    attribute["position"] = (x_avg, hei-1-y_avg)
    # orientation
    a = 0
    b = 0
    c = 0
    for i, j in zip(x_list, y_list):
      a += (i - x_avg)**2
      b += 2 * (i - x_avg) * (y_avg - j)
      c += (j - y_avg)**2
    theta_1 = math.atan(b / (a-c)) / 2
    theta_2 = theta_1 + math.pi/2
    attribute["orientation"] = theta_1
    # roundedness
    E_min = a*(math.sin(theta_1)**2) - b*math.sin(theta_1)*math.cos(theta_1) + c*(math.cos(theta_1)**2)
    E_max = a*(math.sin(theta_2)**2) - b*math.sin(theta_2)*math.cos(theta_2) + c*(math.cos(theta_2)**2)
    if E_min <= E_max:
      roundedness = E_min / E_max
    else:
      roundedness = E_max / E_min
    attribute["roundedness"] = roundedness
    attribute_list.append(attribute)

  return attribute_list


def main(argv):
  img_name = argv[0]
  thresh_val = int(argv[1])
  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  binary_image = binarize(gray_image, thresh_val=thresh_val)
  labeled_image = label(binary_image)
  attribute_list = get_attribute(labeled_image)

  cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
  cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
  cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)
  print(attribute_list)


if __name__ == '__main__':
  main(sys.argv[1:])
