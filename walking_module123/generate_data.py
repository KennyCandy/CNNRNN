from __future__ import print_function
import tensorflow as tf
import numpy as np
import math as math
import argparse



#
# parser = argparse.ArgumentParser()
# parser.add_argument('dataset')
# args = parser.parse_args()

print("first")
exit
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

FLAG = 'make_train_'
NUMBER_CLASSES = 6
skip_header_lines = 0;
DATA_PATH = '../data/Walking_dataset/data_ori/'

USER_NUMBER = '2'
filename = DATA_PATH + USER_NUMBER + '.csv'

import csv
def read_file_csv (filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        your_list = list(reader)
        return your_list

full_data = read_file_csv(filename)

thefile1 = open('body_acc_x_train.txt', 'a+')
thefile2 = open('body_acc_y_train.txt', 'a+')
thefile3 = open('body_acc_z_train.txt', 'a+')

thefile4 = open('body_acc_x_test.txt', 'a+')
thefile5 = open('body_acc_y_test.txt', 'a+')
thefile6 = open('body_acc_z_test.txt', 'a+')

thefile_y_train = open('y_train.txt', 'a+')
thefile_y_test = open('y_test.txt', 'a+')


print(file_len(filename))
print(FLAG)
for m in range(NUMBER_CLASSES):
    m = m + 1
    if( FLAG == 'make_train'):
        filename = DATA_PATH + str(m) + '.csv'
    else:
        filename = DATA_PATH + str(m) + '_test.csv'

    full_data = read_file_csv(filename)
    for j in range(file_len(filename)):
        if(j/127 == j%127):
            if( FLAG == 'make_train'):
                thefile_y_train.write("%s \n" %m )
            else:
                thefile_y_test.write("%s \n" %m )
            if(j==0):
                if( FLAG == 'make_train'):
                    thefile1.write("%s" %full_data[j][1])
                    thefile2.write("%s" %full_data[j][2])
                    thefile3.write("%s" %full_data[j][3])
                else:
                    thefile4.write("%s" %full_data[j][1])
                    thefile5.write("%s" %full_data[j][2])
                    thefile6.write("%s" %full_data[j][3])
            else:
                if( FLAG == 'make_train'):
                    thefile1.write("\n %s" %full_data[j][1])
                    thefile2.write("\n %s" %full_data[j][2])
                    thefile3.write("\n %s" %full_data[j][3])
                else:
                    thefile4.write("\n %s" %full_data[j][1])
                    thefile5.write("\n %s" %full_data[j][2])
                    thefile6.write("\n %s" %full_data[j][3])
        else:
            if( FLAG == 'make_train'):
                thefile1.write(" %s" %full_data[j][1])
                thefile2.write(" %s" %full_data[j][2])
                thefile3.write(" %s" %full_data[j][3])
            else:
                thefile4.write(" %s" %full_data[j][1])
                thefile5.write(" %s" %full_data[j][2])
                thefile6.write(" %s" %full_data[j][3])
    if( FLAG == 'make_train'):
        thefile1.write(" \n")
        thefile2.write(" \n")
        thefile3.write(" \n")
    else:
        thefile4.write(" \n")
        thefile5.write(" \n")
        thefile6.write(" \n")
