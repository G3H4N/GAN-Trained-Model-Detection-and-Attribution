# -*- coding: utf-8 -*-
# !/usr/bin/env python3

'''
Divide face accordance CelebA Attr type.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import os

image_path = "/data/gehan/Datasets/CelebA/png/align/img_align_celeba_png" # has to be absolute path
CelebA_Attr_file = "/data/gehan/Datasets/CelebA/list_attr_celeba_png.txt"

'''
Attr_type0 = 22  # mouth_slightly_open
Attr_type1 = 32  # smiling
Attr_type2 = 37  # Wearing_Lipstick
'''
Attr_type0 = 20  # high cheekbones
Attr_type1 = 21  # male
Attr_type2 = 22  # mouth_slightly_open

output_path = "/data/gehan/Datasets/CelebA/png/align/2Split_align_Attr" + '_' + str(Attr_type2) + '_' + str(Attr_type1) + '_' + str(Attr_type0)


def main():
    train000_dir = os.path.join(output_path, "000")
    if not os.path.isdir(train000_dir):
        os.makedirs(train000_dir)
    train001_dir = os.path.join(output_path, "001")
    if not os.path.isdir(train001_dir):
        os.makedirs(train001_dir)
    train010_dir = os.path.join(output_path, "010")
    if not os.path.isdir(train010_dir):
        os.makedirs(train010_dir)
    train011_dir = os.path.join(output_path, "011")
    if not os.path.isdir(train011_dir):
        os.makedirs(train011_dir)
    train100_dir = os.path.join(output_path, "100")
    if not os.path.isdir(train100_dir):
        os.makedirs(train100_dir)
    train101_dir = os.path.join(output_path, "101")
    if not os.path.isdir(train101_dir):
        os.makedirs(train101_dir)
    train110_dir = os.path.join(output_path, "110")
    if not os.path.isdir(train110_dir):
        os.makedirs(train110_dir)
    train111_dir = os.path.join(output_path, "111")
    if not os.path.isdir(train111_dir):
        os.makedirs(train111_dir)

    not_found_txt = open(os.path.join(output_path, "not_found_img.txt"), "w")

    count_000 = 0
    count_001 = 0
    count_010 = 0
    count_011 = 0
    count_100 = 0
    count_101 = 0
    count_110 = 0
    count_111 = 0

    count_N = 0

    with open(CelebA_Attr_file, "r") as Attr_file:
        Attr_info = Attr_file.readlines()
        Attr_info = Attr_info[2:] # [0] total number of samples, [1] list of all attributes
        index = 0
        for line in Attr_info:
            index += 1
            info = line.split()
            filename = info[0]
            filepath_old = os.path.join(image_path, filename)
            if os.path.isfile(filepath_old):
                if int(info[Attr_type2]) != 1:
                    if int(info[Attr_type1]) != 1:
                        if int(info[Attr_type0]) != 1:
                            filepath_new = os.path.join(train000_dir, filename)
                            shutil.copyfile(filepath_old, filepath_new)
                            count_000 += 1
                        else:
                            filepath_new = os.path.join(train001_dir, filename)
                            shutil.copyfile(filepath_old, filepath_new)
                            count_001 += 1
                    else:
                        if int(info[Attr_type0]) != 1:
                            filepath_new = os.path.join(train010_dir, filename)
                            shutil.copyfile(filepath_old, filepath_new)
                            count_010 += 1
                        else:
                            filepath_new = os.path.join(train011_dir, filename)
                            shutil.copyfile(filepath_old, filepath_new)
                            count_011 += 1
                else:
                    if int(info[Attr_type1]) != 1:
                        if int(info[Attr_type0]) != 1:
                            filepath_new = os.path.join(train100_dir, filename)
                            shutil.copyfile(filepath_old, filepath_new)
                            count_100 += 1
                        else:
                            filepath_new = os.path.join(train101_dir, filename)
                            shutil.copyfile(filepath_old, filepath_new)
                            count_101 += 1
                    else:
                        if int(info[Attr_type0]) != 1:
                            filepath_new = os.path.join(train110_dir, filename)
                            shutil.copyfile(filepath_old, filepath_new)
                            count_110 += 1
                        else:
                            filepath_new = os.path.join(train111_dir, filename)
                            shutil.copyfile(filepath_old, filepath_new)
                            count_111 += 1
                print("%d: success for copy %s %s %s -> %s" % (index, info[Attr_type2], info[Attr_type1], info[Attr_type0], filepath_new))
            else:
                print("%d: not found %s\n" % (index, filepath_old))
                not_found_txt.write(line)
                count_N += 1

    not_found_txt.close()

    print("000 have %d images!" % count_000)
    print("001 have %d images!" % count_001)
    print("010 have %d images!" % count_010)
    print("011 have %d images!" % count_011)
    print("100 have %d images!" % count_100)
    print("101 have %d images!" % count_101)
    print("110 have %d images!" % count_110)
    print("111 have %d images!" % count_111)
    print("Not found %d images!" % count_N)


if __name__ == '__main__':
    main()