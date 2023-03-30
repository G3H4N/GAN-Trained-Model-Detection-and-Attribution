file_data = ""
with open("/data/gehan/Datasets/CelebA/list_attr_celeba.txt", "r") as f:
    for line in f:
        line = line.replace("jpg", "png")
        file_data += line
with open("/data/gehan/Datasets/CelebA/list_attr_celeba_png.txt", "w") as f:
    f.write(file_data)
print('finish')