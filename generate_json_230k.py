import json
import os
with open('LaTex_data/corresponding_png_images.txt', 'r', encoding='utf-8') as images_file:
    images = images_file.readlines()

with open('LaTex_data/final_png_formulas.txt', 'r', encoding='utf-8') as formulas_file:
    formulas = formulas_file.readlines()

print(len(images), len(formulas))
folders = ["split_1", "split_2", "split_3", "split_4", "split_5", 
           "split_6", "split_7", "split_8", "split_9", "split_10"]

image_formula_map = {}
for i in range(len(images)):
    image_formula_map[images[i].strip()] = formulas[i].strip()
print(len(image_formula_map))
with open('image_formula_mapping.json', 'w', encoding='utf-8') as json_file:
    json.dump(image_formula_map, json_file, ensure_ascii=False, indent=4)

print("JSON file created successfully!")

keys = list(image_formula_map.keys())
total = 0
for folder in folders:
    images = os.listdir('LaTex_data/' + folder)
    total += len(images)
print(total)
