import json

with open('LaTex_data/corresponding_png_images.txt', 'r', encoding='utf-8') as images_file:
    images = images_file.readlines()

with open('LaTex_data/final_png_formulas.txt', 'r', encoding='utf-8') as formulas_file:
    formulas = formulas_file.readlines()

image_formula_map = {}
for image, formula in zip(images, formulas):
    image = image.strip()  
    formula = formula.strip()  
    image_formula_map[image] = formula  

with open('image_formula_mapping.json', 'w', encoding='utf-8') as json_file:
    json.dump(image_formula_map, json_file, ensure_ascii=False, indent=4)

print("JSON file created successfully!")
