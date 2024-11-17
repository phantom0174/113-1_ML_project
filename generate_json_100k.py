# thanks chatgpt

import json
import pandas as pd

def csv_to_json(csv_file, json_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    df = df.dropna(subset=['formula', 'image'])
    
    # Create a dictionary from the DataFrame with 'img_name' as keys and 'formula' as values
    data = pd.Series(df['formula'].values, index=df['image']).to_dict()
    
    # Write the dictionary to a JSON file
    with open(json_file, mode='w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

for type in ['test', 'train', 'validate']:
    csv_to_json(f'./100k/im2latex_{type}.csv', f'img_mapping_{type}_100k.json')
