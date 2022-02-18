import csv

def load_data(filepath):
    lst = []
    with open('Pokemon.csv') as file:
        reader = csv.DictReader(file)
        for row in reader:
            row['HP'] = int(row['HP'])
            row['Attack'] = int(row['Attack'])
            row['Defense'] = int(row['Defense'])
            row['Sp. Atk'] = int(row['Sp. Atk'])
            row['Sp. Def'] = int(row['Sp. Def'])
            row['Speed'] = int(row['Speed'])
            lst.append(row)
    return lst


def calculate_x_y(row):
    info = data[row]
    x = info['Attack'] + info['Sp. Atk'] + info['Speed']
    y = info['Defense'] + info['Sp. Def'] + info['HP']
    return [x,y]