import math  
from collections import defaultdict 
import operator
from csv import reader

dataset = {}

def euclidian_distance(p1, p2):
    sum_dimensions_differences = 0
    for i in range(0, len(p1)-1):
        sum_dimensions_differences += (float(p1[i]) - float(p2[i]))**2

    return math.sqrt(sum_dimensions_differences)

def sort_neighbours(dataset, point):
    distances = []
    for p in dataset:
        distance = euclidian_distance(point, p) 
        distances.append((p, distance))

    sorted_neighbours = sorted(distances, key=lambda x: x[1])
    #print("sorted_neighbours:", sorted_neighbours)
    filtered_neighbours = [el[0] for el in sorted_neighbours]
    return filtered_neighbours

def get_k_neighbours(dataset, point, k):
    neighbours = sort_neighbours(dataset,point)
    return neighbours[0:k]

def knn(dataset, point, k):
    k_neighbours = get_k_neighbours(dataset, point, k)
    dic = defaultdict(lambda: 0) 
    for neighbour in k_neighbours: 
        dic[neighbour[-1]] +=1

    #print(k_neighbours)
    #print(dic)
    return max(dic.items(), key=operator.itemgetter(1))[0]


def read_training_dataset():
    with open('Dataset_CancerClassification/Dados_Normalizados/cancer_train.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Get all rows of csv from csv_reader object as list of tuples
        list_of_tuples = list(map(tuple, csv_reader))
        
        # exclude first line (headers)
        return list_of_tuples[1:] 
    

dataset = read_training_dataset()

print(knn(dataset, (0,0), 3))