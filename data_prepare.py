import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  os

edges_list_folder = os.getcwd() + "\CPMN\edge_list"
file_graph_label_path = os.getcwd() + "\CPMN\labels.txt"

file_label = open(file_graph_label_path, 'r')
list_indicator = []
for line in file_label.readlines():
    ss = line.strip()
    list_indicator.append(ss)
file_label.close()
graph_num = [i for i in range(1, 428+1)]
df_node_graph_indicator = pd.DataFrame({'node': graph_num, 'indicator': list_indicator})
dict_node_graph_indicator = df_node_graph_indicator.set_index('node')['indicator'].to_dict()
print(dict_node_graph_indicator)

work = 0
live = 0
for root, dirs, files in os.walk(edges_list_folder):
    for graph in range(1, 429):
        print("Deal with sample No." + str(graph))
        matrix = []
        grid_id_dict = {}
        for t in range(1,10):
            A = "edgelist" + str(t) + ".csv"
            print(A)
            A = os.path.join(root, A)
            with open(A, "r", encoding='utf-8') as f:
                next(f)
                list_graph = []
                grid_id = 0
                for i, line in enumerate(f.readlines(), 1):
                    line_list = line.split(",")
                    if line_list[0]!="":
                        from_grid = int(line_list[0])
                        to_grid = int(line_list[1])
                        graph_id = int(line_list[3])
                        value = int(line_list[4])
                        if graph==graph_id and value!=0:
                            grid_id += 1
                            if from_grid not in grid_id_dict:
                                grid_id_dict[from_grid] = grid_id
                            if to_grid not in grid_id_dict:
                                grid_id_dict[to_grid] = grid_id
                            tup = (grid_id_dict[from_grid], grid_id_dict[to_grid], value)
                            list_graph.append(tup)
                print(list_graph)

                try:
                    n_x = max(user for user, item, rating in list_graph)
                except Exception as e:
                    print(e)
                    n_x = 1
                try:
                    n_y = max(item for user, item, rating in list_graph)
                except Exception as e:
                    print(e)
                    n_y = 1
                matrix_t = np.zeros((n_x, n_y))
                for user, item, rating in list_graph:
                    matrix_t[user - 1][item - 1] = rating
                matrix_t = matrix_t[0].tolist()
                matrix.append(matrix_t)
        matrix_long = max(len(l) for l in matrix)
        for l in matrix:
            if len(l)<matrix_long:
                for i in range(matrix_long-len(l)):
                    l.append(0)
        plt.imshow(matrix, cmap="gray")
        plt.axis('off')
        fig = plt.gcf()
        fig.set_size_inches(len(matrix[0]), len(matrix))
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        if dict_node_graph_indicator[graph]=="1": 
            work += 1
            image_name = "work." + str(work) + '.jpg'
        if dict_node_graph_indicator[graph]=="2": 
            live += 1
            image_name = "live." + str(live) + '.jpg'
        fig.savefig(os.getcwd() + '\CPMN\images/'
                    + image_name, format='jpg', transparent=True, dpi=1, pad_inches=0)
        plt.show()
