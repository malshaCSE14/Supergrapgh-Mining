import operator, csv
from sklearn.cluster import KMeans
import numpy as np
class Data(object):
    def __init__(self, name):
        self.__name = name
        self.__links = set()

    @property
    def name(self):
        return self.__name

    @property
    def links(self):
        return set(self.__links)

    def add_link(self, other):
        self.__links.add(other)
        other.__links.add(self)
def getVehicleDensities():
    vehicle_density = {}
    with open("averageSpeeds.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            # print(" ".join(row))
            try:
                vehicle_density[row[0]] = float(row[2])
            except ValueError:
                pass

    sorted_vehicle_densities = sorted(vehicle_density.items(), key=operator.itemgetter(1))
    vehicle_density = dict(sorted_vehicle_densities)
    return vehicle_density

def kmeans(vehicle_density,k):
    vehicle_density_list = list(vehicle_density.values())
    road_segments = list(vehicle_density.keys())
    x = np.array(vehicle_density_list)
    km = KMeans(n_clusters=k)
    km.fit(x.reshape(-1, 1))

    dict_clusters = {}
    dict_road_clusters = {}
    kmlabels = list(km.labels_)

    for index in range(len(kmlabels)):
        if kmlabels[index] in dict_clusters:
            dict_clusters[kmlabels[index]].append(vehicle_density_list[index])
            dict_road_clusters[kmlabels[index]].append(road_segments[index])
        else:

            dict_clusters[kmlabels[index]] = [vehicle_density_list[index]]
            dict_road_clusters[kmlabels[index]] = [road_segments[index]]
    # print(dict_clusters)
    return dict_road_clusters

def random_adjacency(n):
    adjacency = np.random.randint(0, 2, (n, n))
    for i in range(n):
        for j in range(n):
            if i==j:
                adjacency[i][j] = 1
            else:
                adjacency[i][j] = adjacency[j][i]
            if i>=j:
                pass
    return adjacency

def connected_components(nodes):
    # List of connected components found. The order is random.
    result = []

    # Make a copy of the set, so we can modify it.
    nodes = set(nodes)

    # Iterate while we still have nodes to process.
    while nodes:

        # Get a random node and remove it from the global set.
        n = nodes.pop()

        # This set will contain the next group of nodes connected to each other.
        group = {n}

        # Build a queue with this node in it.
        queue = [n]

        # Iterate the queue.
        # When it's empty, we finished visiting a group of connected nodes.
        while queue:
            # Consume the next item from the queue.
            n = queue.pop(0)

            # Fetch the neighbors.
            neighbors = n.links

            # Remove the neighbors we already visited.
            neighbors.difference_update(group)

            # Remove the remaining nodes from the global set.
            nodes.difference_update(neighbors)

            # Add them to the group of connected nodes.
            group.update(neighbors)

            # Add them to the queue, so we visit them in the next iterations.
            queue.extend(neighbors)

        # Add the group to the list of groups.
        result.append(group)

    # Return the list of groups.
    return result

def connected_component_count(nodesList, Ag):
    nodeNames = []

    for nodeIndex in nodesList:
        nodeNames.append('n'+str(nodeIndex))
    node_dict = {}
    nodes = {Data(x) for x in nodeNames}
    for eachn in nodes:
        node_dict[int(eachn.name[1:])] = eachn

    if __name__ == "__main__":
        for i in nodesList:
            for j in nodesList:
                if int(i) >= int(j):
                    pass
                elif Ag[int(i)-1][int(j)-1] == 1:
                    # print(i, j)
                    node_dict[int(i)].add_link(node_dict[int(j)])

        # Find all the connected components.
        number = 0
        for components in connected_components(nodes):
            names = sorted(node.name for node in components)
            names = ", ".join(names)
            # print("Group #%i: %s" % (number, names))
            number += 1
        return number

def clustering_gain(cluster):
    return 1
def ratio_error_sum(cluster):
    return 1
def mcg(c):
    nd= len(c)
    mcg =0
    global_means = []
    for cluster in range(len(c)):
        global_means.append((1/nd))
    for cluster in range(len(c)):
        mcg+=(clustering_gain(cluster)*ratio_error_sum(cluster))
    return 2

threshold = 1
clustering_config_vectors = {}
vehicle_density = getVehicleDensities()
for k in range(2,len(vehicle_density)):
    cluster_k = kmeans(vehicle_density, k)
    # print(cluster_k)
    mcg_ck = mcg(cluster_k)
    if mcg_ck >= threshold:
        clustering_config_vectors[k] = cluster_k
# print(clustering_config_vectors, 'clustering_config_vectors')
optimal_config_vector = []
optimal_cluster = []
Ag = random_adjacency(153)
min_connected_components = connected_component_count(clustering_config_vectors[2], Ag)

for clusterId in range(3, max(clustering_config_vectors.keys())+1):
    # print(clusterId)
    connected_components_count = connected_component_count(clustering_config_vectors[clusterId], Ag)

    if min_connected_components>connected_components_count:
        min_connected_components = connected_components_count
        optimal_config_vector = clustering_config_vectors[clusterId]
        # print(optimal_config_vector, "each optimal_config_vector")
print(optimal_config_vector, "supernodes")
