def compute_degree(input_file: str, output_file: str):
    degree_dict = {}

    with open(input_file, 'r') as file:
        for line in file:
            node1, node2 = line.strip().split('\t')
            if node1 in degree_dict:
                degree_dict[node1] += 1
            else:
                degree_dict[node1] = 1

            if node2 in degree_dict:
                degree_dict[node2] += 1
            else:
                degree_dict[node2] = 1

    # Sort the dictionary by degree in descending order
    degree_dict = dict(sorted(degree_dict.items(), key=lambda item: item[1], reverse=True))

    with open(output_file, 'w') as file:
        for node, degree in degree_dict.items():
            file.write(f'{node} {degree}\n')


for year in range(2013, 2023):
    if __name__ == "__main__":
        input_file = f"D:/datasets/06_dataset_7_4/03_relationship/{year}/{year}_train_qid_relation.txt"
        output_file = f"D:/datasets/06_dataset_7_4/06_degree/new_entities/{year}/{year}_train_qid_degree.txt"
        compute_degree(input_file, output_file)
