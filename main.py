from simple_fat_paths_draw import draw_fat_tree
from simple_fat_paths_draw_precise import draw_fat_tree_with_host_numbers
from simple_fat_paths_model import Model, LinkType
from test_case_generator import run_test_package


# create a model, remove a link and draw the fat-tree
def scenario_1():
    k = 4  # Example parameter for fat-tree
    model = Model(k)
    model.remove_link((LinkType.E_A, (3, 2)))  # Example of removing a link
    draw_fat_tree(model)

# show larger trees
def scenario_2():
    k = 8  # Example parameter for fat-tree
    model = Model(k)
    draw_fat_tree(model)


def scenario_3():
    k = 6  # Example parameter for fat-tree
    model = Model(k)
    draw_fat_tree_with_host_numbers(model)
    paths = model.calculate_possible_paths(10, 20)
    print("\n Path:".join(map(str, paths)))
    distributions = model.calculate_paths_ecmp_probability_distribution(paths)
    print("\n Distribution:".join(map(str, distributions)))


def scenario_4():
    k = 4  # Example parameter for fat-tree
    model = Model(k)
    print("================================ All Paths ================================")
    draw_fat_tree_with_host_numbers(model)
    paths = model.calculate_possible_paths(0, 1)
    print("In TOR:")
    print(paths)
    distributions = model.calculate_paths_ecmp_probability_distribution(paths)
    print(f"Distributions: {distributions}")

    paths = model.calculate_possible_paths(0, 2)
    print("In POD:")
    print("\n Path:".join(map(str, paths)))
    distributions = model.calculate_paths_ecmp_probability_distribution(paths)
    print(f"Distributions: {distributions}")

    paths = model.calculate_possible_paths(0, 15)
    print("Cross POD:")
    print("\n Path:".join(map(str, paths)))
    distributions = model.calculate_paths_ecmp_probability_distribution(paths)
    print(f"Distributions: {distributions}")

    print("================================ Drop links ================================")
    model.remove_link((LinkType.A_C, (0, 1)))
    draw_fat_tree_with_host_numbers(model)
    paths = model.calculate_possible_paths(0, 1)
    print("In TOR:")
    print("\n Path:".join(map(str, paths)))
    distributions = model.calculate_paths_ecmp_probability_distribution(paths)
    print(f"Distributions: {distributions}")

    paths = model.calculate_possible_paths(0, 2)
    print("In POD:")
    print("\n Path:".join(map(str, paths)))
    distributions = model.calculate_paths_ecmp_probability_distribution(paths)
    print(f"Distributions: {distributions}")

    paths = model.calculate_possible_paths(0, 15)
    print("Cross POD:")
    print("\n Path:".join(map(str, paths)))
    distributions = model.calculate_paths_ecmp_probability_distribution(paths)
    print(f"Distributions: {distributions}")

def scenario_5():
    run_test_package()

def main():
    # scenario_1()
    # scenario_2()
    # scenario_3()
    # scenario_4()
    scenario_5()




if __name__ == '__main__':
    main()
