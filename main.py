from simple_fat_paths_draw import draw_fat_tree
from simple_fat_paths_draw_precise import draw_fat_tree_with_host_numbers
from simple_fat_paths_model import Model, LinkType
from test_case_generator import run_all_to_all_test_case, run_test_package


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
    k = 4  # Example parameter for fat-tree
    model = Model(k)
    for link in [
        (LinkType.E_A, (3, 2)),
        (LinkType.A_C, (0, 1)),
    ]:
        model.remove_link(link)

    iterations = 5
    run_all_to_all_test_case(model, iterations, True, save_prefix="fat_tree_k4")


def scenario_6():
    balancing = {"balanced": True, "unbalanced": False}
    removal_iterations: int = 3
    links_to_remove_list = [0.0, 0.1, 0.2, 0.4]
    for k in [8, 10]:
        for balance_name, balance in balancing.items():
            print(f"running {balance_name} mode for k = {k}")
            for links_to_remove in links_to_remove_list:
                print("running for k =", k, "balance mode ", balance_name , " removing ", links_to_remove*100, "% links")
                for removal_iteration in range(removal_iterations):
                    print(f"Removal iteration {removal_iteration}")
                    model = Model(k)
                    model.remove_links(links_to_remove / 2, links_to_remove / 2, balance)
                    file_prefix = f"fat_tree_{k}_{balance_name}_removed_{links_to_remove:.2f}_remove{removal_iteration}"
                    draw_fat_tree_with_host_numbers(model, show=False, save_path=f"{file_prefix}.png", number_hosts=True,
                                                    start_index=0, max_labels=200)
                    run_all_to_all_test_case(model, iterations=10, draw_each_iter=False, save_prefix=file_prefix)

def scenario_7():
    run_test_package()

def main():
    # scenario_1()
    # scenario_2()
    # scenario_3()
    # scenario_4()
    # scenario_5()
    # scenario_6()
    scenario_7()



if __name__ == '__main__':
    main()
