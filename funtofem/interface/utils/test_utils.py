__all__ = ["make_test_directories"]
import os


def make_test_directories(comm, base_dir):
    results_folder = os.path.join(base_dir, "results")
    if comm.rank == 0:  # make the results folder if doesn't exist
        if not os.path.exists(results_folder):
            os.mkdir(results_folder)

    output_dir = os.path.join(base_dir, "output")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    return results_folder, output_dir
