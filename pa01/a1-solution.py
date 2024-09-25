################################################################
# Machine Learning
# Assignment 1 Starting Code
#
# Author: R. Zanibbi
# Author: E. Lima
################################################################


import argparse
import numpy as np
# import statistics as stats
import matplotlib.pyplot as plt

################################################################
# Metrics and visualization
################################################################


def conf_matrix(class_matrix, data_matrix, title=None, print_results=True):

    # Initialize confusion matrix with 0's
    confusions = np.zeros((2, 2), dtype=int)

    # Generate output/target pairs, convert to list of integer pairs
    # * '-1' indexing indicates the final column
    # * list(zip( ... )) combines the output and target class label columns into a list of pairs
    out_target_pairs = [
        (int(out), int(target))
        for (out, target) in list(zip(class_matrix[:, -1], data_matrix[:, -1]))
    ]

    # Use output/target pairs to compile confusion matrix counts
    for (out, target) in out_target_pairs:
        confusions[out][target] += 1

    # Compute recognition rate
    inputs_correct = confusions[0][0] + confusions[1][1]
    inputs_total = np.sum(confusions)
    recognition_rate = inputs_correct / inputs_total * 100

    if print_results:
        if title:
            print("\n>>>  " + title)
        print(
            "\n    Recognition rate (correct / inputs):\n    ", recognition_rate, "%\n"
        )
        print("    Confusion Matrix:")
        print("              0: Blue-True  1: Org-True")
        print("---------------------------------------")
        print("0: Blue-Pred |{0:12d} {1:12d}".format(confusions[0][0], confusions[0][1]))
        print("1: Org-Pred  |{0:12d} {1:12d}".format(confusions[1][0], confusions[1][1]))

    return (recognition_rate, confusions)


def draw_results(data_matrix, class_fn, title, file_name):

    # Fix axes ranges so that X and Y directions are identical (avoids 'stretching' in one direction or the other)
    # Use numpy amin function on first two columns of the training data matrix to identify range
    pad = 0.25
    min_tick = np.amin(data_matrix[:, 0:2]) - pad
    max_tick = np.amax(data_matrix[:, 0:2]) + pad
    plt.xlim(min_tick, max_tick)
    plt.ylim(min_tick, max_tick)

    ##################################
    # Grid dots to show class regions
    ##################################

    axis_tick_count = 75
    x = np.linspace(min_tick, max_tick, axis_tick_count, endpoint=True)
    y = np.linspace(min_tick, max_tick, axis_tick_count, endpoint=True)
    (xx, yy) = np.meshgrid(x, y)
    grid_points = np.concatenate(
        (xx.reshape(xx.size, 1), yy.reshape(yy.size, 1)), axis=1
    )

    class_out = class_fn(grid_points)

    # Separate rows for blue (0) and orange (1) outputs, plot separately with color
    blue_points = grid_points[np.where(class_out[:, 1] < 1.0)]
    orange_points = grid_points[np.where(class_out[:, 1] > 0.0)]

    plt.scatter(
        blue_points[:, 0],
        blue_points[:, 1],
        marker=".",
        s=1,
        facecolors="blue",
        edgecolors="blue",
        alpha=0.4,
    )
    plt.scatter(
        orange_points[:, 0],
        orange_points[:, 1],
        marker=".",
        s=1,
        facecolors="orange",
        edgecolors="orange",
        alpha=0.4,
    )

    ##################################
    # Decision boundary (black line)
    ##################################
    
    offset = (grid_points[:, 0].max() - grid_points[:, 0].min()) / (axis_tick_count * 2)
    plots = [[[],[]]]
    prev_boundaries = []
    for x in range(0, axis_tick_count):
        boundaries = []
        prev = class_out[x][1]
        for y in range(0, axis_tick_count**2, axis_tick_count):
            if class_out[x+y][1] != prev:
                boundaries.append([x+y, y/axis_tick_count, -1]) #index corresponding to grid points array, vertical placement, associated plot line
                prev = (prev + 1) % 2 #toggle prev between 0 and 1
        unlinked = [] #an array of boundaries that don't connect to the previous column
        linked_prev = [] #an array of previous boundaries that connect to the current column
        unlinked_pair = False #flag to signify that the next boundary is the second part of an unlinked pair
        for l in range(len(prev_boundaries)):
            unlink = True #flag signaling that the boundary is to be unlinked
            polarity = class_out[prev_boundaries[l][0]][1]
            if l + 1 < len(prev_boundaries):
              # print('checking')
              for p in range(int(abs(prev_boundaries[l][1] - prev_boundaries[l+1][1]))):
                  if class_out[prev_boundaries[l][0] + 1 + p*axis_tick_count][1] == polarity:
                      unlink = False
                      continue
            else:
                # print('last boundary')
                if abs(len(prev_boundaries) - len(boundaries)) % 2 == 0:
                    unlink = False
            if unlinked_pair:
                
                unlinked_pair = False
                # print('unlinked pair')
            elif unlink:
                unlinked_pair = True
                # print('check returns unlink')
            else:
                linked_prev.append(prev_boundaries[l])
        unlinked_pair = False
        for i in range(len(boundaries)):
            # print(str(boundaries[i][0] - boundaries[i][1]*axis_tick_count) + '   y:' + str(boundaries[i][1]))
            unlink = True #flag signaling that the boundary is to be unlinked
            polarity = class_out[boundaries[i][0]][1]
            if i + 1 < len(boundaries):
              # print('checking')
              for p in range(int(abs(boundaries[i][1] - boundaries[i+1][1]))):
                  if class_out[boundaries[i][0] - 1 + p*axis_tick_count][1] == polarity:
                      unlink = False
                      continue
            elif len(linked_prev) > 0:
                # print('last boundary')
                unlink = False
            if unlinked_pair:
                unlinked.append(boundaries[i])
                unlinked_pair = False
                # print('unlinked pair')
            elif unlink:
                unlinked.append(boundaries[i])
                unlinked_pair = True
                # print('check returns unlink')
            else:
                # print('linked in')
                plot_line = linked_prev[i-len(unlinked)][2]
                boundary = boundaries[i]
                boundary[2] = plot_line
                plots[plot_line][0].append(grid_points[boundary[0]][0])
                plots[plot_line][1].append(grid_points[boundary[0]][1] - offset)
            # print(boundaries)
        for u in range(len(unlinked)):
            unlinked[u][2] = len(plots)
            plots.append([[grid_points[unlinked[u][0]][0]],[grid_points[unlinked[u][0]][1]]])
            if u % 2 == 1:
              plots[-1][0].insert(0, grid_points[unlinked[u-1][0]][0])
              plots[-1][1].insert(0, grid_points[unlinked[u-1][0]][1])
        link = -1
        for p in range(len(prev_boundaries)):
            if prev_boundaries[p] not in linked_prev:
                if link == -1:
                    link = p
                else:
                    plots[prev_boundaries[link][2]][0].append(grid_points[prev_boundaries[p][0]][0])
                    plots[prev_boundaries[link][2]][1].append(grid_points[prev_boundaries[p][0]][1])
                    link = -1
        prev_boundaries = boundaries
    
    for plot in plots:
        plt.plot(plot[0], plot[1], 'k')


    ##################################
    # Show training samples
    ##################################

    # Separate rows for blue (0) and orange (1) target inputs, plot separately with color
    blue_targets = data_matrix[np.where(data_matrix[:, 2] < 1.0)]
    orange_targets = data_matrix[np.where(data_matrix[:, 2] > 0.0)]

    plt.scatter(
        blue_targets[:, 0],
        blue_targets[:, 1],
        marker="o",
        facecolors="none",
        edgecolors="blue",
    )
    plt.scatter(
        orange_targets[:, 0],
        orange_targets[:, 1],
        marker="o",
        facecolors="none",
        edgecolors="darkorange",
    )
    ##################################
    # Add title and write file
    ##################################

    # Set title and save plot if file name is passed (extension determines type)
    plt.title(title)
    plt.savefig(file_name)
    # print("\nWrote image file: " + file_name)
    plt.close()


################################################################
# Interactive Testing
################################################################
def test_points(data_matrix, beta_hat):

    # print("\n>> Interactive testing for (x_1,x_2) inputs")
    stop = False
    while True:
        x = input("\nEnter x_1 ('stop' to end): ")

        if x == "stop":
            break
        else:
            x = float(x)

        y = float(input("Enter x_2: "))
        k = int(input("Enter k: "))

        lc = linear_classifier(beta_hat)
        knn = knn_classifier(k, data_matrix)

        # print("   least squares: " + str(lc(np.array([x, y]).reshape(1, 2))))
        # print("             knn: " + str(knn(np.array([x, y]).reshape(1, 2))))


################################################################
# Classifiers
################################################################

def linear_classifier(weight_vector):
    # Constructs a linear classifier
    def classifier(input_matrix):
        # Take the dot product of each sample ( data_matrix row ) with the weight_vector (col vector)
        # -- as defined in Hastie equation (2.2)
        row_count = input_matrix.shape[0]
        col_count = input_matrix.shape[1]
        weight_col_count = weight_vector.shape[0]

        if col_count < weight_col_count:
          input_matrix = np.hstack((input_matrix, np.ones((row_count, 1))))
        scores = np.dot(input_matrix, weight_vector)

        scores_classes = np.zeros((row_count,2))
        for i in range(row_count):
          scores_classes[i][0] = np.float64(scores[i])
          if scores[i] < 0.5:
            scores_classes[i][1] = np.float64(0.0)
          else:
            scores_classes[i][1] = np.float64(1.0)

        return scores_classes

    return classifier


def knn_classifier(k, data_matrix):
    # Constructs a knn classifier for the passed value of k and training data matrix
    def classifier(input_matrix):
        (input_count, input_cols) = input_matrix.shape
        (data_count, data_cols) = data_matrix.shape

        distances = np.zeros((data_count,2))
        scores_classes = np.zeros((input_count,2))

        if input_cols == data_cols:
            input_cols -= 1

        for i in range(input_count):
            for j in range(data_count):
                distance = 0.0
                for c in range(input_cols):
                    distance += (input_matrix[i][c] - data_matrix[j][c]) ** 2
                distances[j][0] = np.sqrt(distance)
                distances[j][1] = j
            sorted_distances = sorted(distances, key=lambda x : x[0])
            score = 0.0
            for n in range(k):
                score += data_matrix[int(sorted_distances[n][1])][-1]
            score /= k
            scores_classes[i][0] = np.float64(score)
            if score < 0.5:
              scores_classes[i][1] = np.float64(0.0)
            else:
              scores_classes[i][1] = np.float64(1.0)

        return scores_classes

    return classifier

def find_weight(data_matrix):
    rows = data_matrix.shape[0]
    design_matrix = data_matrix[:, :-1]
    X = np.hstack((design_matrix, np.ones((rows, 1))))
    Y = data_matrix[:, -1]
    B = np.linalg.inv(X.T @ X) @ X.T @ Y
    return B


################################################################
# Main function
################################################################


def main():
    # Process arguments using 'argparse'
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="numpy data matrix file containing samples")
    args = parser.parse_args()

    # Load data
    data_matrix = np.load(args.data_file)
    (confm, rr) = conf_matrix(
        data_matrix, data_matrix, "Data vs. Itself Sanity Check", print_results=True
    )

    # Construct linear classifier
    weight_vector = find_weight(data_matrix)
    lc = linear_classifier(weight_vector)
    lsc_out = lc(data_matrix)

    # Compute results on training set
    conf_matrix(lsc_out, data_matrix, "Least Squares", print_results=True)
    draw_results(data_matrix, lc, "Least Squares Linear Classifier", "ls.pdf")

    # Nearest neighbor
    for k in [1, 15]:
        knn = knn_classifier(k, data_matrix)
        knn_out = knn(data_matrix)
        conf_matrix(knn_out, data_matrix, "knn: k=" + str(k), print_results=True)
        draw_results(
            data_matrix,
            knn,
            "k-NN Classifier (k=" + str(k) + ")",
            "knn-" + str(k) + ".pdf",
        )

    # Interactive testing
    test_points(data_matrix, np.array([1,1,1]))


if __name__ == "__main__":
    main()
