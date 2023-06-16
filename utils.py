import os
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def display_image(image, cmap='gray', title=None):
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.show()


def display_images(*images, titles=None, cmaps=None):
    '''
    usage:
    display(image1, mask1, image2, mask2, titles=['Image 1', 'Mask 1', 'Image 2', 'Mask 2'])
    '''

    rgb_cmap = ListedColormap([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])

    if cmaps is None:
        cmaps = ['gray'] * len(images)  # Default to 'gray' colormap for all image sets
    else:
        cmaps = [rgb_cmap if cmap == 'rgb' else cmap for cmap in cmaps]
        if len(cmaps) < len(images):
            cmaps += ['gray'] * (len(images) - len(cmaps))  # Fill remaining elements with 'gray' colormap

    num_images = len(images)

    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 5, 5))

    for i, image_set in enumerate(images):
        image = image_set
        if isinstance(image, str):
            # Convert string-based image data to numeric format
            image = np.asarray(image, dtype=np.float32)

        axs[i].imshow(image, cmap=cmaps[i])
        axs[i].axis('off')
        if titles is not None:
            axs[i].set_title(titles[i])

    plt.tight_layout()
    plt.show()


def display_image_sets(*image_sets, titles=None, cmaps=None):
    ''''
    usage:
    images_set1 = [image1_set1, image2_set1, image3_set1]
    images_set2 = [image1_set2, image2_set2, image3_set2]
    titles = ['Set 1', 'Set 2']
    cmaps = ['Reds', 'Blues']

    display_images_side_by_side(images_set1, images_set2, titles=titles, cmaps=cmaps)

    '''
    rgb_cmap = ListedColormap([[1, 0, 0] , [0, 1, 0] , [0, 0, 1] , [1, 1, 1] ])

    if cmaps is None:
        cmaps = ['gray'] * len(image_sets)  # Default to 'gray' colormap for all image sets
    else:
        cmaps = [rgb_cmap if cmap == 'rgb' else cmap for cmap in cmaps]
        if len(cmaps) < len(image_sets):
            cmaps += ['gray'] * (len(image_sets) - len(cmaps))  # Fill remaining elements with 'gray' colormap

    num_sets = len(image_sets)
    num_images = len(image_sets[0])

    fig, axs = plt.subplots(num_images, num_sets, figsize=(num_sets * 5, num_images * 5))

    for i, image_set in enumerate(image_sets):
        for j in range(num_images):
            image = image_set[j]
            if type(image) == object:
                # Convert string-based image data to numeric format
                image = np.asarray(image, dtype=np.float32)

            axs[j, i].imshow(image, cmap=cmaps[i])
            axs[j, i].axis('off')
            if titles is not None:
                axs[j, i].set_title(titles[i] + ' ' + str(j+1) if num_sets else titles[i])  # Title for each column
                axs[j, i].set_ylabel(f'Row {j + 1}', rotation=0, labelpad=40)

    plt.tight_layout()
    plt.show()




def write_model_summary(model, file_path='/content/model_summary.txt'):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'w') as f:
        f.truncate(0)
        string_buf = StringIO()
        model.summary(print_fn=lambda x: string_buf.write(x + '\n'))
        model_summary_str = string_buf.getvalue()
        f.write(model_summary_str)


def calculate_accuracies(true_masks, pred_masks):
    accuracies = []
    for i in range(len(true_masks)):
        true_mask = true_masks[i]
        pred_mask = pred_masks[i]
        accuracy = np.mean(true_mask == pred_mask)
        accuracies.append(accuracy)
    overall_accuracy = np.mean(accuracies)

    return accuracies, overall_accuracy

def plot_accuracies(accuracies, overall_accuracy):
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(accuracies) + 1), accuracies, color='steelblue')
    plt.axhline(overall_accuracy, color='red', linestyle='--', label='Overall Accuracy')
    plt.xlabel('Image')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for Each Image')
    plt.legend()
    plt.xticks(range(1, len(accuracies) + 1))
    plt.ylim(0, 1)
    plt.show()


