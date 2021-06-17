import matplotlib.pyplot as plt
from data_loader import read_tif


def display(path):
    disparity = read_tif(path)[:, :, 0]
    plt.figure()
    plt.axis('off')
    plt.title('Predicted disparity')
    plt.imshow(disparity, 'gray')
    plt.show()


if __name__ == '__main__':
    path = '../DSSMNet/prediction/JAX/JAX_168_023_001_LEFT_DSP.tif'
    display(path)
