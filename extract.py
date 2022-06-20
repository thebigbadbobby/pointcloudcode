import numpy as np
import matplotlib.pyplot as plt
file_name="../Datasets/nuspace/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin"
scan = np.fromfile(file_name, dtype=np.float32)
points = scan.reshape((-1, 5))[:, :4]
# print(points)
def remove_close(points,radius: float) -> None:
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """

        x_filt = np.abs(points[:, 0]) < radius
        y_filt = np.abs(points[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close, :]
def remove_low_intensity(points,thresh: float) -> None:
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """

        x_filt = np.abs(points[:, 3]) < thresh
        low_inten = np.logical_not(x_filt)
        return points[low_inten, :]
points=remove_close(points,2)
points=remove_low_intensity(points,30)
print(len(points))
plt.scatter(points[:, 0], points[:, 1],c=points[:, 2],s=points[:, 3]/10)
plt.show()