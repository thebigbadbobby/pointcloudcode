# import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import json
file_name="../../../../Data/Datasets/nuspace/sweeps/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603597909.pcd.bin"
file_ply="../../../..//Data/Datasets/ARKitScenes/raw/Training/41048190/41048190_3dod_mesh.ply"
file_json="../../../..//Data/Datasets/ARKitScenes/raw/Training/41048190/41048190_3dod_annotation.json"
def compute_box_3d(scale, transform, rotation):
    scales = [i / 2 for i in scale]
    l, h, w = scales
    center = np.reshape(transform, (-1, 3))
    center = center.reshape(3)
    x_corners = [l, l, -l, -l, l, l, -l, -l]
    y_corners = [h, -h, -h, h, h, -h, -h, h]
    z_corners = [w, w, w, w, -w, -w, -w, -w]
    corners_3d = np.dot(np.transpose(rotation),
                        np.vstack([x_corners, y_corners, z_corners]))

    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    bbox3d_raw = np.transpose(corners_3d)
    return bbox3d_raw
def bboxes(annotation):
        bbox_list = []
        for label_info in annotation["data"]:
                label=label_info["label"]
                rotation = np.array(label_info["segments"]["obbAligned"]["normalizedAxes"]).reshape(3, 3)
                transform = np.array(label_info["segments"]["obbAligned"]["centroid"]).reshape(-1, 3)
                scale = np.array(label_info["segments"]["obbAligned"]["axesLengths"]).reshape(-1, 3)
                box3d = compute_box_3d(scale.reshape(3).tolist(), transform, rotation)
                bbox_list.append((FURNITURE_DICT[label],box3d))
        # bbox_list = np.asarray(bbox_list)
        return bbox_list
def load_json(js_path):
    with open(js_path, "r") as f:
        json_data = json.load(f)
    return json_data
def extract_from_pci_bin(file_name):
        scan = np.fromfile(file_name, dtype=np.float32)
        return scan.reshape((-1, 5))[:, :4]
def extract_from_ply_bin(filename):
        scan = PlyData.read(filename)
        x = np.array([list(tup) for tup in scan.elements[0].data])
        return x
        # print(scan)
        # print(scan.reshape((int(len(scan)/4),4)))
        # return scan.reshape((int(len(scan)/4),4))
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
COLOR_LIST = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (0, 1, 1),
    (1, 0, 1),
    (0.5, 0, 0),
    (0.6, 1, 0.3),
    (0, 0.5, 0),
    (0.5, 0, 0.5),
    (0.2, 0.3, 0.2),
    (0.8, 0.7, 0.8),
    (0.1, 0.8, 0.9),
    (0.4, 0.1, 0.7),
    (0.5, 0.5, 0),
    (0, 0.5, 0.5),
    (0.5, 0.5, 0.5),
    (0, 0, 0.5),
]
FURNITURE_DICT={
        'cabinet':0,
        'refrigerator':1,
        'shelf':2,
        'stove':3,
        'bed':4,
        'sink':5,
        'washer':6,
        'toilet':7,
        'bathtub':8,
        'oven':9,
        'dishwasher':10,
        'fireplace':11,
        'stool':12,
        'chair':13,
        'table':14,
        'tv/monitor':15,
        'sofa':16
}

def get_lines(box, color=np.array([1.0, 0.0, 0.0])):
    """
    Args:
        box: np.array (8, 3)
            8 corners
        color: line color
    Returns:
        o3d.Linset()
    """
    points = box
    lines = [
        [0, 1],
        [0, 3],
        [1, 2],
        [2, 3],
        [4, 5],
        [4, 7],
        [5, 6],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def visualize_o3d(
    pc,
    boxes,
    pc_color=None,
    width=384,
    height=288,
):
    """
    Visualize result with open3d
    Args:
        pc: np.array of shape (n, 3)
            point cloud
        boxes: a list of m boxes, each item as a tuple:
            (cls, np.array (8, 3), conf) if predicted
            (cls, np.array (8, 3))
            or just np.array (n, 8, 3)
        pc_color: np.array (n, 3) or None
        box_color: np.array (m, 3) or None
        visualize: bool (directly visualize or return an image)
        width: int
            used only when visualize=False
        height: int
            used only when visualize=False
    Returns:
    """
    assert pc.shape[1] == 3
    ratio = max(1, pc.shape[0] // 4000)
    pc_sample = pc[::ratio, :]

    n = pc_sample.shape[0]
    m = len(boxes)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_sample)
    if pc_color is None:
        pc_color = np.zeros((n, 3))
    pcd.colors = o3d.utility.Vector3dVector(pc_color)

    linesets = []
    for i, item in enumerate(boxes):
        print(item)
        if isinstance(item, tuple):
        #     print("a")
            cls_ = item[0]
            assert isinstance(cls_, int)
            corners = item[1]
        else:
            cls_ = None
            corners = item
        assert corners.shape[0] == 8
        assert corners.shape[1] == 3

        if isinstance(cls_, int) and cls_ < len(COLOR_LIST):
            tmp_color = COLOR_LIST[cls_]
        else:
            tmp_color = (0, 0, 0)
        linesets.append(get_lines(corners, color=tmp_color))

    o3d.visualization.draw_geometries([pcd] + linesets)
    return None
annotation=load_json(file_json)
lists=bboxes(annotation)
print(lists)
points=extract_from_ply_bin(file_ply)
print("points done")
# print(points[0:5])
visualize_o3d(points[:,:3],lists,pc_color=points[:,3:6]/points[:,6][0])
# print(points)
# # points=remove_close(points,2)
# # points=remove_low_intensity(points,30)
# print(len(points))
# plt.scatter(points[:, 0], points[:, 1],c=points[:, 2],s=.1)#,s=points[:, 3]/10)
# plt.show()