import numpy as np
import cv2 as cv
from pathlib import Path
from typing import *

class ValidationOptions:
    def __init__(self, dataset:str, left_cam:int, right_cam:int) -> None:
        self.dataset = dataset
        self.left_cam = left_cam
        self.right_cam = right_cam

def point_cloud_selector(opts: ValidationOptions) -> Tuple[np.ndarray, np.ndarray]:
    """
    Captures points selected by the users as pixel co-ordinates in stereo pair of images. 
    Projects those points from 2D to 3D using DLT. Then renders these points in a Visualizer.

    Args:
        opts (CheckboardProjectionOptions): Configuration options for checkerboard projection and stereo camera setup.

    Returns:
        pcd: np.ndarray: 3D point cloud of selected points in stereo pair of images.
    """
    path = Path(opts.dataset)
    left_cap = FrameReader(path, opts.left_cam)
    right_cap = FrameReader(path, opts.right_cam)
    cv.namedWindow("Left Rectified Image")
    cv.namedWindow("Right Rectified Image")
    selected_frames = list()
    while True:
        
        lret, left_frame = next(left_cap.read())
        rret, right_frame = next(right_cap.read())
        
        if not (lret or rret):
            print("Cannot read video frames")
            break
        # left_frame_rect = cv.remap(left_frame, left_map_x, left_map_y, cv.INTER_LANCZOS4)
        # right_frame_rect = cv.remap(right_frame, right_map_x, right_map_y, cv.INTER_LANCZOS4)
        left_frame_rect, right_frame_rect = _write_to_frames(left_frame, right_frame, "Press space to select frames. Selected Frames: {}".format(len(selected_frames)))
        break_loop = False
        while True:
            cv.imshow("Left Rectified Image", left_frame_rect)
            cv.imshow("Right Rectified Image", right_frame_rect)
            key = cv.waitKey(1) & 0xFF

            if key == ord(' '):
                selected_frames.append((left_frame, right_frame))
                break
            
            if key == ord('q') or len(selected_frames) >= 5:
                
                break_loop = True
                break
        
        if break_loop:
            
            break

    cv.destroyAllWindows()

    idx = 0
    left_point_dropper = PointDropper(img=selected_frames[idx][0])
    right_point_dropper = PointDropper(img=selected_frames[idx][1])
    cv.namedWindow("Left Point Selector")
    cv.namedWindow("Right Point Selector")
    cv.setMouseCallback("Left Point Selector", left_point_dropper.on_mouse)
    cv.setMouseCallback("Right Point Selector", right_point_dropper.on_mouse)

    left_points = list()
    right_points = list()
    while True:
        cv.imshow("Left Point Selector", left_point_dropper.img)
        cv.imshow("Right Point Selector", right_point_dropper.img)
        key = cv.waitKey(1) & 0xFF

        if key == ord(' '):
            # flush selected points
            if left_point_dropper.points and right_point_dropper.points and len(left_point_dropper.points) == len(right_point_dropper.points):
                left_points.append(left_point_dropper.points)
                right_points.append(right_point_dropper.points)
            idx += 1
            if idx >= len(selected_frames):
                cv.destroyAllWindows()
                break
            left_point_dropper = PointDropper(img=selected_frames[idx][0])
            right_point_dropper = PointDropper(img=selected_frames[idx][1])
            cv.setMouseCallback("Left Point Selector", left_point_dropper.on_mouse)
            cv.setMouseCallback("Right Point Selector", right_point_dropper.on_mouse)
    
    # get 3D projection
    left_points = np.array(left_points)
    right_points = np.array(right_points)
    print(left_points.shape)
    return left_points, right_points


class PointDropper:
    def __init__(self, img) -> None:
        self.img = img
        self.points = list()
    
    def on_mouse(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            print('x = %d, y = %d'%(x, y))
            self.points.append((y, x))
            self.img = cv.circle(self.img,(x,y),10,(255,0,0),-1)


class FrameReader:
    def __init__(self, path:Path, cam_id:int):
        self.path = path
        self.cam_id = cam_id
        self.count = 0
    
    def read(self):
        image_path = self.path.joinpath(*["camera{}_{}.png".format(self.cam_id, self.count)])
        self.count += 1
        if image_path.exists():
            yield True, cv.imread(str(image_path))
        else:
            yield False, None


def _write_to_frames(left:np.ndarray, right:np.ndarray, text:str) -> Tuple[np.ndarray, np.ndarray]:
    set_text = lambda frame, text: cv.putText(frame, text ,(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3, 1)
    return set_text(left, text), set_text(right, text)
