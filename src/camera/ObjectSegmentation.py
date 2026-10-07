import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

from .data import landmark_names

class ObjectSegmentation:
    def __init__(self, plot: bool):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,   
            smooth_segmentation=True
        )

        self.plot = plot
        self.initial_pose_3d = None

        if self.plot:
            self.__plot_init()

    def __plot_init(self):
        # Setup a 3D figure in matplotlib
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Create line objects for pose connections 
        self.pose_connections = list(self.mp_holistic.POSE_CONNECTIONS)
        self.lines = []
        for _ in self.pose_connections:
            line, = self.ax.plot([0, 0], [0, 0], [0, 0], c='red')
            self.lines.append(line)

        # Create text labels for each of the 33 landmarks
        self.texts = []
        for i in range(33):
            txt = self.ax.text(0, 0, 0, landmark_names.get(i, str(i)), color='black')
            self.texts.append(txt)

        # Create a scatter for the 33 pose points
        self.scatter = self.ax.scatter([], [], [], c='blue', s=20)
        # 3D axis settings
        self.ax.set_title("3D Pose (Current Frame)")
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.view_init(elev=20, azim=-60) 

    def __update(self, aligned_pose, result):
        """
        Update the 3D skeleton/points using the aligned 3D pose.
        """
        # 2) Lower the visibility threshold (e.g. 0.3)
        threshold = 0.3
        visible_indices = [
            i for i, lm in enumerate(result.pose_world_landmarks.landmark)
            if lm.visibility >= threshold
        ]

        # Filter the aligned pose to visible points
        if len(visible_indices) > 0:
            visible_points = aligned_pose[visible_indices]
        else:
            visible_points = np.empty((0, 3))

        # Update scatter
        if len(visible_points) == 0:
            self.scatter._offsets3d = ([], [], [])
        else:
            xv, yv, zv = visible_points[:, 0], visible_points[:, 1], visible_points[:, 2]
            self.scatter._offsets3d = (xv, yv, zv)

        # Update skeleton lines
        for i, (start_idx, end_idx) in enumerate(self.pose_connections):
            if (start_idx in visible_indices) and (end_idx in visible_indices):
                xs = [aligned_pose[start_idx, 0], aligned_pose[end_idx, 0]]
                ys = [aligned_pose[start_idx, 1], aligned_pose[end_idx, 1]]
                zs = [aligned_pose[start_idx, 2], aligned_pose[end_idx, 2]]
            else:
                xs, ys, zs = [], [], []
            self.lines[i].set_data_3d(xs, ys, zs)

        # Update text labels (visible vs invisible)
        for i in range(33):
            if i in visible_indices:
                x, y, z = aligned_pose[i]
                self.texts[i].set_position_3d((x, y, z))
            else:
                self.texts[i].set_position_3d((999, 999, 999))

        # Optional auto-scale of axes
        if len(visible_points) > 0:
            min_xyz = np.min(visible_points, axis=0)
            max_xyz = np.max(visible_points, axis=0)
            range_xyz = max_xyz - min_xyz
            pad = 0.1 * (range_xyz + 1e-5)
            self.ax.set_xlim(min_xyz[0] - pad[0], max_xyz[0] + pad[0])
            self.ax.set_ylim(min_xyz[1] - pad[1], max_xyz[1] + pad[1])
            self.ax.set_zlim(min_xyz[2] - pad[2], max_xyz[2] + pad[2])

        # Redraw the 3D plot
        plt.draw()
        plt.pause(0.001)

    def find_rigid_transform(self, A, B):
        """
        Given two sets of corresponding 3D points A and B (shape Nx3),
        find the rotation matrix (3x3) and translation vector (3,)
        that aligns A -> B in a least-squares sense.
        """
        assert A.shape == B.shape, "Point sets must have the same shape."
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        AA = A - centroid_A
        BB = B - centroid_B
        H = AA.T @ BB  # 3x3
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Fix reflection if needed
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_B - R @ centroid_A
        return R, t

    def body_segmentation(self, frame):
        """
        Draws the 2D results for pose, face, and hands on the 'frame'.
        Even if pose_world_landmarks is missing, we still get 2D info here.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.holistic.process(rgb)

        # 2D pose (always separate from 3D pose)
        if result.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, result.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS
            )
        # Face landmarks (2D)
        if result.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, result.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION
            )
        # Left hand
        if result.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, result.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
            )
        # Right hand
        if result.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, result.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS
            )

        return frame, result

    def run_segmentation(self, frame):
        """
        Runs MediaPipe Holistic on the entire frame and displays:
         - 2D drawings for pose, face, and hands
         - 3D alignment if pose_world_landmarks found
        """
        # Always update 2D drawings for pose, face, hands (if found)
        frame, result = self.body_segmentation(frame)

        # If 3D pose is available, do alignment + plotting
        if result.pose_world_landmarks:
            current_pose_3d = np.array([
                (lm.x, lm.y, lm.z) for lm in result.pose_world_landmarks.landmark
            ], dtype=np.float32)

            # On first detection, store as reference
            if self.initial_pose_3d is None:
                self.initial_pose_3d = current_pose_3d.copy()
                print("[INFO] Stored initial 3D pose reference.")
            else:
                # Compute R, t from initial -> current
                R, t = self.find_rigid_transform(self.initial_pose_3d, current_pose_3d)
                # Transform the current pose
                aligned_pose = (R @ current_pose_3d.T).T + t

                if self.plot:
                    self.__update(aligned_pose, result)
                return frame, (R, t)


        return frame, (None, None)
    

def main():
    cap = cv2.VideoCapture(0)
    segmenter = ObjectSegmentation(True)

    # Adjust resolution if you like
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) run_segmentation draws face/hands/pose in 2D
        #    and updates the 3D pose if available
        output, _ = segmenter.run_segmentation(frame)

        cv2.imshow("Holistic + 3D Pose (No YOLO)", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
