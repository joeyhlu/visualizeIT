import numpy as np
import cv2 as cv
from numpy.typing import NDArray


class CameraCalib:
    def __init__(self, camera_id, board_size, square_size, frame):
        self.w, self.l = board_size
        self.camera_id, self.images = camera_id, []
        self.frame = frame

        # Prepare 3D object points
        self.obj = np.zeros((self.w * self.l, 3), np.float32)
        self.obj[:, :2] = np.mgrid[0:self.w, 0:self.l].T.reshape(-1, 2) * square_size

        # 3D and 2D points
        self.p3d = []
        self.p2d = []

    def __save(self, filepath: str = "camera_matrix.txt", matrix: NDArray[np.float32] = None) -> None:
        """
        Saves poses to a specified file.
        
        Parameters:
            filepath (str): Path to file.
            matrix (NDArray[np.float32]): Camera matrix to save.
        """
        if matrix is not None:
            with open(filepath, 'w') as f:
                np.savetxt(f, matrix, fmt="%6f")
                print(f"Camera matrix saved to {filepath}")

    def add_image(self) -> bool:
        cap = cv.VideoCapture(self.camera_id)
        assert cap.isOpened(), "Unable to open the camera."

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to read from the camera.")
                return False

            # Convert to grayscale for chessboard corner detection
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Find chessboard corners in the live video
            ret, corners = cv.findChessboardCorners(gray, (self.w, self.l), None)

            if ret:
                # Draw detected corners on the frame
                cv.drawChessboardCorners(frame, (self.w, self.l), corners, ret)
                print("asdjasj ")

            # Display the video feed with chessboard corners
            cv.imshow("Frame with Chessboard Corners", frame)

            # Capture image on spacebar press
            if cv.waitKey(1) & 0xFF == ord(" "):
                if ret:
                    self.images.append(gray)
                    print(f"Captured image {len(self.images)} with detected corners.")
                else:
                    print("No corners detected. Image not captured.")

            # Exit on 'ESC'
            elif cv.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv.destroyAllWindows()
        return True

    def find_matrix(self, display: bool = False) -> bool:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for image in self.images:
            ret, corners = cv.findChessboardCorners(image, (self.w, self.l), None)

            if ret:
                self.p3d.append(self.obj)
                self.p2d.append(corners)

                if display:
                    corners2 = cv.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
                    cv.drawChessboardCorners(image, (self.w, self.l), corners2, ret)
                    cv.imshow("Chessboard", image)
                    cv.waitKey(400)
            else:
                print("Failed to find chessboard corners in an image.")
                return False

        cv.destroyAllWindows()

        # Camera calibration
        ret, cameraMatrix, dist, _, _ = cv.calibrateCamera(
            self.p3d, self.p2d, self.frame, None, None
        )

        if ret:
            print("Camera matrix:")
            print(cameraMatrix)
            self.__save(matrix=cameraMatrix)
            return True
        else:
            print("Camera calibration failed.")
            return False


def main():
    # Use integer board size (8, 8) for the chessboard
    c = CameraCalib(0, (8, 8), 1.15, (1440, 1080))
    if c.add_image():
        print("Starting calibration...")
        print(c.find_matrix(True))


if __name__ == "__main__":
    main()
