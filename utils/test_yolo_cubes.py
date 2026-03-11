import cv2
import pyzed.sl as sl
from ultralytics import YOLO

# Initialize ZED camera
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080  # or HD720
init_params.camera_fps = 30

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera")
    exit()

runtime_params = sl.RuntimeParameters()
mat = sl.Mat()

# Load YOLO model
model = YOLO("/home/pdz/franka_ros2_ws/src/superquadric_grasp_system/cubes.pt")  # use yolov8n.pt (nano) for faster FPS

try:
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(mat, sl.VIEW.LEFT)
            frame = mat.get_data()  # convert to numpy array (BGR or BGRA)

            # Ensure 3 channels for the model: drop alpha if present and convert to RGB
            if frame.ndim == 3 and frame.shape[2] == 4:
                # ZED often returns BGRA; convert to BGR first (drops alpha)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                frame_bgr = frame.copy()

            # YOLO expects RGB input; convert BGR->RGB for inference
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Run YOLO inference on RGB image
            results = model(frame_rgb)[0]
            print(results.boxes)
            # Draw bounding boxes on the BGR frame shown to the user
            for box in results.boxes:
                # convert tensor (possibly on CUDA) to CPU numpy ints
                xy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])
                conf = float(box.conf[0].cpu().item())
                cls = int(box.cls[0].cpu().item())
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_bgr, label, (x1, max(15, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 
            # Show the frame (BGR)
 
            # Show the frame
            cv2.imshow("ZED YOLO", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    zed.close()
    cv2.destroyAllWindows()
