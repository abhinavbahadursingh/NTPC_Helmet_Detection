# import cv2

# urls = [
#     "rtsp://admin:Ntpc%40123@192.168.1.250:554/video/live?channel=1&subtype=0",
#     "rtsp://admin:Ntpc%40123@192.168.1.250:554/video/live?channel=1&subtype=1",
#     "rtsp://admin:Ntpc%40123@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0",
#     "rtsp://admin:Ntpc%40123@192.168.1.250:554/cam/realmonitor?channel=1&subtype=1",
#     "rtsp://admin:Ntpc%40123@192.168.1.250:554/video/live?channel=1&subtype=1"
# ]

# for url in urls:
#     print("=" * 60)
#     print(url)
#     cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
#     print("Opened:", cap.isOpened())
#     ret, frame = cap.read()
#     print("Read:", ret)
#     cap.release()



# import cv2

# url = "rtsp://admin:Ntpc%40123@192.168.1.250:554/video/live?channel=1&subtype=1"

# cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

# print("Opened:", cap.isOpened())

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed")
#         break

#     cv2.imshow("Camera", frame)

#     if cv2.waitKey(1) == ord("q"):
#         break

# cap.release()
# # cv2.destroyAllWindows()


# import cv2

# print(cv2.__version__)
# print(cv2.getBuildInformation())

# import cv2

# url = "rtsp://admin:Ntpc%40123@192.168.1.250:1935/video/live?channel=1&subtype=0"

# cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

# print("isOpened :", cap.isOpened())
# print("Backend  :", cap.getBackendName())

# ret, frame = cap.read()

# print("ret =", ret)

# if ret:
#     print(frame.shape)
# else:
#     print("Frame None")


# import cv2

# url = "rtsp://admin:Ntpc%40123@192.168.1.250:554/video/live?channel=1&subtype=0"
# cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

# if not cap.isOpened():
#     print("Cannot open RTSP stream")
# else:
#     ret, frame = cap.read()
#     print("Frame read:", ret)
#     if ret:
#         cv2.imwrite("test_frame.jpg", frame)
#         print("Saved test_frame.jpg")


import socket

HOST = "192.168.1.250"
PORT = 554

request = (
    "OPTIONS rtsp://192.168.1.250:554/video/live?channel=1&subtype=0 RTSP/1.0\r\n"
    "CSeq: 1\r\n"
    "\r\n"
)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(5)
s.connect((HOST, PORT))
s.sendall(request.encode())
response = s.recv(4096)
print(response.decode(errors="replace"))
s.close()