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

import cv2

url = "rtsp://admin:Ntpc%40123@192.168.1.250:554/video/live?channel=1&subtype=0"

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

print("isOpened :", cap.isOpened())
print("Backend  :", cap.getBackendName())

ret, frame = cap.read()

print("ret =", ret)

if ret:
    print(frame.shape)
else:
    print("Frame None")