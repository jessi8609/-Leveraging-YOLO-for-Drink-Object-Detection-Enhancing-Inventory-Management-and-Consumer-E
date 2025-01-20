from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# YOLO modelini yuklash
model = YOLO("yolo11n.pt")

# Kamera ochish
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Model orqali bashorat
    results = model.predict(frame)
    result_frame = results[0].plot()

    # Matplotlib orqali tasvirni ko'rsatish
    plt.imshow(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")  # O'qlarni olib tashlash
    plt.show(block=False)
    plt.pause(0.001)  # Sekundlar ichida yangilanish

    # 'q' bosilganda chiqish
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
plt.close()
