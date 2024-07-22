import cv2
import os

# Check if the Haar Cascade file exists
cascade_path = r'haarcascade_frontalface_default.xml'
if not os.path.isfile(cascade_path):
    print("Error: 'haarcascade_frontalface_default.xml' not found")
else:
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Read the 
    
    img_path=r"E:\files\ml\ml\test.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Image file '{img_path}' not found")
    else:
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (1, 0, 0), 2)

        # Display the image with detected faces
        cv2.imshow('img', img)
        cv2.waitKey(0)

        # Release resources
        cv2.destroyAllWindows()
