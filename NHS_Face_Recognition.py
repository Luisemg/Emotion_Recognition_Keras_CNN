face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('NHS_vgg.h5')

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    labels = []
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_img,1.32,5)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = classifier.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('Happy', 'Neutral', 'Surprise')
        predicted_emotion = emotions[max_index]

        cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_img = cv2.resize(frame, (1000, 700))
    cv2.imshow('Emotion Recognition',frame)

    if(cv2.waitKey(1) & 0xFF == ord('s')):
        break

capture.release()
cv2.destroyAllWindows()
