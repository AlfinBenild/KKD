import traceback
from multiprocessing import Process, Pipe
import cv2
import face_recognition
import numpy as np


def acquire_user_image():

    vid = cv2.VideoCapture(0)
    while(True):

        ret, frame = vid.read()
        cv2.imshow('Acquiring Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('user'+'.jpg', frame)
            break
  
    vid.release()
    cv2.destroyAllWindows()



def find_user_in_frame(conn, frame, user_encoding):
    face_locations = face_recognition.face_locations(frame, model='cnn')
    face_encodings = face_recognition.face_encodings(frame, face_locations, num_jitters=2)

    found_user = False
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces((user_encoding, ), face_encoding, tolerance=0.9)

        found_user = any(matches)
        if found_user:
            break

    conn.send(found_user)


def load_user_data():
    try:
        user_image_face_encoding = np.load('user.npz')
    except FileNotFoundError:
        user_image = face_recognition.load_image_file('user.jpg')
        try:
            user_image_face_encoding = face_recognition.face_encodings(user_image, num_jitters=10)[0]
            np.save('user.npz', user_image_face_encoding)
        except Exception as e:
            print("Could not detect face, please ensure proper lighting and face orientation")

    return user_image_face_encoding


def run():

    user_encoding = load_user_data()
    
    vid = cv2.VideoCapture(0)

    parent_conn, child_conn = Pipe()
    find_user_process = None

    tries = 1

    while True:
        
        while True:
            ret, frame = vid.read()
            cv2.imshow('Scanning Face', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        if find_user_process is None:
            find_user_process = Process(target=find_user_in_frame, args=(child_conn, rgb_small_frame, user_encoding))
            find_user_process.start()
        elif find_user_process is not None and not find_user_process.is_alive():
            user_found = parent_conn.recv()
            find_user_process = None

            if user_found:
                print('ACCESS GRANTED')
                break
            else:
                print('ACCESS DENIED')
                if tries == 5:
                    print('Maximum tries exceeded, try later')
                    break
                tries += 1




if __name__ == '__main__':
    try:
        c = int(input("Enter Input: 1 or 2: "))
        if c == 1:
            acquire_user_image()
            print("Image acquired successfully")
        elif c == 2:
            run()

    except Exception:
        with open('error.log', 'a') as error_file:
            traceback.print_exc(file=error_file)
            error_file.write('\n')
