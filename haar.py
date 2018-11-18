import cv2
import urllib.request
import os
import numpy as np

def learning():
    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152'
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
    pic_num = 1

    if not os.path.exists('neg'):
        os.makedirs('neg')

    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, "neg/" + str(pic_num) + ".jpg")
            img = cv2.imread("neg/" + str(pic_num) + ".jpg", cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite("neg/" + str(pic_num) + ".jpg", resized_image)
            pic_num += 1

        except Exception as e:
            print(str(e))

def delete_uglies():
    match = False
    for file_type in ['neg']:
        for img in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_image_path = str(file_type) + '/' + str(img)
                    ugly = cv2.imread('uglies/' + str(ugly))
                    question = cv2.imread(current_image_path)
                    if ugly.shape == question.shape and not (np.bitwise_xor(ugly, question).any()):
                        print('That is one ugly pic! Deleting!')
                        print(current_image_path)
                        os.remove(current_image_path)
                except Exception as e:
                    print(str(e))


def cascade():

    ten_cascade = cv2.CascadeClassifier('cascade.xml')
    twenty_cascade = cv2.CascadeClassifier('cascade20.xml')
    fifty_cascade = cv2.CascadeClassifier('cascade50.xml')
    # eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')

    cap = cv2.VideoCapture(0)

    while True:
        # img = cv2.imread('pln10.jpg')
        ret, img = cap.read()

        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #print(gray)
        ten = ten_cascade.detectMultiScale(gray, 1.3, 5)
        twenty = twenty_cascade.detectMultiScale(gray, 1.3, 5)
        fifty = fifty_cascade.detectMultiScale(gray, 1.3, 5)
        for(x,y,w,h) in ten:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        for(x,y,w,h) in twenty:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)
        for(x,y,w,h) in fifty:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
            # roi_gray = gray[y:y+h, x:x+w]
            # roi_color = img[y:y+h, x:x+w]

            # eyes = eye_cascade.detectMultiScale(roi_gray)
            # for(ex,ey,ew,eh) in eyes:
            #     cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def create_pos_and_neg():
    for file_type in ['neg10']:

        for img in os.listdir(file_type):
            if file_type == 'neg10':
                line = file_type+'/'+img+'\n'
                with open('bg10.txt', 'a') as f:
                    f.write(line)

            elif file_type == 'pos':
                line = file_type+'/'+img+ '1 0 0 50\n'
                with open('bg10.txt', 'a') as f:
                    f.write(line)
            # print(file_type)

if __name__=='__main__':
    # learning()
    # delete_uglies()
    create_pos_and_neg()
    # cascade()