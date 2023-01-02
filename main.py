import operator
from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np


@dataclass
class NavyBattle:
    image_match_template: np.ndarray = cv2.imread("images/photo_match.jpg")
    # image_match_template: np.ndarray = cv2.imread("photo_match_preenchida.jpg")
    # image_match_template: np.ndarray = cv2.imread("photo_match_teste.jpg")

    image_teste: np.ndarray = cv2.imread("images/photo_match_teste.jpg")

    matrix_navio = np.random.randint(3, size=(5, 5), dtype='int8')
    matrix_jogo = np.ones([10, 10], dtype='int8')

    random_matrix = np.random.randint(0, 2, (10, 10), dtype='int8') * 255

    matrix_uso_teste = matrix_jogo

    matrix_jogo_teste = np.array(
        [[0, 255, 255, 255, 255, 255, 255, 255, 255, 0], [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255], [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255], [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [255, 255, 255, 255, 255, 255, 255, 255, 255, 255], [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
         [0, 255, 255, 255, 255, 255, 255, 255, 255, 0], [255, 255, 255, 255, 255, 255, 255, 255, 255, 255]])

    MAX_FEATURES = 1000
    GOOD_MATCH_PERCENT = 0.15

    def match_template(self):
        ...

    def view(self, name: str, image: np.ndarray):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, image)

    def align_images(self, im1, im2):
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(self.MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        matches = sorted(matches, key=lambda x: x.distance)

        numGoodMatches = int(len(matches) * self.GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
        cv2.imwrite("matches.jpg", imMatches)

        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        height, width, channels = im2.shape
        im1Reg = cv2.warpPerspective(im1, h, (width, height))

        return im1Reg, h

    def find_puzzle(self, image, debug=False):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 3)
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh = cv2.bitwise_not(thresh)

    def pre_process_image(self, img, skip_dilate=False):

        im1Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        proc = cv2.GaussianBlur(im1Gray.copy(), (9, 9), 0)

        proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        proc = cv2.bitwise_not(proc, proc)

        if not skip_dilate:
            kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
            proc = cv2.dilate(proc, kernel)

        return proc

    def find_corners_of_largest_polygon(self, img):
        """Finds the 4 extreme corners of the largest contour in the image."""
        opencv_version = cv2.__version__.split('.')[0]
        if opencv_version == '3':
            _, contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
        else:
            contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending

        if len(contours) == 0:
            print("Não detectamos contornos")
            exit()

        polygon = contours[0]  # Largest image

        # Use of `operator.itemgetter` with `max` and `min` allows us to get the index of the point
        # Each point is an array of 1 coordinate, hence the [0] getter, then [0] or [1] used to get x and y respectively.

        # Bottom-right point has the largest (x + y) value
        # Top-left has point smallest (x + y) value
        # Bottom-left point has smallest (x - y) value
        # Top-right point has largest (x - y) value
        bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

        return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

    def distance_between(self, p1, p2):
        """Returns the scalar distance between two points"""
        a = p2[0] - p1[0]
        b = p2[1] - p1[1]
        return np.sqrt((a ** 2) + (b ** 2))

    def crop_and_warp(self, img, crop_rect):

        top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

        src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

        side = max([
            self.distance_between(bottom_right, top_right),
            self.distance_between(top_left, bottom_left),
            self.distance_between(bottom_right, bottom_left),
            self.distance_between(top_left, top_right)
        ])

        dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

        m = cv2.getPerspectiveTransform(src, dst)

        return cv2.warpPerspective(img, m, (int(side), int(side)))

    def get_omr_image(self, image):

        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

        w, h = image.shape

        temp = np.array(image)

        diff_y = np.diff(temp, axis=0)
        diff_x = np.diff(temp, axis=1)

        temp = np.zeros_like(temp)
        temp[:h - 1, :] |= diff_y
        temp[:, :w - 1] |= diff_x

        diff_y = np.diff(np.nonzero(np.diff(np.sum(temp, axis=0))))
        diff_x = np.diff(np.nonzero(np.diff(np.sum(temp, axis=1))))

        ht = np.median(diff_y[diff_y > 1]) + 38
        wt = np.median(diff_x[diff_x > 1]) + 38

        dim = (int(w / wt), int(h / ht))

        img_resize = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        thresh = 127
        im_bw = cv2.threshold(img_resize, thresh, 255, cv2.THRESH_BINARY)[1]

        array = (np.array(im_bw))

        # A_1 = np.where(np.abs(_) < 127, 0, _)

        return array

    def run(self, video: Union[str, int]):

        captura = cv2.VideoCapture(video)

        while captura.isOpened():

            is_video, frame = captura.read()

            if is_video:

                teste = self.pre_process_image(img=frame, skip_dilate=True)

                # imReg, h = self.align_images(self.image_match_template, teste)

                # self.view('Pre Process imagem', teste)
                #
                te = self.find_corners_of_largest_polygon(teste)

                #
                oo = self.crop_and_warp(frame, te)

                testeooo = self.pre_process_image(img=oo, skip_dilate=False)

                tewww = self.find_corners_of_largest_polygon(testeooo)

                asasasasas = self.crop_and_warp(oo, tewww)

                self.view(name="Teste", image=asasasasas)
                #
                tete = self.get_omr_image(oo)

                print(1)

                # print(tete)

                #
                # print("\n\n\n\n")
                #
                # print("\n\n\n\n")

                # print(self.matrix_jogo)
                # print(self.matrix_jogo.shape)

                # print("\n\n\n\n")
                #
                # print(self.random_matrix)
                # print(self.random_matrix.shape)

                # print(np.random.rand(10, 10).astype(int))

                if self.matrix_jogo_teste.all() == tete.all():
                    print("Deu certo")
                #
                # cv2.imwrite("resultado_extract.jpg", oo)
                #
                # self.view(name="Teste", image=oo)

                key = cv2.waitKey(20)

                if key == ord('q'):
                    break
            else:
                break

        captura.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    navy = NavyBattle()
    navy.run(video="teste_video_omr_sem_animacao.mp4")
