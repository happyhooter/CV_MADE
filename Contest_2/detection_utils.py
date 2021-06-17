import numpy as np
import cv2


class PlateImageExtractor():
    def __init__(self, max_iter=100, border=0.0):
        self.max_iter = max_iter
        self.border = border

    def __call__(self, image, mask, bbox=None):
        polygon = self.findQuadrangle(mask)
        if polygon is None:
            polygon = np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]],
                                [bbox[2], bbox[1]], [bbox[2], bbox[3]]])
        return self.four_point_transform(image, polygon)

    def findQuadrangle(self, mask):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        p = 1
        corners = 0
        num_iter = 0
        while num_iter < self.max_iter and corners != 4:
            eps = p * cv2.arcLength(contours[0], closed=True)
            polygon = cv2.approxPolyDP(contours[0], eps, closed=True)
            corners = polygon.shape[0]
            if corners < 4:
                p = p / 2
            elif corners > 4:
                p = 1.5 * p
            num_iter += 1
        if corners != 4:
            print(f"Couldn't converge")
            return None
        return polygon.reshape(4, 2)

    # from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/  # noqa E501
    def rotate(self, A, B, C):
        return (B[0]-A[0])*(C[1]-B[1])-(B[1]-A[1])*(C[0]-B[0])

    def order_points(self, pts):
        # используем первые шаги алгоритма Грэхэма
        # https://habr.com/ru/post/144921/
        n = len(pts)  # число точек
        P = list(range(n))  # список номеров точек
        for i in range(1, n):
            # если P[i]-ая точка лежит левее P[0]-ой точки
            if pts[P[i]][0] < pts[P[0]][0]:
                # меняем местами номера этих точек
                P[i], P[0] = P[0], P[i]

        for i in range(2, n):
            j = i
            while j > 1 and (self.rotate(pts[P[0]], pts[P[j-1]], pts[P[j]]) < 0): # noqa E501
                P[j], P[j-1] = P[j-1], P[j]
                j -= 1

        # у нас есть последовательность обхода, осталось сделать так, чтобы она начиналась с top-left # noqa E501
        rect = pts[P]
        # Найдем верхнюю сторону - это будет наиболее горизонтальная сторона из первых 2-х # noqa E501
        top_line = 0
        if rect[0, 0] == rect[1, 0]:
            top_line = 1
        elif rect[1, 0] == rect[2, 0]:
            top_line = 0
        else:
            tan_1 = np.abs((rect[0, 1] - rect[1, 1]) / (rect[0, 0] - rect[1, 0])) # noqa E501
            tan_2 = np.abs((rect[1, 1] - rect[2, 1]) / (rect[1, 0] - rect[2, 0])) # noqa E501
            top_line = 0 if tan_1 < tan_2 else 1

        order = list(range(4))
        if top_line == 1:
            order = order[1:] + order[:1]
        return rect[order].astype(np.float32)

    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        max_h, max_w = image.shape[:2]
        if self.border > 0:
            w = (tr[0] + br[0] - tl[0] - bl[0]) // 2
            h = (tr[1] + br[1] - tl[1] - bl[1]) // 2
            tl[0] = np.max([0, tl[0] - w * self.border / 2])
            tl[1] = np.max([0, tl[1] - h * self.border * 2])

            tr[0] = np.min([max_w, tr[0] + w * self.border / 2])
            tr[1] = np.max([0, tr[1] - h * self.border * 2])

            br[0] = np.min([max_w, br[0] + w * self.border / 2])
            br[1] = np.min([max_h, br[1] + h * self.border * 2])

            bl[0] = np.max([0, bl[0] - w * self.border / 2])
            bl[1] = np.min([max_h, bl[1] + h * self.border * 2])

            rect = np.array([tl, tr, br, bl], dtype=np.float32)

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        # return the warped image
        return warped


class PlateImageAdjuster():
    def __init__(self, contrast=60, max_out_tolerance=0.3,
                 sharpen_alpha=1.2, color_balance_alpha=0.15,
                 size=(640, 128)):
        self.contrast = contrast
        self.max_out_tolerance = max_out_tolerance
        self.sharpen_alpha = sharpen_alpha
        self.color_balance_alpha = color_balance_alpha
        self.size = size

    def __call__(self, image):
        adjusted_image = self.add_contrast(image)
        adjusted_image = self.resize_sharpen(adjusted_image)
        adjusted_image = self.color_balance(adjusted_image)
        return adjusted_image

    def add_contrast(self, plate_img):
        img_contrast = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY).std()
        maxed_out = np.sum(plate_img == 255) / np.product(plate_img.shape)
        factor = self.contrast / img_contrast

        adjusted = np.clip(plate_img * factor, a_min=None, a_max=255)
        maxed_out_adjusted = np.sum(adjusted == 255) / np.product(adjusted.shape) # noqa E501
        adjusted_contrast = cv2.cvtColor(np.float32(adjusted),
                                         cv2.COLOR_BGR2GRAY).std()
        if np.abs(adjusted_contrast - self.contrast) > np.abs(img_contrast - self.contrast): # noqa E501
            return plate_img
        if maxed_out_adjusted - maxed_out > self.max_out_tolerance:
            return plate_img
        return adjusted.astype(int)

    def resize_sharpen(self, image):
        resized = cv2.resize(image.astype(np.float),
                             self.size,
                             interpolation=cv2.INTER_CUBIC).astype(int)
        w_f = self.size[0] / image.shape[1]
        h_f = self.size[1] / image.shape[0]
        f_2 = np.sqrt(w_f * h_f)

        alpha = f_2 * f_2 * 2
        adjusted_plate = cv2.GaussianBlur(resized.astype(np.float32), (3,3), 2)# noqa E501
        adjusted_plate = cv2.addWeighted(resized.astype(int), 1 + alpha,
                                         adjusted_plate.astype(int),
                                         -alpha, 0.0)
        return np.clip(adjusted_plate, a_min=0, a_max=255).astype(int)

    def color_balance(self, image):
        low, high = [], []
        image_1 = image.copy()
        for k in range(3):
            low.append(np.quantile(image_1[:, :, k], self.color_balance_alpha / 2)) # noqa E501
            high.append(np.quantile(image_1[:, :, k], 1 - self.color_balance_alpha / 2)) # noqa E501

        for k in range(3):
            image_1[:, :, k][image_1[:, :, k] >= high[k]] = high[k]
            image_1[:, :, k][image_1[:, :, k] <= low[k]] = low[k]
            min_k = np.min(image_1[:, :, k])
            max_k = np.max(image_1[:, :, k])
            image_1[:, :, k] = (image_1[:, :, k] - min_k)/(max_k-min_k) * 255
        return image_1.astype(int)


def build_mask(raw_box, image):
    mask = np.zeros(shape=image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array(raw_box), 1)
    return mask


def get_rectangular_box(box):
    box = np.array(box)
    x_0 = box[:, 0].min().reshape(-1)
    y_0 = box[:, 1].min().reshape(-1)
    x_1 = box[:, 0].max().reshape(-1)
    y_1 = box[:, 1].max().reshape(-1)
    return np.clip(np.array([x_0, y_0, x_1, y_1]),
                   a_min=0, a_max=None).reshape(-1)
