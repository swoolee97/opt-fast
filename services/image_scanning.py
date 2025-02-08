import cv2
import numpy as np
from PIL import Image
import io

def order_points(pts):
    """ 4개의 좌표를 (좌상, 우상, 우하, 좌하) 순서로 정렬 """
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 좌상
    rect[2] = pts[np.argmax(s)]  # 우하

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 우상
    rect[3] = pts[np.argmax(diff)]  # 좌하

    return rect

def four_point_transform(image, pts):
    """ 4개의 좌표를 이용한 원근 변환 """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def scan_document(file):
    """ OCR 실행 전, 이미지 보정 수행 """
    # 파일을 읽고 OpenCV 이미지로 변환
    file_content = file.file.read()
    image = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return None

    orig = image.copy()

    # 그레이스케일 변환 및 엣지 검출
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # 외곽선 검출 및 정렬
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    docContour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            docContour = approx
            break

    if docContour is None:
        return None  # 문서 영역을 찾을 수 없는 경우

    # 보정된 이미지 생성
    scanned = four_point_transform(orig, docContour.reshape(4, 2))
    return scanned
