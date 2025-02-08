import cv2
import numpy as np
import matplotlib.pyplot as plt

def order_points(pts):
    """
    4개의 좌표를 (좌상, 우상, 우하, 좌하) 순서로 정렬합니다.
    """
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    # 좌표들의 x+y 값 합계를 계산하여, 가장 작은 값은 좌상, 가장 큰 값은 우하로 지정
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 좌상
    rect[2] = pts[np.argmax(s)]  # 우하
    
    # 좌표들의 y-x 차이를 계산하여, 가장 작은 값은 우상, 가장 큰 값은 좌하로 지정
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 우상
    rect[3] = pts[np.argmax(diff)]  # 좌하
    
    return rect

def four_point_transform(image, pts):
    """
    입력 이미지와 4개 좌표(pts)를 받아 perspective 변환을 수행하고
    스캔된 문서 이미지를 반환합니다.
    """
    # 좌표 정렬
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 문서의 가로 길이를 계산 (좌측과 우측 상단/하단의 길이 중 큰 값)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 문서의 세로 길이를 계산 (상단과 하단 좌측/우측의 길이 중 큰 값)
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 변환 후 대상 좌표 (좌상부터 시계방향)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # 변환 행렬 계산 후 perspective transform 적용
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def scan_document(image: np.ndarray):
    # 1. 이미지 복사본 생성
    if image is None:
        print("이미지를 불러올 수 없습니다.")
        return None
    orig = image.copy()

    # 2. 전처리: 그레이 스케일 변환, 가우시안 블러, Canny 엣지 검출
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # 디버깅용@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # cv2.imshow("Edged", edged)
    # cv2.waitKey(0)

    # 3. 컨투어 검출: 외곽선을 찾아 내림차순(면적 기준)으로 정렬
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # 4. 가장 큰 컨투어 중 꼭짓점이 4개인 컨투어를 찾아 문서 영역으로 간주
    docContour = None
    for c in contours:
        # 컨투어의 둘레 길이 계산
        peri = cv2.arcLength(c, True)
        # 컨투어 근사화: contour의 모양을 단순화
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            docContour = approx
            break

    if docContour is None:
        print("문서 영역을 찾을 수 없습니다.")
        return None

    # 디버깅용@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    cv2.drawContours(image, [docContour], -1, (0, 255, 0), 2)
    # cv2.imshow("Document Contour", image)
    # cv2.waitKey(0)

    # 5. 검출한 4개 좌표를 이용해 perspective 변환하여 스캔된 이미지 생성
    scanned = four_point_transform(orig, docContour.reshape(4, 2))
    return scanned