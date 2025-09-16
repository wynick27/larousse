import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import json
import cv2
import io
from sklearn.cluster import KMeans

# -----------------------------
# 配置参数（需要时调整）
# -----------------------------
PDF_PATH = "./data/拉鲁斯法汉双解词典.pdf"
PAGE_NUMBER = 70  # 0-based index
MIN_WORD_HEIGHT = 5      # 忽略过小噪点（行高度最小）
HEADER_ROW_RATIO = 0.6   # 顶部行黑色像素比例高于这个值认为是页眉/横线
BLANK_COL_RATIO = 0.02   # 每列黑色像素低于这个比例认为是“空白列”
LEFT_OFFSET_PIX_MIN = 23 # 左偏移阈值最小像素
LEFT_OFFSET_RATIO = 0.06 # 左偏移阈值按列宽的比例
OUTPUT_JSON = "entries.json"
OUTPUT_IMG = "annotated_page.png"


def get_image(page):
    img_list = page.get_images(full=True)

    if img_list:
        # 通常词典每页就是一张整页图像，取第一个
        xref = img_list[0][0]
        base_image = page.parent.extract_image(xref)
        image_bytes = base_image["image"]
        img = Image.open(io.BytesIO(image_bytes))

        # 这时 img 就是原始扫描图（大小与 PDF 内部存储一致，不会再放大）
        return img
    return None


def classify_wordlines(left_positions):
    min_left = min(left_positions) if left_positions else 0
    max_left = max(left_positions) if left_positions else 0
    
    lines = []
    if max_left - min_left < 30: #无headword
        lines.append((0,len(left_positions) -1 , False))
        return lines
    
    l_array = np.array(left_positions).reshape(-1, 1)

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(l_array)
    labels = kmeans.labels_

    # 哪个簇更靠左，就是词头簇
    cluster_means = [np.mean(l_array[labels == i]) for i in range(2)]
    
       #return classify_wordlines(left_positions,False)
    wordhead_cluster = np.argmin(cluster_means)

    headword_lines0 = [i for i, lbl in enumerate(labels) if lbl == wordhead_cluster]
        
    left_offset_threshold = min_left + LEFT_OFFSET_PIX_MIN
    headword_lines1 = [i for i, lpos in enumerate(left_positions) if lpos <= left_offset_threshold]

    if len(headword_lines0) != len(headword_lines1):
        if abs(cluster_means[1] - cluster_means[0]) < 20:
            headword_lines = headword_lines1     #距离没有分散开，可能词条过少
        elif len(headword_lines1) < len(headword_lines0):
            if len(headword_lines1) < 3:
                headword_lines = headword_lines1 #噪点影响
            elif abs(cluster_means[1] - cluster_means[0]) >25 and abs(cluster_means[1] - cluster_means[0]) < 60:
                headword_lines = headword_lines0
            else:
                difference  = set(headword_lines0) - set(headword_lines1)
                headword_lines = headword_lines0
                for index in difference:
                    if left_positions[index] > min_left + LEFT_OFFSET_PIX_MIN + 10:
                        headword_lines = headword_lines1
                        break
            
            

                headword_lines = headword_lines0 if len(headword_lines0) < len(headword_lines1) or len(headword_lines1) < 3 else headword_lines1
        else:
            headword_lines = headword_lines0
    else:
        headword_lines = headword_lines1 
    if headword_lines[0] != 0:
        lines.append((0,headword_lines[0] - 1,False))
    headword_lines.append(len(left_positions))
    for i in range(len(headword_lines)-1):
        lines.append((headword_lines[i],headword_lines[i+1] - 1,True))

    return lines

def get_annotation(img,deskew = False):

    img_w, img_h = img.size

    result = {}

    # 灰度 & 自适应二值化（文字为 255, 背景为 0）
    gray = np.array(img.convert("L"))
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    row_sum = np.sum(binary !=0, axis=1) / binary.shape[1]  # 每行黑色像素比例
    header_rows = 0
    for i, r in enumerate(row_sum):
        if r > HEADER_ROW_RATIO:
            header_rows = i + 1
        else:
            # 如果已检测到一些页眉行而现在降下来，则停止（避免把正文误认为页眉）
            if header_rows > 0:
                break
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)  #去噪点
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area <= 20 :
            binary[labels == i] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    

    header = None
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morph, connectivity=8)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w > 0.5*binary.shape[1] and h <= 20:  # 宽很大，高度很小
            header = [int(x),int(y),int(w),int(h)]
            print("横线 bbox:", x, y, w, h)
            header_rows = y+h
            result['header'] = header
    if not header:
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if h > 50 and (img_w*0.4 < x < img_w*0.6): #h > 3*avg_line_height and
                print("发现中间大字母:", x,y,w,h)
                header = [int(x),int(y),int(w),int(h)]
                header_rows = y+h
                result['letter'] = header
    # binary_trim 用于后续的行/列检测，但我们保留 header_rows 以便作坐标映射
    binary_trim = binary[header_rows:, :] if header_rows < binary.shape[0] else binary.copy()

    # -----------------------------
    # 3. 自动检测两栏分隔线（在 trimmed binary 上）
    #    找到连续最长的“空白列段”，取其中点作为分隔线
    # -----------------------------
    col_sum = np.sum(binary_trim != 0, axis=0) / (binary_trim.shape[0] + 1e-9)
    blank_cols = np.where(col_sum < BLANK_COL_RATIO)[0]

    if len(blank_cols) > 0:
        # 将连续的列聚成段，找最长段
        segments = np.split(blank_cols, np.where(np.diff(blank_cols) != 1)[0] + 1)
        best_seg = max(segments[1:-1], key=lambda s: s.size)
        mid_col = int(best_seg[best_seg.size * 2 // 5])
    else:
        mid_col = binary_trim.shape[1] // 2  # 兜底

    result['mid_column'] = int(mid_col)

    # columns 是相对于整页宽度（因为我们只对行做了 trim，列索引未变）
    columns = [(0, mid_col), (mid_col, binary_trim.shape[1])]

    # -----------------------------
    # 4. 每栏：行检测 + 左边界判断词头 → 生成词条坐标（映射回原始图像坐标）
    # -----------------------------
    entries = []

    result['entries'] = entries
    

    for col_idx, (x_start, x_end) in enumerate(columns):
        col_slice = binary_trim[:, x_start:x_end]

        # 行检测（垂直投影）
        row_sum_col = np.sum(col_slice != 0, axis=1)
        lines = []
        in_line = False
        for i, s in enumerate(row_sum_col):
            if s > 0 and not in_line:
                start = i
                in_line = True
            elif s == 0 and in_line:
                end = i
                in_line = False
                if end - start >= MIN_WORD_HEIGHT:
                    lines.append((start, end))
        if in_line:
            lines.append((start, len(row_sum_col)))

        if not lines:
            continue

        # 每行的左边界（相对于该栏的左边 x_start）
        left_positions = []
        for top, bottom in lines:
            line_slice = col_slice[top:bottom, :]
            left_indices = np.where(np.any(line_slice != 0, axis=0))[0]
            left = int(left_indices[0]) if len(left_indices) > 0 else (x_end - x_start - 1)
            left_positions.append(left)

        # 根据最小左边界设阈值（比例+最小像素）
        col_width = x_end - x_start
        

        headword_lines = classify_wordlines(left_positions)

        for start_line, end_line, is_headword in headword_lines:
            entry_top = lines[start_line][0] + header_rows
            entry_bottom = lines[end_line][1] + header_rows
            # x 需映射到原始坐标系（x_start 在 trimmed 上与原图一致）
            entry_coords = [int(x_start), int(entry_top), int(x_end), int(entry_bottom)]
            entries.append({"column": col_idx, "coords": entry_coords, 'is_headword': is_headword})

    return result

def draw_page(img,annotation):
    rgb =ImageOps.invert(img).convert("RGB")
    draw = ImageDraw.Draw(rgb)

    # 画页眉分界线（若检测到）
    if header := annotation.get('header'):
        x,y,w,h = header
        draw.rectangle([(x, y), (x+w, y+h)], outline="blue", width=2)
    if letter := annotation.get('letter'):
        x,y,w,h = letter
        draw.rectangle([(x, y), (x+w, y+h)], outline="blue", width=2)

    # 画两栏分隔线（从页眉下方到页面底部）
    if mid_col := annotation.get('mid_column'):
    
        draw.line([(mid_col, 0), (mid_col, img.size[1])], fill="green", width=2)
    
    for entry in annotation.get('entries',[]):
        entry_coords = entry['coords']
        draw.rectangle(entry_coords, outline="red", width=2)

    return rgb

def rotate_image(image, angle, change_size = False):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)  # 旋转中心

    # 得到旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    if change_size:
        # 计算旋转后的边界，避免裁掉内容
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
    else:
        new_w = w
        new_h = h

    # 调整旋转矩阵的平移量
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 仿射变换
    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated
def deskew_image(image):
    gray = np.array(image.convert("L"))
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    coords = np.column_stack(np.where(morph > 0))
    rect = cv2.minAreaRect(coords)   # (center, (w,h), angle)
    angle = rect[-1]
    
    # OpenCV 的 angle 取值规则：
    # -90 ~ 0
    # 如果 w < h，angle 倾向于竖直，需要加 90°
   # if angle < -45:
    #    angle = -(90 + angle)
    #else:
    #    angle = -angle
    if angle < 80:
        angle = angle
    else:
        angle = 90 - angle
    result =  rotate_image(binary,angle)
    return Image.fromarray(result)
    #Image.fromarray(result).save("deskew.png")

def annotate_pages(pdf_path, start_page, json_path = None, img_dir = None, deskew = False):
    doc = fitz.open(pdf_path)
    if isinstance(start_page,list):
        pages = start_page
    else:
        pages = range(start_page,doc.page_count)
    result = []
    for page_num in pages:
        page = doc[page_num]
        print(f"处理： {page_num + 1}")
        img = get_image(page)
        if deskew:
            img = deskew_image(img)
            #img.save(f'img/page_{page_num+1:04}_deskew.png')

        annotation = get_annotation(img)
        annotation['page'] = page_num + 1
        result.append(annotation)

        if img_dir:
            annotate_img = draw_page(img, annotation)
            annotate_img.save(f'img/page_{page_num+1:04}.png')
        if json_path:
            with open(f'json/page_{page_num+1:04}.json', "w", encoding="utf-8") as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)


    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    
    print(f"完成：输出 {json_path} （词条坐标）和 {img_dir} （标注图）")


if __name__ == '__main__':
    error_imgs = [132, 134, 150, 192, 197, 206, 245, 272, 291, 345, 365, 370, 380, 477, 486, 507, 542, 576, 607, 615, 630, 633, 635, 686, 688, 689, 703, 706, 752, 754, 790, 802, 822, 827, 832, 848, 870, 891, 892, 896, 905, 907, 914, 919, 956, 966, 1015, 1022, 1053, 1107, 1140, 1142, 1146, 1150, 1151, 1154, 1158, 1212, 1221, 1227, 1247, 1253, 1255, 1267, 1286, 1305, 1308, 1332, 1360, 1397, 1405, 1408, 1420, 1453, 1480, 1500, 1503, 1520, 1530, 1541, 1552, 1556, 1592, 1616, 1649, 1656, 1660, 1748, 1749, 1766, 1795, 1828, 1836, 1838, 1840, 1845, 1859, 1916, 1918, 1942, 1980, 1987, 1992, 2008, 2022, 2038, 2062, 2070, 2082, 2110, 2120] 
    annotate_pages(PDF_PATH,error_imgs, OUTPUT_JSON, OUTPUT_IMG, True)

    annotate_pages(PDF_PATH,[2120], OUTPUT_JSON, OUTPUT_IMG)