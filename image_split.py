import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageOps
import json
import io

def merge_image(pdf, positions, direction="vertical", scale = False):
    """
    pdf: fitz.Document 对象
    positions: [{'page': 0, 'bbox': (x0, y0, x1, y1)}, ...]
    direction: "vertical" or "horizontal"
    """
    crops = []
    
    for pos in positions:
        page = pdf[pos["page"]-1]
        # 获取页面的所有图片
        img_list = page.get_images(full=True)
        if not img_list:
            continue
        
        # 取第一个图片
        xref = img_list[0][0]
        base_image = pdf.extract_image(xref)
        img_data = base_image["image"]
        img = Image.open(io.BytesIO(img_data))
        
        #img = ImageOps.invert(img)
        
        # 注意：PDF 页面和图像坐标可能不一致，通常需要用 matrix 转换
        # 这里假设第一个图像覆盖整个页面（常见情况：扫描件）
        # 所以直接按照 bbox 比例在图像上裁剪
        page_rect = page.rect
        x0, y0, x1, y1 = pos["bbox"]
        
        # 转换 bbox 到图像坐标
        w, h = img.size
        if scale:
            crop_box = (
                int(x0 / page_rect.width * w),
                int(y0 / page_rect.height * h),
                int(x1 / page_rect.width * w),
                int(y1 / page_rect.height * h),
            )
        else:
            crop_box = (x0, y0, x1, y1)
        crop_img = img.crop(crop_box)
        crops.append(crop_img)

    # 拼接图片
    if not crops:
        return None
    
    if direction == "vertical":
        total_width = max(img.width for img in crops)
        total_height = sum(img.height for img in crops)
        result = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        
        y_offset = 0
        for img in crops:
            result.paste(img, (0, y_offset))
            y_offset += img.height
    else:  # horizontal
        total_width = sum(img.width for img in crops)
        total_height = max(img.height for img in crops)
        result = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        
        x_offset = 0
        for img in crops:
            result.paste(img, (x_offset, 0))
            x_offset += img.width
            
    return result


def split_images(pdf_doc, positions):
    for pos in positions:
        merged_img = merge_image(pdf_doc, pos['position'])
        if merged_img:
            merged_img.convert("L").save(f"./split_img/{pos['id']}.jpg")

if __name__ == "__main__":
    # 示例用法
    pdf_path = "./data/ocr-法汉双解词典-2130p.pdf"
    pdf = fitz.open(pdf_path)
    for i,page in enumerate(pdf):
        if len(page.get_images()) != 1:
            print(f"警告： 第 {i+1} 页 有多个图片对象")
    with open("./data/image_pos.json", "r", encoding="utf-8") as f:

        positions = json.load(f)
    
    split_images(pdf, positions)