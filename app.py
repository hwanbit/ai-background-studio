import io
import numpy as np
import torch
from flask import Flask, render_template, request, send_file
from PIL import Image
from transformers import SamModel, SamProcessor

app = Flask(__name__)

# 전역 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading SAM model on {device}...")

try:
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    processor = None

# 이미지 바이트를 받아 SAM을 실행하고, 배경이 투명한(RGBA) 사람 이미지를 바이트로 반환
def run_sam_inference(image_bytes):
    if model is None or processor is None:
        raise RuntimeError("Model not loaded")

    # 이미지 로드
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size

    # 포인트 프롬프트
    # 상단(얼굴), 중단(가슴), 하단(배) 3개의 점을 찍어 y축을 따라 점을 배치
    input_points = [[[w // 2, h // 4], [w // 2, h // 2], [w // 2, 3 * h // 4]]]

    # 전처리 및 추론
    inputs = processor(
        images=image,
        input_points=input_points,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # 마스크 후처리
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )

    # SAM은 하나의 입력에 대해 3개의 마스크(Whole, Part, Sub-part)를 도출하는데, iou 점수가 가장 높거나, 가장 큰 면적을 가진 마스크를 선택
    iou_scores = outputs.iou_scores.cpu()

    # 점수가 가장 높은 마스크 인덱스 추출
    best_idx = torch.argmax(iou_scores[0, 0]).item()
    best_mask = masks[0][0, best_idx].numpy()  # (H, W) bool array

    # 투명 배경 이미지 생성 (RGBA)
    image_np = np.array(image)  # (H, W, 3) RGB

    # 알파 채널 생성: 마스크가 True인 곳은 255(불투명), False인 곳은 0(투명)
    alpha_channel = (best_mask * 255).astype(np.uint8)

    # RGBA 이미지 병합
    rgba_image = np.dstack((image_np, alpha_channel))

    # 결과를 바이트로 변환
    output_image = Image.fromarray(rgba_image)
    img_io = io.BytesIO()
    output_image.save(img_io, 'PNG')
    img_io.seek(0)

    return img_io

# html과 연결
@app.route('/')
def index():
    return render_template('index.html')

# 이미지 업로드 및 다운로드
@app.route('/process', methods=['POST'])
def process_image():
    if 'person_image' not in request.files:
        return "No image uploaded", 400

    file = request.files['person_image']
    if file.filename == '':
        return "No selected file", 400

    try:
        image_bytes = file.read()
        result_io = run_sam_inference(image_bytes)

        return send_file(
            result_io,
            mimetype='image/png',
            as_attachment=False,
            download_name='segmented_person.png'
        )
    except Exception as e:
        print(f"Processing Error: {e}")
        return f"Error: {str(e)}", 500

# 메인문
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)