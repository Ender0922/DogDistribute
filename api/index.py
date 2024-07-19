import imageio
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import tensorflow as tf
import base64
from PIL import Image
import io
import pandas as pd

app = Flask(__name__)

# index 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')

# 이미지 전처리 함수
IMG_SIZE = 224

def preprocess_image(image):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
    return image.numpy().reshape(1, IMG_SIZE, IMG_SIZE, 3)

# 이미지 업로드에 대한 예측값 반환
@app.route('/predict', methods=['POST'])
def make_prediction():
    file_path = './dog_data.xlsx'
    dog_data = pd.read_excel(file_path)

    # Convert the DataFrame to a dictionary
    dog_dict = dog_data.set_index('반려견 종 이름').T.to_dict()
    dog_breeds = [
        "치와와",
        "일본 스파니엘",
        "몰티즈",
        "페키니즈",
        "시추",
        "블레넘 스파니엘",
        "파피용",
        "토이 테리어",
        "로디지안 리지백",
        "아프간 하운드",
        "바셋 하운드",
        "비글",
        "블러드하운드",
        "블루틱",
        "블랙 앤 탄 쿤하운드",
        "워커 하운드",
        "잉글리쉬 폭스하운드",
        "레드본",
        "보르조이",
        "아이리시 울프하운드",
        "이탈리안 그레이하운드",
        "휘핏",
        "이비잔 하운드",
        "노르웨이 엘크하운드",
        "오터하운드",
        "살루키",
        "스코티시 디어하운드",
        "와이마라너",
        "스태퍼드셔 불테리어",
        "아메리칸 스태퍼드셔 테리어",
        "베들링턴 테리어",
        "보더 테리어",
        "케리 블루 테리어",
        "아이리시 테리어",
        "노퍽 테리어",
        "노리치 테리어",
        "요크셔 테리어",
        "와이어 헤어드 폭스 테리어",
        "레이크랜드 테리어",
        "실리햄 테리어",
        "에어데일 테리어",
        "케언 테리어",
        "오스트레일리안 테리어",
        "댄디 딘몬트 테리어",
        "보스턴 불",
        "미니어처 슈나우저",
        "자이언트 슈나우저",
        "스탠다드 슈나우저",
        "스카치 테리어",
        "티베탄 테리어",
        "실키 테리어",
        "소프트 코티드 휘튼 테리어",
        "웨스트 하일랜드 화이트 테리어",
        "라사 압소",
        "플랫 코티드 리트리버",
        "컬리 코티드 리트리버",
        "골든 리트리버",
        "래브라도 리트리버",
        "체서피크 베이 리트리버",
        "저먼 쇼트헤어 포인터",
        "비즐라",
        "잉글리쉬 세터",
        "아이리시 세터",
        "고든 세터",
        "브리터니 스파니엘",
        "클럼버 스파니엘",
        "잉글리쉬 스프링거 스파니엘",
        "웰시 스프링거 스파니엘",
        "코커 스파니엘",
        "서식스 스파니엘",
        "아이리시 워터 스파니엘",
        "쿠바즈",
        "스키퍼키",
        "그루넨달",
        "말리노이즈",
        "브리아드",
        "켈피",
        "코몬도르",
        "올드 잉글리쉬 쉽독",
        "셰틀랜드 쉽독",
        "콜리",
        "보더 콜리",
        "부비에 데 플랑드르",
        "로트와일러",
        "저먼 셰퍼드",
        "도베르만",
        "미니어처 핀셔",
        "그레이터 스위스 마운틴 도그",
        "버니즈 마운틴 도그",
        "아펜젤러",
        "엔틀레부처",
        "복서",
        "불 마스티프",
        "티베탄 마스티프",
        "프렌치 불도그",
        "그레이트 데인",
        "세인트 버나드",
        "에스키모 도그",
        "말라뮤트",
        "시베리안 허스키",
        "아펜핀셔",
        "바센지",
        "퍼그",
        "레온베르거",
        "뉴펀들랜드",
        "그레이트 피레니즈",
        "사모예드",
        "포메라니안",
        "차우차우",
        "키스혼드",
        "브라반손 그리폰",
        "펨브로크 웰시 코기",
        "카디건 웰시 코기",
        "토이 푸들",
        "미니어처 푸들",
        "스탠다드 푸들",
        "멕시칸 헤어리스",
        "딩고",
        "드홀",
        "아프리칸 헌팅 도그"
    ]

    if request.method == 'POST':

        # 업로드 파일 처리 분기
        file = request.files['image']
        if not file: 
            return render_template('index.html', breed="No Files")

        # 이미지 픽셀 정보 읽기 및 전처리
        img = imageio.imread(file)
        img = img[:, :, :3]
        img = preprocess_image(img)

        # 특징 추출
        features = base_model.predict(img)
        features = features.reshape(1, -1)

        # 입력 받은 이미지 예측
        prediction = model.predict(features)

        # 예측 값을 1차원 배열로부터 확인 가능한 문자열로 변환
        breed = dog_breeds[prediction[0]]

        # 이미지 base64로 인코딩
        buffered = io.BytesIO()
        img_pil = Image.fromarray((img[0] * 255).astype(np.uint8))
        img_pil.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        dog_info = dog_dict[breed]

        # 결과 리턴
        return jsonify({
            'imageBase64': image_base64,
            'breed': breed,
            'size': dog_info['크기'],
            'lifetime': dog_info['수명'],
            'personality': dog_info['성격'],
            'environment': dog_info['살아가는 환경(온도 및 기타 외부요인)'],
            'feedamount': dog_info['사료 섭취량'],
            'working': dog_info['1일 산책 시간'],
            'disease': dog_info['특이 질병'],
            'caution': dog_info['키울때 주의해야 할 점']
        })

# 미리 학습시켜서 만들어둔 모델 로드
if __name__ == '__main__':
    # 사전 학습된 MobileNetV2 모델 로드
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    # RandomForest 분류기 로드
    model = joblib.load('dog_breed_classifier.pkl')
