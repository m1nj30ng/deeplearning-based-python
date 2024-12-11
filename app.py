from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from flask import session
import os
import cv2
import numpy as np
import base64

app = Flask(__name__, static_folder='static', template_folder='templates')
UPLOAD_FOLDER='static/uploads'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

app.secret_key ='secret_key_21012029'

#모델
model = YOLO(r'C:\Users\Minjeong\Desktop\파기딥프젝\best.pt')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        #각 파일 get
        home_file = request.files.get('home_image')
        tree_file = request.files.get('tree_image')
        person_file = request.files.get('person_image')

        #파일 저장
        if home_file:
            home_filename = secure_filename(home_file.filename)
            home_file.save(os.path.join(app.config['UPLOAD_FOLDER'], home_filename))
            session['home_image_path'] = os.path.join(app.config['UPLOAD_FOLDER'], home_filename)
        if tree_file:
            tree_filename = secure_filename(tree_file.filename)
            tree_file.save(os.path.join(app.config['UPLOAD_FOLDER'], tree_filename))
            session['tree_image_path'] = os.path.join(app.config['UPLOAD_FOLDER'], tree_filename)
        if person_file:
            person_filename = secure_filename(person_file.filename)
            person_file.save(os.path.join(app.config['UPLOAD_FOLDER'], person_filename))
            session['person_image_path'] = os.path.join(app.config['UPLOAD_FOLDER'], person_filename)
    return render_template('web.html')


@app.route('/result', methods=['POST'])
def result():
    #detection 결과 리스트
    res = []
    img_path=[]
    #업로드한 이미지 저장 폴더
    image_folder = app.config['UPLOAD_FOLDER']
    #저장된 이미지 가져오기
    
    #HOME
    home_image_path = session.get('home_image_path')
    #이미지 처리
    output_path = app.config['UPLOAD_FOLDER']
    otsu1(home_image_path, output_path)
    #전처리된 이미지 경로
    processed_image_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(home_image_path))[0]}_adaptive.jpg")
    #모델
    results = model(processed_image_path)
    #결과 저장
    output_image_path = os.path.join(image_folder, 'output_' + os.path.basename(processed_image_path))
    results[0].save(output_image_path)
    img_path.append(f"uploads/{os.path.basename(output_image_path)}")
    #분석
    labels,Bboxes,conf,image_height,image_width=test(home_image_path)
    res.append(test_home(labels, Bboxes, conf, image_height, image_width))

    #TREE
    tree_image_path = session.get('tree_image_path')
    #이미지 처리
    output_path = app.config['UPLOAD_FOLDER']
    otsu1(tree_image_path, output_path)
    #전처리된 이미지 경로
    processed_image_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(tree_image_path))[0]}_adaptive.jpg")
    #모델
    results = model(processed_image_path)
    #결과 저장
    output_image_path = os.path.join(image_folder, 'output_' + os.path.basename(tree_image_path)) 
    results[0].save(output_image_path)
    img_path.append(f"uploads/{os.path.basename(output_image_path)}")
    #분석
    labels,Bboxes,conf,image_height,image_width=test(processed_image_path)
    res.append(test_tree(labels, Bboxes, conf, image_height, image_width))
    
    #PERSON
    person_image_path = session.get('person_image_path')
    #이미지 처리
    output_path = app.config['UPLOAD_FOLDER']
    otsu1(person_image_path, output_path)
    #전처리된 이미지 경로
    processed_image_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(person_image_path))[0]}_adaptive.jpg")
    #모델
    results = model(processed_image_path)
    #결과 저장
    output_image_path = os.path.join(image_folder, 'output_' + os.path.basename(person_image_path)) 
    results[0].save(output_image_path)
    img_path.append(f"uploads/{os.path.basename(output_image_path)}")
    #분석
    labels,Bboxes,conf,image_height,image_width=test(processed_image_path)
    res.append(test_person(labels, Bboxes, conf, image_height, image_width))

    return render_template('result.html', results=res, images=img_path, zip=zip)





if __name__ == '__main__':
    #이미지 전처리
    def otsu1(img_path, output_path):
        stream = open(img_path, 'rb')
        bytes_array = bytearray(stream.read())
        np_array = np.asarray(bytes_array, dtype=np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
        #가우시안 블러
        img = cv2.GaussianBlur(img, (5, 5), 0)

        #CLAHE
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        #Adaptive Thresholding
        img = cv2.adaptiveThreshold(img, 255,  cv2.ADAPTIVE_THRESH_MEAN_C,  cv2.THRESH_BINARY, 11, 5)
        

        #Morphological Transformations
        kernel = np.ones((3, 3), np.uint8)
        res = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        res=cv2.fastNlMeansDenoising(img, dst=None, h=10, templateWindowSize=7, searchWindowSize=21)

        filename = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_path, f"{filename}_adaptive.jpg")
        ext = os.path.splitext(save_path)[1]
        result, encoded_img = cv2.imencode(ext, res)
        with open(save_path, mode='wb') as file:
            file.write(encoded_img)


    #detection 결과 가져오기
    def test(img_dir): #원소<-사진
        results = model(img_dir) #사진

        image = cv2.imread(img_dir)
        image_height, image_width, _ = image.shape

        boxes = results[0].boxes
        names = model.names

        labels=[]
        Bboxes=[]
        conf=[]

        for box in boxes:
            #바운딩 박스 좌표 (x_min, y_min, x_max, y_max)
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            #클래스 라벨
            class_id = int(box.cls)  #클래스 ID
            label = names[class_id]  #클래스 이름
            #신뢰도
            confidence = box.conf[0].item()
            #리스트에 값 저장
            labels.append(label)
            Bboxes.append([x_min, y_min, x_max, y_max]) #이중리스트
            conf.append(confidence)

        return(labels, Bboxes, conf, image_height, image_width)
    

    #집 그림 해석
    def test_home(labels, Bboxes, conf, image_height, image_width):
        home_exp=""

        home_exp+="집 그림은 그린 사람이 성장한 가정환경을 나타냅니다. 자신의 가정생활과 가족 관계를 어떻게 바라보고 있는지, 이상적인 가정 또는 과거의 가정에 대한 소망이 드러납니다. "

        if '집전체' in labels:
            home_idx=labels.index('집전체')
            #바운딩 박스의 크기
            home_x=Bboxes[home_idx][2]-Bboxes[home_idx][0]
            home_y=Bboxes[home_idx][3]-Bboxes[home_idx][1]
            home_area=home_x*home_y
            #전체 이미지 크기
            img_area=image_width * image_height
            #전체 용지에 대한 그림 크기 비율
            home_ratio=(home_area/img_area)*100

            if 60 <= home_ratio <= 70:
                home_exp+=f"집 그림은 전체 용지의 약{home_ratio:.2f}%를 차지하며, 일반적인 크기로 그리셨군요. "
            elif home_ratio <= 60:
                home_exp+=f"집 그림은 전체 용지의 약{home_ratio:.2f}%를 차지하며, 작게 그림을 그리셨군요. 작은 그림은 열등감, 불안감, 위축감, 낮은 자존감, 의존성 등을 반영합니다. 어딘가 위축되어 있거나 스트레스를 받고 계시네요. "
            else:
                home_exp+=f"집 그림은 전체 용지의 약{home_ratio:.2f}%를 차지하며, 크게 그림을 그리셨군요. 큰 그림은 공격성, 과장성, 낙천성 등을 반영합니다. "

        #지붕
        home_exp+="지붕은 내적인 인지과정과 관련되어 있습니다. 지붕에 문이나 창문을 그린 경우, 외부 세계와 접촉하는 방법이 공상세계에 몰두해 있는 것과 관련있음을 나타냅니다. "
        if '지붕' in labels:
            roof_idx=labels.index('지붕')
            roof_x=Bboxes[roof_idx][2]-Bboxes[roof_idx][0]
            roof_y=Bboxes[roof_idx][3]-Bboxes[roof_idx][1]
            roof_area=roof_x*roof_y
            roof_ratio=(roof_area/home_area)*100

            if roof_ratio >= 75:
                home_exp+="지붕을 크게 그리셨네요. 큰 지붕은 환상에 과몰입 되어있거나 외부의 사람과의 접촉이 적음을 의미합니다. "
            elif roof_ratio <= 40:
                home_exp+="지붕을 작게 그리셨네요. 작은 지붕은 내적인 인지활동이 활발하지 않음을 의미합니다. "
        else:
            home_exp+="지붕을 생략한 경우, 심리적으로 위축되어있음을 위미합니다. "

        #문
        home_exp+="문은 환경과 직접적인 상호작용, 대인관계에 대한 태도를 나타냅니다. 문이 열려 있다면 외부로부터 정서적 따뜻함을 갈망하고 있는 상태일 수 있습니다. "
        if '문' in labels:
            door_idx=labels.index('문')

            door_x=Bboxes[door_idx][2]-Bboxes[door_idx][0]
            door_y=Bboxes[door_idx][3]-Bboxes[door_idx][1]
            door_area=door_x*door_y
            door_ratio=(door_area/home_area)*100

            if door_ratio >= 15:
                home_exp+="문을 크게 그리셨네요. 너무 큰 문은 사회적 접근을 통해 타인에게 인상적인 존재가 되고싶은 욕구를 나타냅니다. "
            elif door_ratio <= 5:
                home_exp+="문을 작게 그리셨네요. 누군가의 접근을 까다롭게 생각하는 경향이 있거나 수줍음이 많은 성격이시군요. "
            
        else:
            home_exp+="현재 이루기 힘든 것이 있나요? 문이 없음은 현재 얻기 어려운 것이 있음을 의미합니다. "

        #창문
        home_exp+="창문은 환경과의 수동적인 접촉을 나타내며, 피검자의 주관적인 관계 경험과 자기 또는 타인과의 대인관계가 관련되어 있습니다. "
        if '창문' in labels:
            window_idx=np.where(np.array(labels)=='창문')[0]
            window_num=len(window_idx)
            if window_num > 4:
                home_exp+=f"창문을 {window_num}개 그리셨네요. 너무 많은 창문은 애정욕구가 과도함을 의미합니다. "
            elif 2<= window_num <= 4:
                home_exp+=f"창문을 {window_num}개 그리셨네요. 큰 창문이나 많은 창문은 외부세계에 대해 개방적이고 호기심이 많음을 의미합니다. 사회적 상호작용에 적극적이고, 새로운 경험을 추구하는 성향일 가능성이 높습니다. "
        else:
            home_exp+="창문을 그리지 않으셨나요? 창문 생략은 외부 관심의 결여와 적의, 폐쇄적인 사고와 관련되어 있으며, 환경과의 접촉을 차단하고 있는 상태입니다. "

        #굴뚝
        home_exp+="굴뚝은 가족 내의 온정적 분위기, 가족 간의 교류 양상을 나타냅니다. "
        if '굴뚝' in labels:
            home_exp+="굴뚝에서 연기가 나고 있다면 가정 내 따뜻함과 안락함을 느끼고 있을 확률이 높습니다. 그러나 연기를 그리지 않았다면 가정 내 온기가 부족하다는 의미입니다. "
        else:
            home_exp+="굴뚝을 그리지 않으셨네요. 가족 내 분위기가 조금은 정적이고 딱딱한가요? "

        #울타리
        if '울타리' in labels:
            home_exp+="울타리를 그린 사람은 조심성이 높으며, 모험보다는 안전을 추구하고, 비밀이 많거나 신중한 성격일 가능성이 높습니다. 방어적이고 경계적 태도를 가지고 계시네요. "
        #산
        if '산' in labels:
            home_exp+="산을 그리셨네요. 산과 울타리, 관목은 방어 욕구가 높음을 의미합니다. 외부로부터 위협을 느끼고 계시나요? 보호받고 싶어하고, 경계심이 높군요. "
        if '나무' in labels:
            home_exp+="나무는 울타리, 산과 마찬가지로 외부에 대한 경계심이 높고, 보호를 원함을 의미합니다. "
        #꽃 & 잔디
        if '꽃' in labels or '잔디' in labels:
            home_exp+="적당한 잔디와 꽃은 피검자의 생동감과 에너지를 의미하지만 지나친 경우엔 강한 의존 욕구를 반영합니다. "

        #태양
        if '태양' in labels:
            home_exp+="태양은 강한 애정욕구, 의존성 혹은 이에 대한 좌절감을 의미합니다. 부모에 대한 의존성을 나타내기도 합니다."

        return home_exp 
        

    #나무 그림 해석
    def test_tree(labels, Bboxes, conf, image_height, image_width):
        tree_exp=""
        if "나무전체" in labels:
            tree_idx=labels.index('나무전체')
            #바운딩 박스의 크기
            tree_x=Bboxes[tree_idx][2]-Bboxes[tree_idx][0]
            tree_y=Bboxes[tree_idx][3]-Bboxes[tree_idx][1]
            tree_area=tree_x*tree_y
            #전체 이미지 크기
            img_area=image_width * image_height
            #전체 용지에 대한 그림 크기 비율
            tree_ratio=(tree_area/img_area)*100

            if 60 <= tree_ratio <= 70:
                tree_exp+=f"나무의 크기는 자아의 크기와 관련 있습니다. 나무 그림은 전체 용지의 약{tree_ratio:.2f}%를 차지하며, 일반적인 크기로 그리셨군요. "
            elif tree_ratio <= 60:
                tree_exp+=f"나무의 크기는 자아의 크기와 관련 있습니다. 나무 그림은 전체 용지의 약{tree_ratio:.2f}%를 차지하며, 작은 나무를 그리셨군요. 작은 나무는 자존감의 결여를 나타냅니다. "
            else:
                tree_exp+=f"나무의 크기는 자아의 크기와 관련 있습니다. 나무 그림은 전체 용지의 약{tree_ratio:.2f}%를 차지하며, 큰 나무를 그리셨군요. 큰 나무는 자존감이 높음을 의미합니다. "

        tree_exp+="나무는 기본적인 자기 상을 나타냅니다. 그림 검사를 통해 가장 솔직한 내면 세계를 자유롭게 분출 할 수 있습니다. "
        #기둥
        if '기둥' in labels:
            tree_exp+="기둥(줄기)는 자아와 성장을 나타냅니다. 줄기가 곧고 튼튼하고 힘차면 강한 자아와 자신감을 가지고 있으며, 자신의 능력과 자아에 대해 긍정적인 태도를 가지고 있음을 의미합니다. "
            tree_exp+="반면, 가늘고 약하거나, 휘어지거나 부러진 줄기는 자신의 자아에 대해 불안정감을 느끼거나 자신감의 부족, 불안감 등 정서적인 어려움을 의미할 수 있습니다. "

        #수관
        if '수관' in labels:
            crown_idx=labels.index('수관')
            crown_x=Bboxes[crown_idx][2]-Bboxes[crown_idx][0]
            tree_exp+="구름 같이 그린 수관은 공상을 많이 하는 성격일 수 있습니다. 아무렇게나 그린 선으로 뒤범벅된 수관은 혼란, 정서적 불안을 의미합니다. "
            if '기둥' in labels:
                pillar_idx=labels.index('수관')
                pillar_x=Bboxes[pillar_idx][2]-Bboxes[pillar_idx][0]
                ratio=crown_x/pillar_x
                if ratio >= 2:
                    tree_exp+="기둥 폭 보다 수관 폭을 과도하게 크게 그리셨군요. 신체적 활발함 또는 미성숙함, 유아 퇴행 등을 나타냅니다. "
                elif ratio<=1:
                    tree_exp+="기둥 폭 보다 수관 폭을 작게 그리셨군요. 종종 퇴행적 성격을 가진다고 해석됩니다. "
                else:
                    tree_exp+="기둥 폭 보다 수관 폭을 크게 그리셨군요. 열정이 있으며 명성에 대한 욕심이 있음을 나타내거나 현실감각 결여를 의미합니다. "

        #가지
        if '가지' in labels:
            tree_exp+="가지가 무성하고 잎이 풍성하다면 자신의 성취와 성장에 긍정적이며 다양한 사회적 관계와 관심사를 가지고 외부세계와 활발히 교류하고 있음을 의미합니다. "
        else:
            tree_exp+="가지가 적거나 생략되었다면 외부와의 세계에 소극적이거나 사회적 상호작용애 어려움을 겪고 있을 수 있습니다. "

        #뿌리
        if '뿌리' in labels:
            tree_exp+="뿌리는 안정감, 과거와의 연결을 상징하기도 합니다. 지나치게 강조된 뿌리는 현실 접촉을 과도하게 강조하거나 염려하는 상태, 미성숙함 등을 나타냅니다. "
        else:
            tree_exp+="뿌리는 안정감, 과거와의 연결을 상징하기도 합니다. 뿌리와 지면을 모두 그리지 않으셨나요? 현재 불안정한 상태에 놓여있다고 느낄 수 있습니다. "

        #나뭇잎
        if '나뭇잎' in labels:
            tree_exp+="잎이 무성하게 그려졌다면 자신의 발전 가능성에 대해 낙관적인 태도를 가지고 있음을 의미합니다. "
        else:
            tree_exp+="나뭇잎이 생략되어 있다면 고립감을 느끼고 있거나 타인과의 관계 형성에 어려움을 겪고 있으며, 미래에 대한 불안을 가지고 있음을 나타냅니다. 성장에 대한 불안을 느끼고 있을 수 있습니다. "

        #열매
        if '열매' in labels:
            fruit_idx=np.where(np.array(labels)=='열매')[0]
            tree_exp+="가지에 달려있는 열매는 관찰 능력이 뛰어남을 의미합니다. 다소 즉흥적이거나 낙천적인 성격일 수 있습니다. 바닥에 떨어진 열매는 단념, 상실감, 체념, 집중력 결여 등과 관련이 있습니다. "
            tree_exp+="과일 나무를 그렸다면 성취와 성과에 대한 관심이 있음을 나타내며, 사랑과 관심을 주고받고 싶어하는 경향이 있음을 의미합니다. "

        return tree_exp
        

    #사람 그림 해석    
    def test_person(labels, Bboxes, conf, image_height, image_width):
        person_exp=""
        if '사람전체' in labels:
            person_idx=labels.index('사람전체')
            #바운딩 박스의 크기
            person_x=Bboxes[person_idx][2]-Bboxes[person_idx][0]
            person_y=Bboxes[person_idx][3]-Bboxes[person_idx][1]
            person_area=person_x*person_y
            #전체 이미지 크기
            img_area=image_width * image_height
            #전체 용지에 대한 그림 크기 비율
            person_ratio=(person_area/img_area)*100

            if 60 <= person_ratio <= 70:
                person_exp+=f"사람 그림은 전체 용지의 약{person_ratio:.2f}%를 차지하며, 일반적인 크기로 그리셨군요. "
            elif person_ratio <= 60:
                person_exp+=f"사람 그림은 전체 용지의 약{person_ratio:.2f}%를 차지하며, 작게 그림을 그리셨군요. 작은 사람은 자신감 부족 또는 열등감을 나타냅니다. "
            else:
                person_exp+=f"사람 그림은 전체 용지의 약{person_ratio:.2f}%를 차지하며, 크게 그림을 그리셨군요. 큰 사람은 자아에 대한 긍정적 인식을 나타냅니다. "


        person_exp+=""
        #머리
        if '머리' in labels:
            head_idx=labels.index('머리')
            head_x=Bboxes[head_idx][2]-Bboxes[head_idx][0]
            head_y=Bboxes[head_idx][3]-Bboxes[head_idx][1]
            head_area=head_x*head_y
            head_ratio=(head_area/person_area)*100
            person_exp+="머리는 상징적으로 지적 능력의 원천으로 자아의 근원이며, 공상과 사회적 관심, 충동 및 정서적 통제를 나타냅니다. 모자를 쓰거나 머리를 가리는 경우 자신감 부족 또는 현실 세계에 대힌 회피를 의미합니다. "
            if head_ratio >=50:
                person_exp+="머리를 크게 그리셨네요. 이는 과도한 지적 욕구에 대한 불안, 보상심리를 반영합니다. "
            elif head_ratio <=25:
                person_exp+="머리를 지나치게 작게 그리셨네요. 이는 강박성향이 있거나 죄책감에 대한 부정, 수동성, 열등감, 나약한 자아의 반영일 수 있습니다. "


        #얼굴
        if '얼굴' in labels:
            person_exp+="얼굴은 상징적으로 개인이 현실세계와 어떻게 접촉하는지를 나타냅니다. 얼굴의 형태가 흐릿하거나 측면으로 그린 경우 회피적인 성향을 가지고 있을 수 있습니다. "
        else:
            person_exp+="얼굴을 생략하셨나요? 자기 자신을 부정적으로 생각하고 있거나 정체성 혼란, 우울감 등을 느끼고 계시네요. "

        #눈
        if '눈' in labels:
            eye_idx=labels.index('눈')
            eye_x=Bboxes[eye_idx][2]-Bboxes[eye_idx][0]
            eye_y=Bboxes[eye_idx][3]-Bboxes[eye_idx][1]
            eye_area=eye_x*eye_y
            eye_ratio=(eye_area/head_area)*100
            person_exp+="눈은 타인을 어떻게 인식하고 감정표현을 어떻게 하는지 나타냅니다. 눈동자 없이 원모양으로 그린 눈은 자기중심적인 성향 또는 미성숙함, 히스테리함을 나타냅니다. "
            if eye_ratio >= 40:
                person_exp+="눈을 크게 그리셨네요. 이 경우 망상에 빠져 있거나, 의심과 불안을 가지고 있음을 나타냅니다. "
            elif eye_ratio <= 10:
                person_exp+="눈을 작게 그리셨네요. 작은 눈 또는 감고 있는 눈은 내향적 성격과 자기 도취적 성향을 나타냅니다. "


        #코
        if '코' in labels:
            person_exp+="긴 코와 날카로운 코는 공격성을 나타냅니다. 콧구멍을 강조했다면 어딘가 불만이 있는 상태입니다. "
        else:
            person_exp+="코를 생략하셨나요? 인물을 잘 그리고 싶은데 자신이 없으신 분들이 코를 생략하곤 한답니다. 또한 부끄러움을 많이 타고 수줍음이 많거나, 자신감이 없는 사람을 의미합니다. "

        #입
        if '입' in labels:
            person_exp+="입을 강조해서 그린 것은 유아 퇴행적 성향을 의미합니다. 이를 드러나게 그린 경우 적개심과 분노를 나타냅니다. 일직선으로 그은 단선의 입은 방어기제, 거부에 대한 불안을 나타냅니다. "
        else:
            person_exp+="입을 생략하셨네요. 표현에 있어서 문제가 있거나 우울한 상태일 가능성이 높습니다. 타인과 소통하는 것에 어려움이 있지는 않으신가요? "

        #귀
        if '귀' in labels:
            person_exp+="귀는 망상과 관련이 있으며 지나치게 크게 그려진 귀는 청각장애의 가능성 또는 타인에 대한 민감성을 의미합니다. "

        #목
        if '목' in labels:
            person_exp+="목은 머리의 지적(통제력) 영역과 몸통의 감정적(충동적) 영역을 연결하는 기관으로서 상징적으로 충동통제와 관련되어 있습니다. 가늘고 긴 목은 융통성이 없으며 지나치게 도덕적인 경향을 나타내며, 굵고 짧은 목은 난폭하고 완고하며 저동적인 경향을 나타냅니다. "
        else: 
            person_exp+="목을 그리지 않은 사람들은 미숙한 정서를 가지고 있거나 자기애적 성향을 가지고 있을 수 있습니다. "

        #팔
        if '팔' in labels:
            person_exp+="팔이 몸과 밀접할수록 수동적이거나 방어적이며 몸의 바깥으로 향해있으면 외부로 향한 공격적인 성향을 의미합니다. "

        #손
        if '손' in labels:
            person_exp+="크고 명확한 손은 자신감을, 작은 손은 불안감을 의미합니다. "
        else:
            person_exp+="손을 생략함은 무언가 실행/실천 하는 것이 힘들거나 만족스럽지 않음을 의미합니다. 부끄럼이 많거나, 죄책감을 가지고 있을 수도 있습니다. "

        #발
        if '발' in labels:
            person_exp+="발이 명확하게 그려졌다면 피검사자가 자신의 기반과 근거를 중요시함을 의미합니다. "
        else:
            person_exp+="발을 그리지 않으셨네요. 지금 처한 환경에 잘 적응하지 못하는 상태이진 않으신가요? 무언가 스스로 경정하기 어려운 상태이며, 어디론가 도피하고싶어 하시네요. "

        #단추
        if '단추' in labels:
            person_exp+="단추는 모성에 의존함을 나타냅니다. "

        #주머니
        if '주머니' in labels:
            person_exp+="만약 주머니가 가슴위치에 있다면 유아적이고 의존적인 성향을 나타냅니다. "

        return person_exp
    

    app.run(debug=True)