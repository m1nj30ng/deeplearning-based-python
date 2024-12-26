2024-2 파이썬 기반 딥러닝 프러젝트로 제작한 HTP검사 웹 서비스 입니다.

사용 데이터는 AIHub의 'AI 기반 아동 미술심리 진단을 위한 그림 데이터 구축'입니다. 
YOLO11을 학습시켜 Object Detection을 통해 피검자가 그린 그림을 인식한 뒤, 크기나 개수 등의 특징을 이용해 해설을 제공합니다. 

cmd 창에 docker pull ghcr.io/m1nj30ng/deeplearning-based-python:v1 을 입력한 다음,
docker run -d -p 8000:8000 ghcr.io/m1nj30ng/deeplearning-based-python:v1 를 입력 한 뒤
http://localhost:8000/ 에 들어가시면 아래와 같은 웹 서비스를 확인할 수 있습니다. 

![image](https://github.com/user-attachments/assets/5f71669c-2d05-4eee-8fc7-eb1154a61c69)
![image](https://github.com/user-attachments/assets/81a5c28e-1808-467a-8d7e-a4771dabc166)
