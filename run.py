from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from konlpy.tag import Okt
from Tools import clean_text
import pickle
import argparse
import os

argparser = argparse.ArgumentParser(description="카카오톡 대화 내역 txt 파일로 구분 모델을 학습한 결과를 테스트합니다.")
argparser.add_argument("--path", "-p", required=True, help="결과가 저장된 폴더의 경로")
args = argparser.parse_args()

path = args.path
if not os.path.exists(path):
    print(f"{path}는 올바르지 않은 경로입니다.")
    exit()

try:
    loaded_model = load_model(f"{path}/model.keras")
except Exception as e:
    print(e)
    exit()

with open(f"{path}/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open(f"{path}/tags.pkl", "rb") as f:
    tags = pickle.load(f)
    id_to_user = {v: k for k, v in tags.items()}

with open(f"{path}/max_len.txt", "r") as f:
    max_len = int(f.read().strip())

okt = Okt()

def predict(new_sentence):
    raw_sent = new_sentence
    new_sentence = clean_text(new_sentence)
    tokens = okt.morphs(new_sentence)
    encoded = tokenizer.texts_to_sequences([tokens])
    pad_new = pad_sequences(encoded, maxlen=max_len)

    probs = loaded_model.predict(pad_new)[0]

    print(f"예측 문장: {raw_sent}")
    max_label = None
    max_p = 0
    for i, p in enumerate(probs):
        print(f" - {id_to_user[i]}: {p * 100:.2f}%")
        if p > max_p:
            max_p = p
            max_label = id_to_user[i]
    print(f"최대 확률 > {max_label}({max_p * 100:.2f}%)")

while True:
    os.system('cls')
    raw = input("문장을 입력하세요 (분리된 채팅 사이엔 ' JJOINN ' 을 추가하세요) > ")
    predict(raw)
    os.system('pause')
