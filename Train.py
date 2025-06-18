import argparse
from Tools import parse_kakaotalk_message_file, clean_text
from os import mkdir, system
import pandas as pd
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

argparser = argparse.ArgumentParser(description="카카오톡 대화 내역 txt 파일로 구분 모델을 학습합니다.")
argparser.add_argument("--file", "-f", required=True, help="대화 내역 txt 파일 경로")
argparser.add_argument("--save", "-s", default="model", help="학습 결과 모델이 저장될 폴더 경로")
argparser.add_argument("--token_min", default=5, help="토큰의 최소 길이")
argparser.add_argument("--token_max", default=20, help="토큰의 최대 길이")
argparser.add_argument("--word_threshold", default=2, help="단어의 최소 등장 빈도수")
argparser.add_argument("--embedding_dim", default=100)
argparser.add_argument("--hidden_units", default=128)
argparser.add_argument("--batch_size", default=32)
argparser.add_argument("--epochs", default=50)
args = argparser.parse_args()

MIN_LEN = args.token_min
MAX_LEN = args.token_max
WORD_THRESHOLD = args.word_threshold
EMBEDDING_DIM = args.embedding_dim
HIDDEN_UNITS = args.hidden_units
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs

try:
    parse_data = parse_kakaotalk_message_file(args.file)
except FileNotFoundError:
    print(f"파일 {args.file}은 존재하지 않습니다.")
    exit()
except Exception as e:
    print(e)
    exit()

try:
    mkdir(args.save)
except FileExistsError:
    input(f"경로 {args.save}는 이미 존재합니다. 해당 위치를 계속 사용합니다.")
except Exception as e:
    print(e)
    exit()

while True:
    system('cls')
    print(f"{len(parse_data.keys())}명의 사용자가 존재합니다.")
    for index, name in enumerate(parse_data.keys()):
        print(f"{index + 1}. {name}")
    print("그대로 진행하려면 0, 사용자를 삭제하려면 해당 번호를 입력하세요.")
    try:
        num = int(input(">"))
    except ValueError:
        continue

    if num < 0 or num > len(parse_data.keys()):
        continue
    elif num == 0:
        break
    else:
        name = list(parse_data.keys())[num-1]
        del parse_data[name]
        continue

total_data = {
    'label': [],
    'text': []
}
tags = {name:i for i, name in enumerate(parse_data.keys())}

for key, text in parse_data.items():
    for i in text:
        total_data['label'].append(tags[key])
        total_data['text'].append(i)
total_data = pd.DataFrame(total_data)
total_data['text'] = total_data['text'].apply(clean_text)
total_data.drop_duplicates(subset=['text'], inplace=True)
total_data.dropna(subset=['text'], inplace=True)

okt = Okt()

new_texts = []
new_labels = []

for idx, row in total_data.iterrows():
    label = row['label']
    text = row['text']

    tokens = okt.morphs(text)
    token_len = len(tokens)

    if token_len < MIN_LEN:
        continue

    if token_len <= MAX_LEN:
        new_texts.append(text)
        new_labels.append(label)
    else:
        for i in range(token_len - MAX_LEN + 1):
            sliced = tokens[i:i+MAX_LEN]
            new_texts.append(sliced)
            new_labels.append(label)

total_data = pd.DataFrame({'text': new_texts, 'label': new_labels})
total_data.drop_duplicates(subset=['text'], inplace=True)
total_data.dropna(subset=['text'], inplace=True)

label_counts = total_data['label'].value_counts()

system('cls')
print(f"총 데이터 수: {len(total_data)}")
for label, count in label_counts.items():
    print(f"{label}({list(tags.keys())[label]}): {count}({count/len(total_data)*100:.2f}%)")
input("진행하려면 엔터키를 입력하세요.")

system('cls')
X_data, Y_data = total_data['text'], total_data['label']
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

total_cnt = len(tokenizer.word_index)
rare_cnt = 0
total_freq = 0
rare_freq = 0

for key, value in tokenizer.word_counts.items():
    total_freq += value

    if value < WORD_THRESHOLD:
        rare_cnt += 1
        rare_freq += value
vocab_size = total_cnt - rare_cnt + 2

tokenizer = Tokenizer(vocab_size, oov_token= 'OOV')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train, maxlen=MAX_LEN)
X_test = pad_sequences(X_test, maxlen=MAX_LEN)

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM))
model.add(GRU(HIDDEN_UNITS))
model.add(Dense(len(parse_data.keys()), activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint(f"{args.save}/model.keras", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es, mc], validation_split=0.2)


loaded_model = load_model(f"{args.save}/model.keras")
system('cls')
print(f"테스트 정확도: {loaded_model.evaluate(X_test, Y_test)[1]}")

with open(f"{args.save}/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open(f"{args.save}/tags.pkl", "wb") as f:
    pickle.dump(tags, f)

with open(f"{args.save}/max_len.txt", "w") as f:
    f.write(str(MAX_LEN))

print(f"{args.save}에 저장완료")