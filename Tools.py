import re
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np

ONE_MESSAGE_TIME_DELTA = timedelta(minutes=1)
SKIP_TEXTS = ['사진', '이모티콘', '보이스톡 해요', '동영상', '페이스톡 해요']
SKIP_PATTERNS = [
    re.compile(r'^보이스톡\s+\d{1,2}:\d{2}(:\d{2})?$'),
    re.compile(r'^페이스톡\s+\d{1,2}:\d{2}(:\d{2})?$'),
    re.compile(r'^사진\s+\d+장$'),
    re.compile(r'^파일:\s*.+'),
    re.compile(r'^(http|https)://\S+'),
    re.compile(r'^www\.\S+'),
    re.compile(r'^(\d{1,3}\.){3}\d{1,3}$'),
]


def parse_kakaotalk_message_file(file_path):
    """
    카카오톡 대화 저장하기를 통해 저장된 txt 파일을 파싱하는 함수
    :param file_path: 파일의 경로
    :return:
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    message_pattern = re.compile(r'^\[(.+?)] \[(오전|오후) (\d+):(\d+)] (.+)$')
    user_msgs = defaultdict(list)

    current_user = None
    current_time = datetime(1999, 1, 1)
    current_message = ""

    def should_skip(message):
        if message in SKIP_TEXTS:
            return True
        for pattern in SKIP_PATTERNS:
            if pattern.match(message):
                return True
        return False

    def flush(user, msg):
        if user and msg:
            user_msgs[user].append(msg.strip("JJOINN"))

    for line in lines:
        line = line.strip("\n")
        if not line:
            continue

        msg_match = message_pattern.match(line)
        if msg_match:
            user, ampm, hour, minute, text = msg_match.groups()
            if should_skip(text):
                continue

            hour = int(hour)
            minute = int(minute)
            if ampm == '오후' and hour != 12:
                hour += 12
            if ampm == '오전' and hour == 12:
                hour = 0
            timestamp = datetime(2024, 11, 29, hour, minute)

            if current_user == user and timestamp - current_time <= ONE_MESSAGE_TIME_DELTA:
                current_message += f" JJOINN {text}"
            else:
                flush(current_user, current_message)
                current_user = user
                current_time = timestamp
                current_message = text
        else:
            if line.endswith("님과 카카오톡 대화"):
                continue
            if line.startswith("저장한 날짜 : "):
                continue
            if line.startswith("---------------"):
                continue
            if line.endswith("을 초대했습니다."):
                continue
            current_message += '\n' + line
    flush(current_user, current_message)
    return user_msgs

def clean_text(text):
    if pd.isna(text):
        return np.nan

    text = re.sub(r"[^ㄱ-ㅎ가-힣a-zA-Z0-9 .,!?~_<>]", "", text)

    if text.strip() == "":
        return np.nan
    if re.fullmatch(r'(JJOINN\s*)+', text.strip()):
        return np.nan

    return text.strip()
