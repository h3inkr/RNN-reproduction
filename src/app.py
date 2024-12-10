# config.json instance 불러오기
import json

class Dictobj(object):
    def __init__(self, data):
        for name, value in data.items(): # name: key, value: value
            setattr(self, name, self._wrap(value)) # setattr: 객체에 속성을 추가
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)): # 입력값 value가 특정 컬렉션 타입인 경우,
            return type(value)([self._wrap(v) for v in value]) # 내부값을 재귀적으로 호출하여 처리, return값 형식은 입력값과 같음
        else: # 그 외의 타입인 경우 (중첩된 딕셔너리인 경우)
            return Dictobj(value) if isinstance(value, dict) else value # Dictobj의 인스턴스로 변환하여 반환

def load_config(): # config: config 파일 주소, run_train에서 인자로 받음
    
    config_path = "./config/config.json"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return Dictobj(config)