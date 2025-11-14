import json
import os
import importlib.machinery
import importlib.util
from collections import Counter

# 동폴더의 'algorithm' 스크립트를 모듈로 로드 (확장자 없이 저장된 파일 지원)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALG_PATH = os.path.join(BASE_DIR, "algorithm")
loader = importlib.machinery.SourceFileLoader("algorithm_module", ALG_PATH)
spec = importlib.util.spec_from_loader(loader.name, loader)
algorithm = importlib.util.module_from_spec(spec)
loader.exec_module(algorithm)

Person = algorithm.Person
DataManager = algorithm.DataManager
make_groups = algorithm.make_groups
print_groups = algorithm.print_groups


def main():
    # 입력 명단 (이름, 성별)
    raw_list = [
        ("이성환", "M"),
        ("이원섭", "M"),
        ("최동호", "M"),
        ("염규민", "M"),
        ("김종윤", "M"),
        ("김민서", "M"),
        ("유한웅", "M"),
        ("김재준", "M"),
        ("문현희", "F"),
        ("박서연", "F"),
        ("김수민", "F"),
        ("윤샛별", "F"),
        ("노주영", "F"),
        ("최선영", "F"),
        ("정세이", "F"),
        ("주아진", "F"),
        ("홍승아", "F"),
        ("김예원", "F"),
        ("김시은", "F"),
        ("홍한비", "F"),
        ("김준성", "M"),
        ("김준호", "M"),
        ("마성수", "M"),
        ("김채원", "F"),
        ("문채운", "F"),
        ("최유정", "F"),
    ]

    # 제외할 임원진
    exclude_names = {"이채연", "최주현", "최형우", "문형준"}

    # 고정 임원진 (예시와 동일, 필요시 수정)
    leaders = ['정민서','이수진','우화영','신희영','김성민']
    leaders = [n for n in leaders if n not in exclude_names]

    # 데이터 매니저 로드 (이전 주차 히스토리 반영)
    dm = DataManager(data_file=os.path.join(BASE_DIR, "group_history.json"))

    # Person 생성 (제외 대상 제거)
    people = []
    seen = set()
    for name, gender in raw_list:
        if name in exclude_names:
            continue
        if name in seen:
            continue
        seen.add(name)
        is_leader = name in leaders
        # 연임자 임의 설정: 과거 데이터 없을 때 분산 가중치용
        veteran_candidates = {'김준호','마성수','김채원','문채운','최유정'}
        is_veteran = name in veteran_candidates
        people.append(Person(name=name, gender=gender, is_leader=is_leader, data_manager=dm, is_veteran=is_veteran))

    # 임원진을 people 목록에 포함 (조장으로 배치되도록)
    leader_gender_map = {
        '정민서': 'F', '이수진': 'F', '우화영': 'F', '신희영': 'F', '김성민': 'M'
    }
    for ln in leaders:
        if ln in exclude_names:
            continue
        if ln in seen:
            continue
        seen.add(ln)
        people.append(Person(name=ln, gender=leader_gender_map.get(ln, None), is_leader=True, data_manager=dm, is_veteran=False))

    # 5조 고정, 각 조 7명 고정
    num_groups = 5
    group_capacity = 7

    # 설정값 (성비 균형 강조, 연임자 분산 강화, 이전 주차 겹침 최소화)
    config = {
        'w_week1': 200.0,  # 2주차(최근) 같은 조 회피 강화 (기존 50.0에서 증가)
        'w_week2': 8.0,
        'w_history_overlap': 50.0,  # 전체 히스토리 겹침 방지 강화 (기존 15.0에서 증가)
        'w_vet': 1000.0,  # 연임자 분산 강화 (매우 높은 페널티로 완전 분산 유도)
        'w_gender': 50.0,  # 성비 균형 강화 (기존 6.0에서 증가)
        'target_female_ratio': 0.5,
        'w_unknown_gender': 0.2,
        'w_extreme_gender': 100.0,  # 극단적 성비 불균형 강력 방지 (기존 20.0에서 증가)
    }

    groups, penalty = make_groups(
        people=people,
        leaders=leaders,
        num_groups=num_groups,
        group_capacity=group_capacity,
        config=config,
        data_manager=dm,
        max_iters=2000,  # 최적화 활성화 (기존 0에서 변경)
    )

    # 결과 출력
    print_groups(groups, show_details=True)

    # JSON 저장 (이름만 저장)
    out = {
        "week": 3,
        "groups": [
            {
                "id": g['id'],
                "leader": g['leader'].name if g['leader'] else None,
                "members": [m.name for m in g['members']],
            }
            for g in groups
        ],
    }

    out_path = os.path.join(BASE_DIR, 'week3_groups.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nweek3_groups.json 파일로 저장되었습니다.")


if __name__ == "__main__":
    main()


