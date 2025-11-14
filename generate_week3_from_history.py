import json
import os
import csv
import math
import importlib.machinery
import importlib.util

# 동폴더의 'algorithm' 스크립트를 모듈로 로드
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


def load_weeks_into_history(dm, week_files):
    history = {"groups": [], "people_history": {}}
    for path in week_files:
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # data format: { week, timestamp, groups:[{id,leader,members}...] }
        history["groups"].append(data)
    dm.history = history
    dm.save_history()


def load_people(csv_path, dm, exclude_names=None):
    exclude_names = exclude_names or set()
    people = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['name'].strip()
            if name in exclude_names:
                continue
            gender = (row.get('gender') or '').strip() or None
            is_leader = (row.get('is_leader') or '').strip().lower() in ('1','true','yes','y')
            is_veteran = (row.get('is_veteran') or '').strip().lower() in ('1','true','yes','y')
            people.append(Person(name=name, gender=gender, is_leader=is_leader, is_veteran=is_veteran, data_manager=dm))

    # 중복 제거
    unique = []
    seen = set()
    for p in people:
        if p.name not in seen:
            unique.append(p)
            seen.add(p.name)
    return unique


def choose_num_groups(total_people, leaders_present, min_groups=4, max_groups=7):
    candidates = list(range(min_groups, max_groups + 1))
    candidates = [g for g in candidates if g <= total_people and g > 0]
    if not candidates:
        return max(1, min(total_people, leaders_present or 1))
    # 잔여 인원(waste) 최소화, 동률이면 리더 수와의 차이 최소화, 그 다음 평균 그룹 크기 6에 가까운 것
    best = None
    best_key = None
    for g in candidates:
        waste = total_people % g
        leader_diff = abs((leaders_present or g) - g)
        avg_size = total_people / g
        size_diff = abs(avg_size - 6.0)
        key = (waste, leader_diff, size_diff)
        if best_key is None or key < best_key:
            best = g
            best_key = key
    return best


def main():
    # 설정
    week1_path = os.path.join(BASE_DIR, 'week1_groups.json')
    week2_path = os.path.join(BASE_DIR, 'week2_groups.json')
    csv_path = os.path.join(BASE_DIR, 'hyangyeon_34.csv')
    out_path = os.path.join(BASE_DIR, 'week3_groups.json')

    # 제외할 인원 (예: 불참 임원진)
    exclude = { '이채연' }

    # 데이터 매니저 로드 및 1,2주차 히스토리 주입
    dm = DataManager(data_file=os.path.join(BASE_DIR, 'group_history.json'))
    load_weeks_into_history(dm, [week1_path, week2_path])

    # 사람/성별/리더/연임자 로드
    people = load_people(csv_path, dm, exclude_names=exclude)
    all_names = {p.name for p in people}
    leaders = [p.name for p in people if p.is_leader and p.name not in exclude]

    total_people = len(people)
    num_groups = choose_num_groups(total_people, leaders_present=len(leaders), min_groups=4, max_groups=7)
    group_capacity = math.ceil(total_people / num_groups)

    config = {
        'w_week1': 50.0,
        'w_week2': 8.0,
        'w_history_overlap': 15.0,
        'w_vet': 30.0,
        'w_gender': 6.0,
        'target_female_ratio': 0.5,
        'w_unknown_gender': 0.2,
        'w_extreme_gender': 20.0,
    }

    groups, penalty = make_groups(
        people=people,
        leaders=leaders,
        num_groups=num_groups,
        group_capacity=group_capacity,
        config=config,
        data_manager=dm,
        max_iters=2000,
    )

    print_groups(groups, show_details=True)

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
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nweek3_groups.json 파일로 저장되었습니다.")

    # 최종 검증: 중복 및 누락 확인
    all_member_names = []
    for grp in out['groups']:
        for mem in grp['members']:
            all_member_names.append(mem)
    
    expected = {p.name for p in people}
    actual = set(all_member_names)
    missing = expected - actual
    duplicates = [name for name in all_member_names if all_member_names.count(name) > 1]
    if duplicates:
        from collections import Counter
        dup_dict = Counter(all_member_names)
        print(f"\n경고: 중복된 인원 발견: {[(k, v) for k, v in dup_dict.items() if v > 1]}")
    if missing:
        print(f"\n경고: 누락된 인원: {missing}")
    if not missing and not duplicates:
        print("\n✓ 모든 인원 정상 분배 완료 (중복 없음, 누락 없음)")


if __name__ == '__main__':
    main()


