#!/usr/bin/env python3
"""
완전 탐색 + DP 방식으로 겹침(과거 같은 조였던 횟수)의 총합을 최소화하는 조 편성 스크립트.

주의사항:
    - 참가자 수가 많고 조 크기가 크면 조합 수가 매우 빠르게 증가합니다.
    - 기본적으로 branch_limit 파라미터로 각 단계에서 고려할 후보 조합 수를 제한해
      현실적인 시간 안에 결과를 얻도록 설계했습니다.
    - branch_limit를 크게 설정할수록 최적 해에 가까워지지만 계산 시간이 기하급수적으로 늘어납니다.

사용 예시:
    python dp_group_allocator.py --weeks 1 2 3 4 --num-groups 5 --branch-limit 80 \\
        --members members_labels.csv --base-path .

출력:
    - 표준 출력에 총 겹침 비용과 각 조 편성 결과를 요약합니다.
    - --output 옵션을 지정하면 해당 경로에 JSON 파일로 결과를 저장합니다.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DP 기반 최소 겹침 조 편성기")
    parser.add_argument(
        "--members",
        default="members_labels.csv",
        help="이름 및 성별 정보를 포함한 CSV 경로 (default: members_labels.csv)",
    )
    parser.add_argument(
        "--weeks",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="겹침 계산에 사용할 주차 번호 목록 (default: 1 2 3 4)",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=5,
        help="편성할 조의 수 (default: 5)",
    )
    parser.add_argument(
        "--group-sizes",
        nargs="+",
        type=int,
        help="조별 인원수 리스트. 지정하지 않으면 총 인원수와 num-groups로 자동 계산",
    )
    parser.add_argument(
        "--group-males",
        nargs="+",
        type=int,
        help="조별 남성 인원수를 지정합니다. 지정하지 않으면 전체 남성 수를 균등 분배합니다.",
    )
    parser.add_argument(
        "--branch-limit",
        type=int,
        default=60,
        help="DP 단계마다 고려할 후보 조합 최대 개수 (default: 60)",
    )
    parser.add_argument(
        "--output",
        help="편성 결과를 저장할 JSON 경로 (생략 시 파일로 저장하지 않음)",
    )
    parser.add_argument(
        "--base-path",
        default=".",
        help="주차별 JSON 파일과 CSV가 위치한 기본 디렉터리 (default: 현재 디렉터리)",
    )
    parser.add_argument(
        "--leaders",
        nargs="+",
        help="조별로 고정할 운영진 이름 목록. 지정한 순서대로 각 조에 배정됩니다.",
    )
    parser.add_argument(
        "--absent",
        nargs="+",
        default=[],
        help="해당 주차에 참석하지 않는 인원 이름 목록",
    )
    parser.add_argument(
        "--veterans",
        nargs="+",
        help="연임자로 분류할 인원 목록. 지정하지 않으면 CSV의 veteran_candidates를 사용합니다.",
    )
    parser.add_argument(
        "--max-veterans-per-group",
        type=int,
        default=1,
        help="한 조에 함께 배치될 수 있는 연임자 최대 인원 (default: 1)",
    )
    parser.add_argument(
        "--same-group",
        nargs="+",
        action="append",
        help="같은 조에 배치해야 하는 인원 그룹. 예: --same-group 염규민 문형준",
    )
    parser.add_argument(
        "--balance-weight",
        type=float,
        default=0.3,
        help="비용 분산 최소화 가중치 (0.0=총비용만, 1.0=분산만, 0.3=하이브리드 기본값)",
    )
    return parser.parse_args()


def load_member_data(csv_path: str) -> Tuple[List[str], Dict[str, str], List[str]]:
    members: List[str] = []
    genders: Dict[str, str] = {}
    veteran_line: str | None = None
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            if not name or name.startswith("veteran_candidates"):
                veteran_line = name
                break
            gender = row[1].strip() if len(row) > 1 else ""
            members.append(name)
            genders[name] = gender or "U"
    if len(set(members)) != len(members):
        raise ValueError("CSV에 중복된 이름이 있습니다. DP 알고리즘은 고유 이름만 처리합니다.")
    veterans: List[str] = []
    if veteran_line:
        eq_idx = veteran_line.find("=")
        if eq_idx != -1:
            subset = veteran_line[eq_idx + 1 :].strip()
            veterans = [v.strip(" '") for v in subset.strip("{} ").split(",") if v.strip()]
    return members, genders, veterans


def load_week_groups(base_path: str, week_numbers: Iterable[int]) -> List[Dict]:
    history = []
    for week in week_numbers:
        path = os.path.join(base_path, f"week{week}_groups.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"주차 파일을 찾을 수 없습니다: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        history.append(data)
    return history


def build_overlap_matrix(members: Sequence[str], history: Iterable[Dict]) -> List[List[int]]:
    index = {name: i for i, name in enumerate(members)}
    size = len(members)
    matrix = [[0] * size for _ in range(size)]

    for week_data in history:
        for group in week_data.get("groups", []):
            names = group.get("members", [])
            for a, b in itertools.combinations(names, 2):
                if a not in index or b not in index:
                    continue
                ia, ib = index[a], index[b]
                matrix[ia][ib] += 1
                matrix[ib][ia] += 1
    return matrix


def compute_group_sizes(total_people: int, num_groups: int) -> List[int]:
    base = total_people // num_groups
    extra = total_people % num_groups
    return [base + 1 if i < extra else base for i in range(num_groups)]


def team_cost(team: Sequence[int], overlap: Sequence[Sequence[int]]) -> int:
    cost = 0
    for i, j in itertools.combinations(team, 2):
        cost += overlap[i][j]
    return cost


def best_combinations(
    first: int,
    remaining: Sequence[int],
    team_size: int,
    overlap: Sequence[Sequence[int]],
    branch_limit: int,
    validator,
) -> List[Tuple[int, Tuple[int, ...]]]:
    """
    first를 포함하는 team_size 인원의 후보 조합 중 겹침 비용이 낮은 상위 branch_limit개를 반환.
    """
    if team_size < 1:
        raise ValueError("team_size는 1 이상이어야 합니다.")
    if team_size == 1:
        team = (first,)
        if not validator(team):
            return []
        return [(0, team)]

    candidates: List[Tuple[int, Tuple[int, ...]]] = []
    # branch_limit가 None 또는 음수일 경우 모든 조합을 고려합니다(매우 느릴 수 있음)
    limit = branch_limit if branch_limit and branch_limit > 0 else None

    # remaining은 이미 first보다 큰 인덱스만 전달받도록 구성
    combos = itertools.combinations(remaining, team_size - 1)
    if limit is None:
        for combo in combos:
            team = (first,) + combo
            if not validator(team):
                continue
            cost = team_cost(team, overlap)
            candidates.append((cost, team))
        candidates.sort(key=lambda x: x[0])
        return candidates

    # limit이 있는 경우: 힙을 사용해 상위 limit개만 유지
    import heapq

    heap: List[Tuple[int, Tuple[int, ...]]] = []
    for combo in combos:
        team = (first,) + combo
        if not validator(team):
            continue
        cost = team_cost(team, overlap)
        heapq.heappush(heap, (-cost, team))
        if len(heap) > limit:
            heapq.heappop(heap)
    candidates = [(-neg_cost, team) for neg_cost, team in heap]
    candidates.sort(key=lambda x: x[0])
    return candidates


def dp_min_overlap(
    group_specs: Sequence[Dict],
    overlap: Sequence[Sequence[int]],
    branch_limit: int,
    genders: Sequence[str],
    veteran_flags: Sequence[bool],
    same_groups: Optional[List[List[str]]] = None,
    member_names: Optional[List[str]] = None,
) -> Tuple[int, List[Tuple[int, ...]]]:
    n = len(overlap)
    full_mask = (1 << n) - 1

    @lru_cache(maxsize=None)
    def solve(mask: int, group_index: int) -> Tuple[int, Tuple[Tuple[int, ...], ...]]:
        if group_index == len(group_specs):
            if mask == full_mask:
                return 0, ()
            return math.inf, ()

        spec = group_specs[group_index]
        required_idx = spec.get("required")

        if required_idx is not None:
            if mask & (1 << required_idx):
                return math.inf, ()
            first = required_idx
        else:
            try:
                first = next(i for i in range(n) if not (mask & (1 << i)))
            except StopIteration:
                return math.inf, ()

        needed = spec["size"]
        remaining = [i for i in range(n) if i != first and not (mask & (1 << i))]
        if needed - 1 > len(remaining):
            return math.inf, ()

        min_male = spec.get("min_male", 0)
        max_male = spec.get("max_male", needed)
        max_veterans = spec.get("max_veterans", len(veteran_flags))

        def validator(team: Tuple[int, ...]) -> bool:
            if required_idx is not None and required_idx not in team:
                return False
            male_count = sum(1 for idx in team if genders[idx] == "M")
            if male_count < min_male or male_count > max_male:
                return False
            vet_count = sum(1 for idx in team if veteran_flags[idx])
            if vet_count > max_veterans:
                return False
            return True

        candidates = best_combinations(first, remaining, needed, overlap, branch_limit, validator)
        best_cost = math.inf
        best_assignment: Tuple[Tuple[int, ...], ...] = ()

        for cost, team in candidates:
            new_mask = mask
            for member in team:
                new_mask |= 1 << member
            tail_cost, tail_assignment = solve(new_mask, group_index + 1)
            if math.isinf(tail_cost):
                continue
            total_cost = cost + tail_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_assignment = (team,) + tail_assignment

        return best_cost, best_assignment

    total_cost, assignment = solve(0, 0)
    if math.isinf(total_cost):
        raise RuntimeError("유효한 조 편성 조합을 찾지 못했습니다. branch_limit를 늘리거나 설정을 조정하세요.")
    
    return total_cost, list(assignment)


def balance_costs(
    assignment: Tuple[Tuple[int, ...], ...],
    overlap: Sequence[Sequence[int]],
    group_specs: Sequence[Dict],
    genders: Sequence[str],
    veteran_flags: Sequence[bool],
    same_groups: Optional[List[List[str]]] = None,
    member_names: Optional[List[str]] = None,
    balance_weight: float = 0.3,
    max_iterations: int = 50,
) -> Tuple[Tuple[int, ...], ...]:
    """하이브리드 방식: 비용 분산 최소화 + 총 비용 증가 최소화"""
    assignment = list(list(team) for team in assignment)
    n_groups = len(assignment)
    
    # same_groups 제약 적용: 같은 조에 있어야 하는 사람들
    if same_groups and member_names:
        name_to_idx = {name: i for i, name in enumerate(member_names)}
        for group in same_groups:
            if len(group) < 2:
                continue
            # 첫 번째 사람이 있는 조를 찾고, 나머지 사람들도 같은 조로 이동
            first_name = group[0]
            if first_name not in name_to_idx:
                continue
            first_idx = name_to_idx[first_name]
            target_group_idx = None
            for i, team in enumerate(assignment):
                if first_idx in team:
                    target_group_idx = i
                    break
            if target_group_idx is None:
                continue
            # 나머지 사람들도 같은 조로 이동
            for name in group[1:]:
                if name not in name_to_idx:
                    continue
                person_idx = name_to_idx[name]
                # 현재 어느 조에 있는지 찾기
                current_group_idx = None
                for i, team in enumerate(assignment):
                    if person_idx in team:
                        current_group_idx = i
                        break
                if current_group_idx is not None and current_group_idx != target_group_idx:
                    # 다른 조에 있으면 이동
                    assignment[current_group_idx].remove(person_idx)
                    assignment[target_group_idx].append(person_idx)
    
    def get_costs() -> List[int]:
        return [team_cost(team, overlap) for team in assignment]
    
    def check_same_groups_constraint() -> bool:
        """same_groups 제약이 만족되는지 확인"""
        if not same_groups or not member_names:
            return True
        name_to_idx = {name: i for i, name in enumerate(member_names)}
        for group in same_groups:
            if len(group) < 2:
                continue
            indices = [name_to_idx.get(name) for name in group if name in name_to_idx]
            if len(indices) < 2:
                continue
            # 모든 인덱스가 같은 조에 있는지 확인
            group_indices = set()
            for idx in indices:
                for i, team in enumerate(assignment):
                    if idx in team:
                        group_indices.add(i)
                        break
            if len(group_indices) > 1:
                return False
        return True
    
    for _ in range(max_iterations):
        costs = get_costs()
        max_idx = max(range(n_groups), key=lambda i: costs[i])
        min_idx = min(range(n_groups), key=lambda i: costs[i])
        
        if costs[max_idx] - costs[min_idx] <= 1:
            break  # 비용이 이미 고르게 분배됨
        
        # 최대 비용 조와 최소 비용 조 간에 인원 교환 시도
        max_team = assignment[max_idx]
        min_team = assignment[min_idx]
        
        improved = False
        for i, person_a in enumerate(max_team):
            for j, person_b in enumerate(min_team):
                # 스왑 가능 여부 확인
                if person_a == person_b:
                    continue
                
                # 스왑 후 제약 조건 확인
                new_max_team = max_team[:i] + [person_b] + max_team[i+1:]
                new_min_team = min_team[:j] + [person_a] + min_team[j+1:]
                
                # 제약 조건 검증
                spec_max = group_specs[max_idx]
                spec_min = group_specs[min_idx]
                
                if not validate_team(new_max_team, spec_max, genders, veteran_flags):
                    continue
                if not validate_team(new_min_team, spec_min, genders, veteran_flags):
                    continue
                
                # same_groups 제약 확인
                assignment[max_idx] = new_max_team
                assignment[min_idx] = new_min_team
                if not check_same_groups_constraint():
                    # 제약 위반 시 원래대로 복구
                    assignment[max_idx] = max_team
                    assignment[min_idx] = min_team
                    continue
                
                # 비용 개선 확인
                old_max_cost = team_cost(max_team, overlap)
                old_min_cost = team_cost(min_team, overlap)
                new_max_cost = team_cost(new_max_team, overlap)
                new_min_cost = team_cost(new_min_team, overlap)
                
                old_diff = old_max_cost - old_min_cost
                new_diff = max(new_max_cost, new_min_cost) - min(new_max_cost, new_min_cost)
                
                # 총 비용 변화
                old_total = old_max_cost + old_min_cost
                new_total = new_max_cost + new_min_cost
                cost_increase = new_total - old_total
                
                # 하이브리드 목적 함수: 분산 감소 - α × 총비용 증가
                # balance_weight가 클수록 분산 감소를 더 중요시
                improvement = (old_diff - new_diff) - balance_weight * cost_increase
                
                if improvement > 0:
                    improved = True
                    break
                else:
                    # 개선되지 않으면 원래대로 복구
                    assignment[max_idx] = max_team
                    assignment[min_idx] = min_team
            
            if improved:
                break
        
        if not improved:
            break
    
    return tuple(tuple(team) for team in assignment)


def validate_team(
    team: List[int],
    spec: Dict,
    genders: Sequence[str],
    veteran_flags: Sequence[bool],
) -> bool:
    """팀이 제약 조건을 만족하는지 확인"""
    required_idx = spec.get("required")
    if required_idx is not None and required_idx not in team:
        return False
    
    min_male = spec.get("min_male", 0)
    max_male = spec.get("max_male", len(team))
    male_count = sum(1 for idx in team if genders[idx] == "M")
    if male_count < min_male or male_count > max_male:
        return False
    
    max_veterans = spec.get("max_veterans", len(veteran_flags))
    vet_count = sum(1 for idx in team if veteran_flags[idx])
    if vet_count > max_veterans:
        return False
    
    return True


def summarize(
    assignment: Sequence[Sequence[int]],
    members: Sequence[str],
    overlap: Sequence[Sequence[int]],
    genders: Sequence[str],
) -> None:
    print("=== 조 편성 결과 (DP 기반 최소 겹침) ===")
    total = 0
    for group_id, team in enumerate(assignment, start=1):
        names = [members[idx] for idx in team]
        cost = team_cost(team, overlap)
        total += cost
        male_count = sum(1 for idx in team if genders[idx] == "M")
        female_count = sum(1 for idx in team if genders[idx] == "F")
        print(
            f"조 {group_id} (인원 {len(team)}, 비용 {cost}, 남 {male_count}, 여 {female_count})"
        )
        print("  " + ", ".join(names))
    print(f"\n총 겹침 비용: {total}")


def main() -> None:
    args = parse_args()
    base_path = os.path.abspath(args.base_path)
    members_csv = os.path.join(base_path, args.members)

    all_members, gender_map, veterans_csv = load_member_data(members_csv)
    absent_set = set(args.absent or [])
    present_members = [name for name in all_members if name not in absent_set]
    if not present_members:
        raise ValueError("참석 인원이 없습니다. --absent 옵션을 확인하세요.")

    history = load_week_groups(base_path, args.weeks)
    overlap = build_overlap_matrix(present_members, history)

    if args.group_sizes:
        group_sizes = args.group_sizes
        if sum(group_sizes) != len(present_members):
            raise ValueError("group_sizes의 합이 전체 인원 수와 일치하지 않습니다.")
    else:
        group_sizes = compute_group_sizes(len(present_members), args.num_groups)

    if len(group_sizes) != args.num_groups:
        raise ValueError("group_sizes 길이가 num_groups와 일치해야 합니다.")

    if any(size <= 0 for size in group_sizes):
        raise ValueError("각 조의 인원 수는 1 이상이어야 합니다.")

    veterans = set(args.veterans or veterans_csv)
    veteran_flags = [name in veterans for name in present_members]
    genders = [gender_map.get(name, "U") for name in present_members]

    default_leaders = ["정민서", "이수진", "우화영", "신희영", "김성민", "이채연"]
    leaders_input = args.leaders or default_leaders
    filtered_leaders = [name for name in leaders_input if name in present_members]
    leader_indices: List[Optional[int]] = []
    for i in range(args.num_groups):
        leader_name = filtered_leaders[i] if i < len(filtered_leaders) else None
        leader_indices.append(
            present_members.index(leader_name) if leader_name is not None else None
        )

    total_males = sum(1 for g in genders if g == "M")
    if args.group_males:
        if len(args.group_males) != args.num_groups:
            raise ValueError("group_males의 길이가 조 수와 일치해야 합니다.")
        male_targets = args.group_males
        if sum(male_targets) != total_males:
            raise ValueError("group_males 의 합이 전체 남성 수와 일치해야 합니다.")
    else:
        base_male = total_males // args.num_groups
        extra_male = total_males % args.num_groups
        male_targets = [
            base_male + (1 if i < extra_male else 0) for i in range(args.num_groups)
        ]

    group_specs = []
    for idx, size in enumerate(group_sizes):
        group_specs.append(
            {
                "size": size,
                "required": leader_indices[idx],
                "min_male": male_targets[idx],
                "max_male": male_targets[idx],
                "max_veterans": args.max_veterans_per_group,
            }
        )

    same_groups = args.same_group if args.same_group else None
    total_cost, assignment = dp_min_overlap(
        group_specs, overlap, args.branch_limit, genders, veteran_flags, same_groups, present_members
    )
    
    # 하이브리드 방식: 비용 분산 최소화 + 총 비용 증가 최소화
    assignment = balance_costs(
        tuple(tuple(team) for team in assignment),
        overlap,
        group_specs,
        genders,
        veteran_flags,
        same_groups,
        present_members,
        args.balance_weight,
    )
    assignment = list(assignment)
    
    # 최종 총 비용 재계산
    total_cost = sum(team_cost(team, overlap) for team in assignment)
    
    summarize(assignment, present_members, overlap, genders)

    if args.output:
        result = {
            "week": None,
            "total_overlap_cost": total_cost,
            "groups": [
                {
                    "id": group_id,
                    "members": [present_members[idx] for idx in team],
                    "overlap_cost": team_cost(team, overlap),
                }
                for group_id, team in enumerate(assignment, start=1)
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n결과를 {args.output}에 저장했습니다.")


if __name__ == "__main__":
    main()

