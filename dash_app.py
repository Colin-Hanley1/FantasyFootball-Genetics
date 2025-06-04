import dash
from dash import dcc, html, dash_table, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import random
import math
import csv
import os
from collections import Counter
import numpy as np

# --- Configuration Constants ---
GAMES_IN_SEASON = 17
PICKS_PER_ROUND = 12
DEFAULT_CSV_FILENAME = "player_pool.csv"
GA_LOG_FILENAME = "ga_training_log.csv"

# --- Roster Configuration (Initial Default) ---
FLEX_ELIGIBILITY = {
    "W/R/T": ("WR", "RB", "TE"), "W/R": ("WR", "RB"), "R/T": ("RB", "TE"),
    "SUPERFLEX": ("QB", "WR", "RB", "TE"), "BN_SUPERFLEX": ("QB", "WR", "RB", "TE"),
    "BN_FLX": ("WR", "RB", "TE")
}
ROSTER_STRUCTURE = {
    "QB": 1, "RB": 2, "WR": 2, "TE": 1, "W/R/T": 1,
    "BN_SUPERFLEX": 1, "BN_FLX": 4
}

# --- Global Variables ---
INITIAL_PLAYER_POOL_DATA = []
MASTER_PLAYER_ID_TO_DATA = {}
GLOBALLY_DRAFTED_PLAYER_IDS = set()
USER_DRAFTED_PLAYERS_DATA = []
CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA = []
CURRENT_PLAYERS_BY_POSITION_FOR_GA = {}
USER_PLAYER_SLOT_ASSIGNMENTS = {}
POSITION_ORDER = []
TOTAL_ROSTER_SPOTS = 0
CURRENT_SCORING_MODE = "PPR"

# --- GA Parameters ---
POPULATION_SIZE = 100
N_GENERATIONS = 100
MUTATION_RATE = 0.20
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5
PENALTY_VIOLATION = 10000
STARTER_PPG_MULTIPLIER = 1.6
BENCH_ADP_PENALTY_SCALER = 0.3
STARTER_ADP_WEAKNESS_THRESHOLD = 2
EARLY_ROUND_ADP_BENCH_PENALTY = 0.5
BYE_WEEK_CONFLICT_PENALTY_FACTOR = 0.001
BACKUP_POSITION_PENALTY_SCALER = 0.4

# --- Data Loading and Setup ---
def load_player_pool_from_csv(filename=DEFAULT_CSV_FILENAME):
    player_pool_list = []
    expected_headers = ["ID", "Name", "Position", "PPRPoints", "PPRADP", "STDPoints", "STDADP", "ByeWeek"]
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames or not all(h in reader.fieldnames for h in expected_headers):
                missing = [h for h in expected_headers if h not in (reader.fieldnames or [])]
                print(f"Error: CSV '{filename}' missing headers: {missing}. Got: {reader.fieldnames}")
                return []
            for row_num, row in enumerate(reader, 1):
                try:
                    player_id = int(row["ID"])
                    name_in_csv = row["Name"]
                    pos_from_col = row["Position"].upper()
                    ppr_points_raw = float(row["PPRPoints"])
                    std_points_raw = float(row["STDPoints"])
                    try: ppr_adp_raw = int(float(row["PPRADP"]))
                    except ValueError: ppr_adp_raw = 999 ; print(f"Warning: Row {row_num}, Player ID {player_id}: Invalid PPRADP '{row['PPRADP']}', using {ppr_adp_raw}.")
                    try: std_adp_raw = int(float(row["STDADP"]))
                    except ValueError: std_adp_raw = 999 ; print(f"Warning: Row {row_num}, Player ID {player_id}: Invalid STDADP '{row['STDADP']}', using {std_adp_raw}.")
                    try: bye_week = int(row["ByeWeek"]) if row["ByeWeek"] and row["ByeWeek"].strip() else 0
                    except ValueError: bye_week = 0; print(f"Warning: Row {row_num}, Player ID {player_id}: Invalid ByeWeek '{row['ByeWeek']}', using 0.")
                    
                    active_points_val = ppr_points_raw if CURRENT_SCORING_MODE == "PPR" else std_points_raw
                    active_adp_val = ppr_adp_raw if CURRENT_SCORING_MODE == "PPR" else std_adp_raw
                    current_ppg = active_points_val / GAMES_IN_SEASON if GAMES_IN_SEASON > 0 else 0
                    current_calculated_round = max(1, math.ceil(active_adp_val / PICKS_PER_ROUND)) if PICKS_PER_ROUND > 0 else 1
                    
                    player_pool_list.append((player_id, name_in_csv, pos_from_col, current_ppg, current_calculated_round, bye_week, ppr_points_raw, ppr_adp_raw, std_points_raw, std_adp_raw))
                except (KeyError, ValueError, ZeroDivisionError) as e: print(f"Warning: Skipping row {row_num} in '{filename}': {e}. Row: {row}")
    except FileNotFoundError: print(f"Error: CSV file '{filename}' not found."); return []
    except Exception as e: print(f"Unexpected error reading '{filename}': {e}"); return []
    if not player_pool_list: print(f"Warning: No players loaded from '{filename}'.")
    return player_pool_list

def update_player_data_for_scoring_mode(selected_mode):
    global INITIAL_PLAYER_POOL_DATA, MASTER_PLAYER_ID_TO_DATA, USER_DRAFTED_PLAYERS_DATA, CURRENT_SCORING_MODE
    if not INITIAL_PLAYER_POOL_DATA: return
    CURRENT_SCORING_MODE = selected_mode
    new_initial_pool = []
    for p_tuple_full in INITIAL_PLAYER_POOL_DATA:
        pid, name, pos, _, _, bye, ppr_pts, ppr_adp, std_pts, std_adp = p_tuple_full[:10]
        active_points_val = ppr_pts if selected_mode == "PPR" else std_pts
        active_adp_val = ppr_adp if selected_mode == "PPR" else std_adp
        current_ppg_new = active_points_val / GAMES_IN_SEASON if GAMES_IN_SEASON > 0 else 0
        current_calc_round_new = max(1, math.ceil(active_adp_val / PICKS_PER_ROUND)) if PICKS_PER_ROUND > 0 else 1
        new_initial_pool.append((pid, name, pos, current_ppg_new, current_calc_round_new, bye, ppr_pts, ppr_adp, std_pts, std_adp))
    INITIAL_PLAYER_POOL_DATA = new_initial_pool
    MASTER_PLAYER_ID_TO_DATA = {p[0]: p for p in INITIAL_PLAYER_POOL_DATA}
    new_user_drafted_players = []
    for user_p_tuple in USER_DRAFTED_PLAYERS_DATA:
        pid, name, pos, _, _, bye, ppr_pts, ppr_adp, std_pts, std_adp = user_p_tuple[:10]
        active_points_val = ppr_pts if selected_mode == "PPR" else std_pts
        active_adp_val = ppr_adp if selected_mode == "PPR" else std_adp
        current_ppg_new = active_points_val / GAMES_IN_SEASON if GAMES_IN_SEASON > 0 else 0
        current_calc_round_new = max(1, math.ceil(active_adp_val / PICKS_PER_ROUND)) if PICKS_PER_ROUND > 0 else 1
        new_user_drafted_players.append((pid, name, pos, current_ppg_new, current_calc_round_new, bye, ppr_pts, ppr_adp, std_pts, std_adp))
    USER_DRAFTED_PLAYERS_DATA = new_user_drafted_players
    print(f"Player data updated for {selected_mode} scoring mode.")

def update_roster_derived_globals():
    global POSITION_ORDER, TOTAL_ROSTER_SPOTS, ROSTER_STRUCTURE
    temp_position_order, current_total_spots = [], 0
    starters = {k: v for k, v in ROSTER_STRUCTURE.items() if not k.startswith("BN_")}
    bench = {k: v for k, v in ROSTER_STRUCTURE.items() if k.startswith("BN_")}
    for slot_type, count in starters.items():
        if count > 0: temp_position_order.extend([slot_type] * count); current_total_spots += count
    for slot_type, count in bench.items():
        if count > 0: temp_position_order.extend([slot_type] * count); current_total_spots += count
    POSITION_ORDER, TOTAL_ROSTER_SPOTS = temp_position_order, current_total_spots
    print(f"Roster derived globals updated. Total Spots: {TOTAL_ROSTER_SPOTS}, Order: {len(POSITION_ORDER)} slots")
    if INITIAL_PLAYER_POOL_DATA:
        unique_initial_positions = set(p[2] for p in INITIAL_PLAYER_POOL_DATA)
        for slot_key, count in ROSTER_STRUCTURE.items():
            if count > 0 and slot_key not in FLEX_ELIGIBILITY and slot_key not in unique_initial_positions:
                print(f"Warning (Roster Update): Slot '{slot_key}' used but not base pos or FLEX.")
        for flex_key, eligible_list in FLEX_ELIGIBILITY.items():
            if ROSTER_STRUCTURE.get(flex_key, 0) > 0:
                for pos_flex in eligible_list:
                    if pos_flex not in unique_initial_positions:
                        print(f"Warning (Roster Update): Pos '{pos_flex}' for active FLEX '{flex_key}' not in player data.")

def initial_setup(csv_filename=DEFAULT_CSV_FILENAME):
    global INITIAL_PLAYER_POOL_DATA, MASTER_PLAYER_ID_TO_DATA, CURRENT_SCORING_MODE
    if GAMES_IN_SEASON <= 0: print("Critical Error: GAMES_IN_SEASON positive. Exiting."); exit()
    if PICKS_PER_ROUND <= 0: print("Critical Error: PICKS_PER_ROUND must be positive. Exiting."); exit()
    INITIAL_PLAYER_POOL_DATA = load_player_pool_from_csv(csv_filename)
    if not INITIAL_PLAYER_POOL_DATA: print("Critical Error: Initial player pool is empty. Exiting."); exit()
    update_player_data_for_scoring_mode(CURRENT_SCORING_MODE) 
    update_roster_derived_globals()
    print(f"Initial data loaded. Default scoring mode: {CURRENT_SCORING_MODE}.")

def prepare_for_ga_run():
    global CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA, CURRENT_PLAYERS_BY_POSITION_FOR_GA, USER_PLAYER_SLOT_ASSIGNMENTS
    CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA = [p for p in INITIAL_PLAYER_POOL_DATA if p[0] not in GLOBALLY_DRAFTED_PLAYER_IDS]
    CURRENT_PLAYERS_BY_POSITION_FOR_GA = {}
    all_avail_positions = set(p[2] for p in CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA)
    for pos_key in all_avail_positions:
        CURRENT_PLAYERS_BY_POSITION_FOR_GA[pos_key] = sorted([p for p in CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA if p[2] == pos_key], key=lambda x: (x[4], -x[3]))
    new_user_player_slot_assignments = {}
    available_slots_indices = list(range(TOTAL_ROSTER_SPOTS))
    sorted_user_players = sorted(USER_DRAFTED_PLAYERS_DATA, key=lambda x: x[4])
    processed_player_ids_for_assignment = set()
    for player_data_tuple in sorted_user_players:
        player_id, _, actual_player_pos, _, _, _, _, _, _, _ = player_data_tuple
        if player_id in processed_player_ids_for_assignment: continue
        assigned_this_player = False
        for slot_idx in sorted(list(available_slots_indices)):
            slot_type = POSITION_ORDER[slot_idx]
            if not slot_type.startswith("BN_") and slot_type == actual_player_pos:
                new_user_player_slot_assignments[player_id] = slot_idx; available_slots_indices.remove(slot_idx); processed_player_ids_for_assignment.add(player_id); assigned_this_player = True; break
        if assigned_this_player: continue
        for slot_idx in sorted(list(available_slots_indices)):
            slot_type = POSITION_ORDER[slot_idx]
            if not slot_type.startswith("BN_") and slot_type in FLEX_ELIGIBILITY and actual_player_pos in FLEX_ELIGIBILITY[slot_type]:
                new_user_player_slot_assignments[player_id] = slot_idx; available_slots_indices.remove(slot_idx); processed_player_ids_for_assignment.add(player_id); assigned_this_player = True; break
        if assigned_this_player: continue
        for slot_idx in sorted(list(available_slots_indices)):
            slot_type = POSITION_ORDER[slot_idx]
            if slot_type.startswith("BN_") and slot_type == "BN_" + actual_player_pos:
                 new_user_player_slot_assignments[player_id] = slot_idx; available_slots_indices.remove(slot_idx); processed_player_ids_for_assignment.add(player_id); assigned_this_player = True; break
        if assigned_this_player: continue
        for slot_idx in sorted(list(available_slots_indices)):
            slot_type = POSITION_ORDER[slot_idx]
            if slot_type.startswith("BN_") and slot_type in FLEX_ELIGIBILITY and actual_player_pos in FLEX_ELIGIBILITY[slot_type]:
                new_user_player_slot_assignments[player_id] = slot_idx; available_slots_indices.remove(slot_idx); processed_player_ids_for_assignment.add(player_id); assigned_this_player = True; break
        if not assigned_this_player: print(f"UI Warning: Could not auto-assign {player_data_tuple[1]} ({actual_player_pos}). Will be surplus.")
    USER_PLAYER_SLOT_ASSIGNMENTS.clear(); USER_PLAYER_SLOT_ASSIGNMENTS.update(new_user_player_slot_assignments)

def get_player_data(player_id): return MASTER_PLAYER_ID_TO_DATA.get(player_id)
def get_player_round(player_id): p = get_player_data(player_id); return p[4] if p else -1
def get_slot_type_for_index(index): return POSITION_ORDER[index] if 0 <= index < len(POSITION_ORDER) else None

def get_eligible_players_for_slot_type_for_ga(slot_type_value):
    eligible_players, processed_ids = [], set()
    if not CURRENT_PLAYERS_BY_POSITION_FOR_GA and CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA:
        all_avail_positions = set(p[2] for p in CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA)
        for pos_key in all_avail_positions:
            CURRENT_PLAYERS_BY_POSITION_FOR_GA[pos_key] = sorted([p for p in CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA if p[2] == pos_key], key=lambda x: (x[4], -x[3]))
    if slot_type_value in FLEX_ELIGIBILITY:
        for actual_pos in FLEX_ELIGIBILITY[slot_type_value]:
            for player in CURRENT_PLAYERS_BY_POSITION_FOR_GA.get(actual_pos, []):
                if player[0] not in processed_ids: eligible_players.append(player); processed_ids.add(player[0])
    elif slot_type_value in CURRENT_PLAYERS_BY_POSITION_FOR_GA:
        for player in CURRENT_PLAYERS_BY_POSITION_FOR_GA[slot_type_value]:
            if player[0] not in processed_ids: eligible_players.append(player); processed_ids.add(player[0])
    return eligible_players

def normalize_name_for_search(name_str):
    if not name_str: return ""
    cleaned_name = name_str.lower().replace('.', '').replace("'", "").strip()
    words = cleaned_name.split()
    if not words: return ""
    if len(words) == 1: return words[0]
    return words[0][0] + "".join(words[1:])

def parse_csv_name_to_fi_ln(csv_formatted_name):
    try:
        name_part = csv_formatted_name.split('_', 1)[-1] 
        if not name_part: return "", ""
        first_initials, last_name = "", ""
        i = 0
        while i < len(name_part) and name_part[i].isupper(): i += 1
        first_initials = name_part[:i]
        last_name = name_part[i:]
        if len(first_initials) > 2 and not last_name: last_name, first_initials = first_initials, ""
        return first_initials.lower(), last_name.lower()
    except Exception: return "", ""

def find_player_flexible(query_str):
    user_query_normalized_filn = normalize_name_for_search(query_str)
    user_query_simple_lower = query_str.lower().replace('.', '').replace("'", "").strip()
    POSSIBLE_QUERY_POSITIONS = {"qb", "wr", "rb", "te", "k", "def", "dst"}
    query_pos_part = None
    name_only_query_words_for_ln_match = [] # For lastname matching specifically
    
    temp_query_words = user_query_simple_lower.split()
    for word in temp_query_words:
        if word in POSSIBLE_QUERY_POSITIONS: query_pos_part = word
        else: name_only_query_words_for_ln_match.append(word)
    potential_query_lastname_norm = normalize_name_for_search(" ".join(name_only_query_words_for_ln_match[-1:])) if name_only_query_words_for_ln_match else ""


    exact_csv_matches, filn_format_matches, lastname_matches, strong_partial_matches, partial_matches = [], [], [], [], []

    for p_data in INITIAL_PLAYER_POOL_DATA:
        pid, csv_name, csv_pos_col, _, _, _, _, _, _, _ = p_data
        csv_name_lower, csv_pos_col_lower = csv_name.lower(), csv_pos_col.lower()

        if user_query_simple_lower == csv_name_lower: exact_csv_matches.append(p_data); continue
        
        fi_csv, ln_csv = parse_csv_name_to_fi_ln(csv_name)
        csv_name_normalized_filn = fi_csv + ln_csv

        if user_query_normalized_filn == csv_name_normalized_filn:
            if query_pos_part and query_pos_part == csv_pos_col_lower: filn_format_matches.insert(0,p_data)
            elif not query_pos_part: filn_format_matches.append(p_data)
            else: partial_matches.append(p_data)
            continue
            
        if ln_csv and user_query_normalized_filn == ln_csv: # User query was single word, normalized, matches ln_csv
            if query_pos_part and query_pos_part == csv_pos_col_lower: lastname_matches.insert(0,p_data)
            elif not query_pos_part: lastname_matches.append(p_data)
            else: partial_matches.append(p_data)
            continue
            
        if ln_csv and potential_query_lastname_norm and len(potential_query_lastname_norm) >= 3 and ln_csv.startswith(potential_query_lastname_norm):
            if query_pos_part and query_pos_part == csv_pos_col_lower: strong_partial_matches.append(p_data)
            elif not query_pos_part: partial_matches.append(p_data)
            continue

        csv_name_part_after_underscore = csv_name_lower.split('_',1)[-1]
        if len(user_query_simple_lower) >= 3 and user_query_simple_lower in csv_name_part_after_underscore:
            partial_matches.append(p_data)
            continue
            
    final_results, seen_ids = [], set()
    for p_list in [exact_csv_matches, filn_format_matches, lastname_matches, strong_partial_matches, partial_matches]:
        for p_item in p_list:
            if p_item[0] not in seen_ids: final_results.append(p_item); seen_ids.add(p_item[0])
    return final_results

# --- GA Core Functions (create_individual, calculate_fitness, repair_lineup, etc. - Unchanged) ---
def create_individual(): # Unchanged
    individual_ids, used_player_ids = [None] * TOTAL_ROSTER_SPOTS, set()
    for pid, slot_idx in USER_PLAYER_SLOT_ASSIGNMENTS.items():
        if 0 <= slot_idx < TOTAL_ROSTER_SPOTS: individual_ids[slot_idx] = pid; used_player_ids.add(pid)
    for i in range(TOTAL_ROSTER_SPOTS):
        if individual_ids[i] is not None: continue
        slot_type = POSITION_ORDER[i]
        candidates = [p for p in get_eligible_players_for_slot_type_for_ga(slot_type) if p[0] not in used_player_ids]
        if candidates: chosen_p = random.choice(candidates); individual_ids[i] = chosen_p[0]; used_player_ids.add(chosen_p[0])
        else: individual_ids[i] = -99
    for i in range(len(individual_ids)):
        if individual_ids[i] is None and not any(s_idx == i for s_idx in USER_PLAYER_SLOT_ASSIGNMENTS.values()):
            individual_ids[i] = -99
    return individual_ids
def create_initial_population(): return [create_individual() for _ in range(POPULATION_SIZE)] # Unchanged
def calculate_fitness(individual_ids, curr_round): # Unchanged
    if not individual_ids or len(individual_ids) != TOTAL_ROSTER_SPOTS: return -float('inf'), 0, set(), []
    if any(pid is None or (not isinstance(pid, int)) or (pid <= 0 and pid != -99) for pid in individual_ids):
        inv_spots = sum(1 for pid in individual_ids if pid != -99 and (pid is None or not isinstance(pid, int) or pid <= 0))
        if inv_spots > 0: return -PENALTY_VIOLATION * (inv_spots + 20), 0, set(), []
    lineup_player_objects = [None] * TOTAL_ROSTER_SPOTS
    for i, pid in enumerate(individual_ids):
        slot_type, is_starting_slot = POSITION_ORDER[i], not POSITION_ORDER[i].startswith("BN_")
        if pid == -99:
            if is_starting_slot: return -PENALTY_VIOLATION * 75, 0, set(), []
        else:
            p_data = get_player_data(pid)
            if p_data is None:
                if is_starting_slot: return -PENALTY_VIOLATION * 65, 0, set(), []
                else: return -PENALTY_VIOLATION * 30, 0, set(), []
            lineup_player_objects[i] = p_data
    valid_player_data_for_checks = [p_obj for p_obj in lineup_player_objects if p_obj is not None]
    for i, p_data_current in enumerate(lineup_player_objects):
        if p_data_current is None: continue
        slot_type, actual_pos = POSITION_ORDER[i], p_data_current[2]
        is_valid = (slot_type == actual_pos) or (slot_type in FLEX_ELIGIBILITY and actual_pos in FLEX_ELIGIBILITY[slot_type])
        if not is_valid: return -PENALTY_VIOLATION * 10, 0, set(), valid_player_data_for_checks
    player_ids_for_dup_check = [p[0] for p in valid_player_data_for_checks]
    if len(set(player_ids_for_dup_check)) != len(player_ids_for_dup_check):
        return -PENALTY_VIOLATION * 5, 0, set(), valid_player_data_for_checks
    raw_ppg_sum, fitness_ppg_component = 0, 0
    for i, p_data_calc in enumerate(lineup_player_objects):
        if p_data_calc is None: continue
        ppg = p_data_calc[3]; raw_ppg_sum += ppg
        fitness_ppg_component += (ppg * STARTER_PPG_MULTIPLIER) if not POSITION_ORDER[i].startswith("BN_") else ppg
    fitness_score = fitness_ppg_component
    bench_adp_mismanagement_penalty = 0
    num_s_filled, sum_s_adp_rnds, min_s_adp_rnd_val = 0, 0, float('inf')
    for i_check, p_data_check in enumerate(lineup_player_objects):
        if p_data_check is None: continue
        if not POSITION_ORDER[i_check].startswith("BN_"):
            num_s_filled += 1; sum_s_adp_rnds += p_data_check[4]; min_s_adp_rnd_val = min(min_s_adp_rnd_val, p_data_check[4])
    avg_s_adp_rnd = sum_s_adp_rnds / num_s_filled if num_s_filled > 0 else float('inf')
    for i_bench, p_data_bench in enumerate(lineup_player_objects):
        if p_data_bench is None: continue
        if POSITION_ORDER[i_bench].startswith("BN_"):
            bench_p_adp_rnd = p_data_bench[4]
            if bench_p_adp_rnd < (avg_s_adp_rnd - STARTER_ADP_WEAKNESS_THRESHOLD):
                bench_adp_mismanagement_penalty += (PENALTY_VIOLATION * BENCH_ADP_PENALTY_SCALER * (avg_s_adp_rnd - bench_p_adp_rnd) / 10.0)
            if bench_p_adp_rnd <= 3 and min_s_adp_rnd_val > (bench_p_adp_rnd + STARTER_ADP_WEAKNESS_THRESHOLD):
                bench_adp_mismanagement_penalty += (PENALTY_VIOLATION * EARLY_ROUND_ADP_BENCH_PENALTY * (4 - bench_p_adp_rnd) / 5.0)
    fitness_score -= bench_adp_mismanagement_penalty
    bye_weeks_on_roster = [p[5] for p in valid_player_data_for_checks if p[5] is not None and 0 < p[5] < 20]
    bye_week_counts = Counter(bye_weeks_on_roster)
    num_bye_week_conflicts = sum(count - 1 for count in bye_week_counts.values() if count >= 2)
    if num_bye_week_conflicts > 0:
        fitness_score -= (PENALTY_VIOLATION * BYE_WEEK_CONFLICT_PENALTY_FACTOR * num_bye_week_conflicts)
    missing_backup_penalty_points = 0
    core_starter_positions_defined = set()
    for slot, count in ROSTER_STRUCTURE.items():
        if count > 0 and not slot.startswith("BN_") and slot not in FLEX_ELIGIBILITY: core_starter_positions_defined.add(slot)
    if core_starter_positions_defined:
        bench_player_actual_positions = Counter()
        for i, p_data in enumerate(lineup_player_objects):
            if p_data is not None and POSITION_ORDER[i].startswith("BN_"): bench_player_actual_positions[p_data[2]] += 1
        for core_pos in core_starter_positions_defined:
            is_core_pos_started = any(p_obj and not POSITION_ORDER[i].startswith("BN_") and POSITION_ORDER[i] == core_pos and p_obj[2] == core_pos for i, p_obj in enumerate(lineup_player_objects))
            if is_core_pos_started and bench_player_actual_positions[core_pos] == 0: missing_backup_penalty_points += 1
        if missing_backup_penalty_points > 0: fitness_score -= (PENALTY_VIOLATION * BACKUP_POSITION_PENALTY_SCALER * missing_backup_penalty_points)
    player_adp_rounds_in_lineup = [p[4] for p in valid_player_data_for_checks]
    adp_round_counts = Counter(player_adp_rounds_in_lineup)
    num_future_round_stacking_violations = sum(count - 1 for adp_r, count in adp_round_counts.items() if count > 1 and adp_r >= curr_round)
    if num_future_round_stacking_violations > 0:
        fitness_score -= (PENALTY_VIOLATION * num_future_round_stacking_violations * 1.5)
    return fitness_score, raw_ppg_sum, set(player_adp_rounds_in_lineup), valid_player_data_for_checks
def repair_lineup(lineup_ids_to_repair): # Unchanged
    repaired_ids = list(lineup_ids_to_repair); # ... (rest of function as provided previously) ...
    if not repaired_ids or len(repaired_ids) != TOTAL_ROSTER_SPOTS: return create_individual()
    user_assigned_slots = set(USER_PLAYER_SLOT_ASSIGNMENTS.values())
    for pid, slot_idx in USER_PLAYER_SLOT_ASSIGNMENTS.items():
        if 0 <= slot_idx < TOTAL_ROSTER_SPOTS:
            if repaired_ids[slot_idx] != pid:
                for j_idx, j_pid in enumerate(repaired_ids):
                    if j_pid == pid and j_idx != slot_idx: repaired_ids[j_idx] = -99
                repaired_ids[slot_idx] = pid
    ga_slots_to_repair = [i for i in range(TOTAL_ROSTER_SPOTS) if i not in user_assigned_slots]
    ga_slots_to_repair.sort(key=lambda i: POSITION_ORDER[i].startswith("BN_"))
    for _ in range(TOTAL_ROSTER_SPOTS + 5):
        current_valid_pids_in_repair = {pid for pid in repaired_ids if pid is not None and pid > 0 and pid != -99}
        id_counts_in_repair = Counter(current_valid_pids_in_repair)
        made_change = False
        for i in ga_slots_to_repair:
            current_pid_at_slot = repaired_ids[i]
            needs_replace = (current_pid_at_slot == -99 or current_pid_at_slot is None or \
                             (current_pid_at_slot <= 0 and current_pid_at_slot != -99) or \
                             (current_pid_at_slot != -99 and id_counts_in_repair.get(current_pid_at_slot, 0) > 1))
            if needs_replace:
                slot_type_fill = POSITION_ORDER[i]
                ids_to_avoid_new_pick = {repaired_ids[j] for j in range(TOTAL_ROSTER_SPOTS) if j != i and repaired_ids[j] not in [-99, None] and repaired_ids[j] > 0}
                options = [p for p in get_eligible_players_for_slot_type_for_ga(slot_type_fill) if p[0] not in ids_to_avoid_new_pick]
                if options:
                    new_player = random.choice(options)
                    if current_pid_at_slot != -99 and current_pid_at_slot is not None and current_pid_at_slot > 0:
                         id_counts_in_repair[current_pid_at_slot] = max(0, id_counts_in_repair.get(current_pid_at_slot,0)-1)
                    repaired_ids[i] = new_player[0]
                    id_counts_in_repair[new_player[0]] = id_counts_in_repair.get(new_player[0], 0) + 1
                    made_change = True
                else:
                    if repaired_ids[i] != -99:
                        if current_pid_at_slot != -99 and current_pid_at_slot is not None and current_pid_at_slot > 0:
                             id_counts_in_repair[current_pid_at_slot] = max(0, id_counts_in_repair.get(current_pid_at_slot,0)-1)
                        repaired_ids[i] = -99; made_change = True
        final_valid_pids_check = [pid for pid in repaired_ids if pid not in [-99, None] and pid > 0]
        if not made_change and len(set(final_valid_pids_check)) == len(final_valid_pids_check): break
    for i in ga_slots_to_repair:
        if repaired_ids[i] is None: repaired_ids[i] = -99
    return repaired_ids
def tournament_selection(population, fitness_scores_only): # Unchanged
    if not population: return create_individual()
    actual_tourn_size = min(TOURNAMENT_SIZE, len(population))
    if actual_tourn_size <= 0: return random.choice(population) if population else create_individual()
    selected_indices = random.sample(range(len(population)), actual_tourn_size)
    tournament_contenders = [(population[i], fitness_scores_only[i]) for i in selected_indices]
    winner = max(tournament_contenders, key=lambda x: x[1])
    return winner[0]
def crossover(parent1_ids, parent2_ids): # Unchanged
    child1, child2 = list(parent1_ids), list(parent2_ids)
    if random.random() < CROSSOVER_RATE and TOTAL_ROSTER_SPOTS > 1:
        pt = random.randint(1, TOTAL_ROSTER_SPOTS - 1) if TOTAL_ROSTER_SPOTS > 1 else 0
        if pt > 0 :
            temp_c1 = parent1_ids[:pt] + parent2_ids[pt:]
            temp_c2 = parent2_ids[:pt] + parent1_ids[pt:]
            for player_id, assigned_slot_index in USER_PLAYER_SLOT_ASSIGNMENTS.items():
                if 0 <= assigned_slot_index < TOTAL_ROSTER_SPOTS:
                    if temp_c1[assigned_slot_index] != player_id:
                        for i in range(TOTAL_ROSTER_SPOTS):
                            if i != assigned_slot_index and temp_c1[i] == player_id: temp_c1[i] = -99
                        temp_c1[assigned_slot_index] = player_id
                    if temp_c2[assigned_slot_index] != player_id:
                        for i in range(TOTAL_ROSTER_SPOTS):
                            if i != assigned_slot_index and temp_c2[i] == player_id: temp_c2[i] = -99
                        temp_c2[assigned_slot_index] = player_id
            child1, child2 = temp_c1, temp_c2
    return repair_lineup(child1), repair_lineup(child2)
def mutate(individual_ids): # Unchanged
    mutated = list(individual_ids)
    user_assigned_indices = set(USER_PLAYER_SLOT_ASSIGNMENTS.values())
    if random.random() < MUTATION_RATE:
        if not mutated or len(mutated) != TOTAL_ROSTER_SPOTS: return repair_lineup(mutated)
        ga_controlled_indices = [i for i in range(TOTAL_ROSTER_SPOTS) if i not in user_assigned_indices]
        if not ga_controlled_indices: return repair_lineup(mutated)
        mutation_idx = random.choice(ga_controlled_indices)
        original_pid_at_idx = mutated[mutation_idx]
        slot_type_mut = POSITION_ORDER[mutation_idx]
        eligible_pool_mut = get_eligible_players_for_slot_type_for_ga(slot_type_mut)
        if not eligible_pool_mut: return repair_lineup(mutated)
        other_pids_in_mutated = {pid for idx, pid in enumerate(mutated) if idx != mutation_idx and pid not in [-99, None] and pid > 0}
        options = [p for p in eligible_pool_mut if p[0] != original_pid_at_idx and p[0] not in other_pids_in_mutated]
        if not options: options = [p for p in eligible_pool_mut if p[0] not in other_pids_in_mutated]
        if not options: options = eligible_pool_mut
        if options: mutated[mutation_idx] = random.choice(options)[0]
    return repair_lineup(mutated)
def suggest_next_best_pick(best_ga_lineup_ids, user_drafted_player_ids_set): # Unchanged
    ga_suggested_additions_details = []
    for slot_index, player_id in enumerate(best_ga_lineup_ids):
        if player_id not in user_drafted_player_ids_set and player_id > 0:
            player_data = get_player_data(player_id)
            if player_data:
                is_starter_in_ga_lineup = not POSITION_ORDER[slot_index].startswith("BN_")
                ga_suggested_additions_details.append({
                    "data": player_data, "adp_round": player_data[4], "ppg": player_data[3], "is_starter": is_starter_in_ga_lineup
                })
    if not ga_suggested_additions_details: return None
    starters_ga_would_add = [p for p in ga_suggested_additions_details if p["is_starter"]]
    bench_ga_would_add = [p for p in ga_suggested_additions_details if not p["is_starter"]]
    if starters_ga_would_add:
        starters_ga_would_add.sort(key=lambda p: (p["adp_round"], -p["ppg"]))
        return starters_ga_would_add[0]["data"]
    if bench_ga_would_add:
        bench_ga_would_add.sort(key=lambda p: (p["adp_round"], -p["ppg"]))
        return bench_ga_would_add[0]["data"]
    return None
def genetic_algorithm_adp_lineup(curr_round): # Unchanged
    ui_messages, open_ga_slots = [], TOTAL_ROSTER_SPOTS - len(USER_PLAYER_SLOT_ASSIGNMENTS)
    num_s_defined = sum(1 for s in POSITION_ORDER if not s.startswith("BN_"))
    num_s_user = sum(1 for si in USER_PLAYER_SLOT_ASSIGNMENTS.values() if not POSITION_ORDER[si].startswith("BN_"))
    if open_ga_slots <= 0: ui_messages.append(dbc.Alert("Roster full.", color="info"))
    elif num_s_user >= num_s_defined and open_ga_slots > 0: ui_messages.append(dbc.Alert("Starters filled. GA optimizing bench.", color="info"))
    if open_ga_slots <= 0 or (not CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA and open_ga_slots > 0) or \
       not (pop := create_initial_population()) or not all(ind and len(ind) == TOTAL_ROSTER_SPOTS for ind in pop):
        if open_ga_slots > 0 and not CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA: ui_messages.append(dbc.Alert("No GA: No available players.", color="warning"))
        elif open_ga_slots > 0 : ui_messages.append(dbc.Alert("No GA: Population error.", color="danger"))
        ids = [-99]*TOTAL_ROSTER_SPOTS; [(ids.__setitem__(si, pid)) for pid, si in USER_PLAYER_SLOT_ASSIGNMENTS.items()]
        fit, ppg, _, _ = calculate_fitness(ids, curr_round)
        ppg_disp = f"{ppg:.2f}" if fit > -PENALTY_VIOLATION*70 else "N/A"
        ui_messages.append(dbc.Alert(f"Current Team: Fit {fit:.2f}, PPG {ppg_disp}", color="primary" if open_ga_slots <=0 else "info"))
        return ids, fit, ui_messages
    population = pop
    best_ids, (best_fit, _, _, _) = population[0], calculate_fitness(population[0], curr_round)
    for gen in range(N_GENERATIONS):
        fit_results = [calculate_fitness(ind, curr_round) for ind in population]
        fit_scores = [res[0] for res in fit_results]
        best_idx_gen = np.argmax(fit_scores)
        if fit_scores[best_idx_gen] > best_fit: best_fit, best_ids = fit_scores[best_idx_gen], list(population[best_idx_gen])
        next_pop = []
        elites = sorted([(population[i], fit_scores[i]) for i in range(len(population)) if fit_scores[i] > -PENALTY_VIOLATION * 70], key=lambda x: x[1], reverse=True)
        next_pop.extend([e[0] for i, e in enumerate(elites) if i < max(1, int(0.05*POPULATION_SIZE)) and (not next_pop or not any(Counter(ex)==Counter(e[0]) for ex in next_pop))])
        while len(next_pop) < POPULATION_SIZE:
            p1, p2 = tournament_selection(population, fit_scores), tournament_selection(population, fit_scores)
            c1, c2 = crossover(p1, p2); next_pop.append(mutate(c1))
            if len(next_pop) < POPULATION_SIZE: next_pop.append(mutate(c2))
        population = next_pop[:POPULATION_SIZE]
    ui_messages.append(dbc.Alert("--- GA Finished ---", color="success"))
    if best_ids and best_fit > -PENALTY_VIOLATION*70:
        final_fit, final_ppg, _, _ = calculate_fitness(best_ids, curr_round)
        ui_messages.extend([dbc.Alert(f"🏆 Best Lineup (Fit: {final_fit:.2f})", color="success", className="fw-bold"), dbc.Alert(f"Proj. True PPG: {final_ppg:.2f}", color="info")])
        return best_ids, best_fit, ui_messages
    else:
        ui_messages.append(dbc.Alert("⚠️ No valid/improved GA solution.", color="warning"))
        ids = [-99]*TOTAL_ROSTER_SPOTS; [(ids.__setitem__(si, pid)) for pid, si in USER_PLAYER_SLOT_ASSIGNMENTS.items()]
        fit, ppg, _, _ = calculate_fitness(ids, curr_round)
        ppg_disp = f"{ppg:.2f}" if fit > -PENALTY_VIOLATION*70 else "N/A"
        ui_messages.append(dbc.Alert(f"Current Team: Fit {fit:.2f}, PPG {ppg_disp}", color="info"))
        return ids, fit, ui_messages

# --- Dash App ---
try:
    initial_setup(DEFAULT_CSV_FILENAME)
    INITIAL_SETUP_SUCCESS = True
except SystemExit: INITIAL_SETUP_SUCCESS = False

DBC_THEME = dbc.themes.FLATLY
app = dash.Dash(__name__, external_stylesheets=[DBC_THEME], suppress_callback_exceptions=True)
server = app.server

if not INITIAL_SETUP_SUCCESS:
    app.layout = dbc.Container([dbc.Alert("App Init Failed.", color="danger", className="mt-5")], fluid=True)
else:
    _INITIAL_ROSTER_KEYS = list(ROSTER_STRUCTURE.keys())
    def get_roster_structure_info(): # Unchanged
        items = [dbc.ListGroupItem(html.H5("Current Roster Settings", className="mb-0"), className="bg-light")]
        for slot, count in ROSTER_STRUCTURE.items():
            elig_str = f" (Eligible: {', '.join(FLEX_ELIGIBILITY[slot])})" if slot in FLEX_ELIGIBILITY else ""
            items.append(dbc.ListGroupItem(f"{slot}: {count}{elig_str}"))
        items.append(dbc.ListGroupItem(f"Starter PPGs x{STARTER_PPG_MULTIPLIER} in fitness.", color="info"))
        items.append(dbc.ListGroupItem(f"Active Scoring Mode: {CURRENT_SCORING_MODE}", className="text-primary fw-bold"))
        return dbc.ListGroup(items, className="mb-3")
    def create_roster_input_rows(current_roster_config): # Unchanged
        rows = []
        for pos_key in _INITIAL_ROSTER_KEYS:
            rows.append(dbc.Row([
                dbc.Col(dbc.Label(pos_key, html_for=f"roster-input-{pos_key.replace('/', '-')}", className="text-end"), width=5),
                dbc.Col(dbc.Input(id=f"roster-input-{pos_key.replace('/', '-')}", type="number", value=current_roster_config.get(pos_key, 0), min=0, step=1, size="sm"), width=7)
            ], className="mb-2 align-items-center"))
        return rows
    def format_team_display_data(): # Unchanged
        prepare_for_ga_run()
        headers = [{"id": "slot", "name": "Slot"}, {"id": "pos", "name": "Pos"},
                   {"id": "player", "name": "Player (Actual Pos)"}, {"id": "ppg", "name": "PPG"},
                   {"id": "adp_rd", "name": "ADP Rd"}, {"id": "bye", "name": "Bye"}]
        starters_d, bench_d = [], []
        starters_ppg_sum, num_s, total_s = 0, 0, sum(1 for s in POSITION_ORDER if not s.startswith('BN_'))
        num_b, total_b = 0, sum(1 for s in POSITION_ORDER if s.startswith('BN_'))
        for i, slot_type in enumerate(POSITION_ORDER):
            is_starter = not slot_type.startswith("BN_")
            pid_in_slot = next((pid for pid, s_idx in USER_PLAYER_SLOT_ASSIGNMENTS.items() if s_idx == i), None)
            p_data = get_player_data(pid_in_slot)
            if p_data:
                row_data = {"slot": i, "pos": slot_type, "player": f"{p_data[1]} ({p_data[2]})", 
                            "ppg": f"{p_data[3]:.2f}", "adp_rd": p_data[4], "bye": p_data[5] if p_data[5] > 0 else "-"}
                if is_starter: starters_d.append(row_data); starters_ppg_sum += p_data[3]; num_s +=1
                else: bench_d.append(row_data); num_b +=1
            else:
                row_data = {"slot": i, "pos": slot_type, "player": "-", "ppg": "-", "adp_rd": "-", "bye": "-"}
                if is_starter: starters_d.append(row_data)
                else: bench_d.append(row_data)
        surplus_elems = [dbc.ListGroupItem(f"{p[1]}({p[2]}) PPG:{p[3]:.2f} Rd:{p[4]} Bye:{p[5] if p[5] > 0 else '-'}")
                         for p in USER_DRAFTED_PLAYERS_DATA if p[0] not in USER_PLAYER_SLOT_ASSIGNMENTS]
        return headers, starters_d, starters_ppg_sum, num_s, total_s, headers, bench_d, num_b, total_b, surplus_elems
    def format_ga_results_display(best_lineup_ids, best_fitness, ga_messages_alerts, next_pick_suggestion_data=None): # Unchanged
        children = [html.H5("GA Suggestions", className="card-title")]
        if next_pick_suggestion_data:
            p_sugg = next_pick_suggestion_data
            suggestion_alert = dbc.Alert(
                [html.H5("🎯 Next Pick Suggestion:", className="alert-heading"),
                 html.P(f"Consider: {p_sugg[1]} ({p_sugg[2]})", className="mb-1"),
                 html.P(f"PPG: {p_sugg[3]:.2f} | ADP Rd: {p_sugg[4]} | Bye: {p_sugg[5] if p_sugg[5] > 0 else '-'}", style={'fontSize': '0.9rem'})],
                color="primary", className="mt-1 mb-3 shadow-sm")
            children.append(suggestion_alert)
        children.extend(ga_messages_alerts)
        if best_lineup_ids and best_fitness > -PENALTY_VIOLATION * 70 :
            lineup_details = [html.H6("📋 GA's Optimal Full Lineup:", className="mt-3")]
            items = []
            for i, pid_slot in enumerate(best_lineup_ids):
                p_data = get_player_data(pid_slot)
                slot_type_disp = get_slot_type_for_index(i)
                user_pick_str = " (Your Pick)" if pid_slot in USER_PLAYER_SLOT_ASSIGNMENTS and USER_PLAYER_SLOT_ASSIGNMENTS.get(pid_slot) == i else ""
                txt = ""
                if p_data: txt = f"S{i:02d}({slot_type_disp:<10}): {p_data[1]:<20}({p_data[2]:<2}) PPG:{p_data[3]:>5.2f} Bye:{p_data[5] if p_data[5] > 0 else '-'} Rd:{p_data[4]:>2} {user_pick_str}"
                elif pid_slot == -99: txt = f"S{i:02d}({slot_type_disp:<10}): -"
                else: txt = f"S{i:02d}({slot_type_disp:<10}): Invalid ID {pid_slot}"
                items.append(dbc.ListGroupItem(txt, style={'fontSize': '0.8rem', 'padding': '0.25rem 0.5rem'}))
            lineup_details.append(dbc.ListGroup(items, flush=True, className="mt-2"))
            children.extend(lineup_details)
            actual_ids_in_best = [pid for pid in best_lineup_ids if pid is not None and pid > 0 and pid != -99]
            if len(set(actual_ids_in_best)) != len(actual_ids_in_best):
                 children.append(dbc.Alert("WARNING: DUPLICATES IN BEST LINEUP.", color="danger", className="mt-2 fw-bold"))
        return children

    app.layout = dbc.Container([ # Unchanged
        dcc.Store(id='ui-update-trigger', data=0),
        dbc.Modal(
            [dbc.ModalHeader(dbc.ModalTitle("Confirm Restart Draft")),
             dbc.ModalBody("Are you sure you want to restart? All draft progress will be lost."),
             dbc.ModalFooter([dbc.Button("Cancel", id="restart-draft-cancel-btn", color="secondary", className="ms-auto"),
                              dbc.Button("Confirm Restart", id="restart-draft-confirm-btn", color="danger")])],
            id="restart-draft-modal", is_open=False, centered=True),
        dbc.Row(dbc.Col(html.H1("🏈 Live Fantasy Football Draft Assistant", className="text-center my-4"))),
        dbc.Row(dbc.Col(id='current-round-info', className="text-center mb-3 fw-bold fs-5")),
        dbc.Row(dbc.Col(id='action-messages-div')), 
        dbc.Row(dbc.Col(id='roster-update-messages-div')),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Scoring Mode", className="mb-0")),
                    dbc.CardBody(dcc.RadioItems(id='scoring-mode-selector',
                                                options=[{'label': 'PPR', 'value': 'PPR'}, {'label': 'Standard', 'value': 'STD'}],
                                                value=CURRENT_SCORING_MODE, labelStyle={'display': 'inline-block', 'marginRight': '20px'}, inputClassName="me-1"))
                ], className="mb-3"),
                dbc.Card([
                    dbc.CardHeader(html.H5("Customize Roster Structure", className="mb-0")),
                    dbc.CardBody(create_roster_input_rows(ROSTER_STRUCTURE) + [dbc.Button("Apply Roster Changes", id="apply-roster-changes-btn", color="success", outline=True, className="w-100 mt-3")])
                ], className="mb-3"),
                dbc.Card([
                    dbc.CardHeader(html.H4("Draft Actions", className="mb-0")),
                    dbc.CardBody([
                        dbc.Input(id='player-query-input', type='text', placeholder='Player Name or ID', className='mb-2'),
                        dbc.ButtonGroup([dbc.Button('Draft for Opponent', id='draft-opponent-btn', color='warning', outline=True, className='me-1'),
                                         dbc.Button('Draft for My Team', id='draft-my-team-btn', color='success', outline=True)], className="d-grid gap-2 d-md-flex mb-3"),
                        dbc.Input(id='undo-player-id-input', type='number', placeholder='Player ID to Undo', className='mb-2'),
                        dbc.Button('Undo Draft', id='undo-draft-btn', color='danger', outline=True, className="d-grid gap-2 mb-3"),
                        html.Hr(className="my-2"),
                        dbc.Button("Restart Entire Draft", id="restart-draft-open-modal-btn", color="danger", className="w-100")
                    ])
                ], className="mb-3"),
                dbc.Card([
                    dbc.CardHeader(html.H4("Available Players", className="mb-0")),
                    dbc.CardBody([
                        dcc.Dropdown(id='available-pos-filter-dd', options=[{'label': 'All', 'value': 'ALL'}] + [{'label': p, 'value': p} for p in sorted(list(set(pl[2] for pl in INITIAL_PLAYER_POOL_DATA)))], value='ALL', clearable=False, className="mb-2"),
                        html.Div(id='available-players-display', style={'maxHeight': '300px', 'overflowY': 'auto'})])
                ]),
                 dbc.Card(id='roster-structure-summary-display', className="mt-3", body=False)
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("My Team", className="mb-0")),
                    dbc.CardBody([
                        html.H5("Starters", className="mt-2"), html.Div(id='my-team-starters-table'), html.P(id='my-team-starters-summary', className="fw-bold mt-2 text-end"),
                        html.H5("Bench", className="mt-3"), html.Div(id='my-team-bench-table'), html.P(id='my-team-bench-summary', className="fw-bold mt-2 text-end"),
                        html.Div(id='my-team-surplus-players', className="mt-3")])
                ], className="mb-3"),
                dbc.Card([
                    dbc.CardHeader(html.H4("Algorithm Suggestions", className="mb-0")),
                    dbc.CardBody([dbc.Button('Run GA Suggestions 🚀', id='run-ga-btn', color='primary', className="mb-3 w-100"), dcc.Loading(id="loading-ga-spinner", type="default", children=[html.Div(id='ga-results-display')])])])
            ], md=5),
            dbc.Col([
                dbc.Card([dbc.CardHeader(html.H4("Globally Drafted Players", className="mb-0")), dbc.CardBody(html.Div(id='drafted-players-display', style={'maxHeight': 'calc(85vh - 120px)', 'overflowY': 'auto'}))])
            ], md=3)
        ], className="mb-5")
    ], fluid=True)

    _roster_input_outputs_for_callback = [Output(f"roster-input-{pos_key.replace('/', '-')}", "value") for pos_key in _INITIAL_ROSTER_KEYS]
    _roster_input_states_for_callback = [State(f"roster-input-{pos_key.replace('/', '-')}", "value") for pos_key in _INITIAL_ROSTER_KEYS]

    @app.callback(
        [Output('roster-update-messages-div', 'children'),
         Output('ui-update-trigger', 'data', allow_duplicate=True),
         Output('roster-structure-summary-display', 'children', allow_duplicate=True)] +
        _roster_input_outputs_for_callback,
        Input('apply-roster-changes-btn', 'n_clicks'),
        _roster_input_states_for_callback + [State('ui-update-trigger', 'data')],
        prevent_initial_call=True
    )
    def handle_apply_roster_changes(n_clicks, *all_input_and_state_values): # Unchanged
        global ROSTER_STRUCTURE, USER_PLAYER_SLOT_ASSIGNMENTS, CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA, CURRENT_PLAYERS_BY_POSITION_FOR_GA
        num_roster_inputs = len(_INITIAL_ROSTER_KEYS)
        new_roster_counts_from_ui = all_input_and_state_values[:num_roster_inputs]
        current_trigger_val = all_input_and_state_values[num_roster_inputs]
        alerts, temp_new_roster, valid_update = [], {}, True
        previous_valid_roster_structure = ROSTER_STRUCTURE.copy() if isinstance(ROSTER_STRUCTURE, dict) else {}
        for i, key in enumerate(_INITIAL_ROSTER_KEYS):
            count_val = new_roster_counts_from_ui[i]
            if count_val is None: count_val = 0 
            if not isinstance(count_val, int) or count_val < 0:
                alerts.append(dbc.Alert(f"Invalid count for {key}: '{count_val}'. Must be non-negative integer.", color="danger", duration=4000))
                valid_update = False; break
            temp_new_roster[key] = count_val
        if not valid_update:
            reverted_input_values = [previous_valid_roster_structure.get(key, 0) for key in _INITIAL_ROSTER_KEYS]
            return [alerts, current_trigger_val, get_roster_structure_info()] + reverted_input_values
        if sum(temp_new_roster.values()) == 0 :
            alerts.append(dbc.Alert("Roster cannot have zero total spots.", color="danger", duration=5000))
            reverted_input_values = [previous_valid_roster_structure.get(key, 0) for key in _INITIAL_ROSTER_KEYS]
            return [alerts, current_trigger_val, get_roster_structure_info()] + reverted_input_values
        ROSTER_STRUCTURE = temp_new_roster
        update_roster_derived_globals()
        USER_PLAYER_SLOT_ASSIGNMENTS = {}; CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA, CURRENT_PLAYERS_BY_POSITION_FOR_GA = [], {}
        alerts.append(dbc.Alert("Roster structure updated! Team assignments re-evaluated.", color="success", duration=4000))
        current_input_values = [ROSTER_STRUCTURE.get(key, 0) for key in _INITIAL_ROSTER_KEYS]
        return [alerts, current_trigger_val + 1, get_roster_structure_info()] + current_input_values

    @app.callback( # MODIFIED for clearing inputs and Enter key
        [Output('action-messages-div', 'children', allow_duplicate=True),
         Output('ui-update-trigger', 'data', allow_duplicate=True),
         Output('player-query-input', 'value', allow_duplicate=True),
         Output('undo-player-id-input', 'value', allow_duplicate=True)],
        [Input('draft-opponent-btn', 'n_clicks'), Input('draft-my-team-btn', 'n_clicks'),
         Input('undo-draft-btn', 'n_clicks'), Input('player-query-input', 'n_submit')],
        [State('player-query-input', 'value'), State('undo-player-id-input', 'value'),
         State('ui-update-trigger', 'data')],
        prevent_initial_call=True
    )
    def handle_draft_actions(opp_clicks, my_clicks, undo_clicks, query_n_submit,
                             query_val_state, undo_pid_state, trigger_val):
        global GLOBALLY_DRAFTED_PLAYER_IDS, USER_DRAFTED_PLAYERS_DATA, USER_PLAYER_SLOT_ASSIGNMENTS
        triggered_prop_id = ctx.triggered[0]['prop_id']
        triggered_component_id = triggered_prop_id.split('.')[0]
        alerts, new_trigger_val = [], trigger_val
        player_query_return_value, undo_input_return_value = query_val_state, undo_pid_state # Default to not clearing
        active_query, action_type = "", None

        if triggered_component_id == 'player-query-input' and triggered_prop_id.endswith('.n_submit'): active_query, action_type = query_val_state, "draft_opponent_enter"
        elif triggered_component_id == 'draft-opponent-btn': active_query, action_type = query_val_state, "draft_opponent_button"
        elif triggered_component_id == 'draft-my-team-btn': active_query, action_type = query_val_state, "draft_my_team"
        elif triggered_component_id == 'undo-draft-btn': active_query, action_type = undo_pid_state, "undo"
        
        if not active_query and action_type not in [None, "undo"]:
             alerts.append(dbc.Alert("Player name/ID for drafting is empty.", color="warning", duration=3000)); return alerts, new_trigger_val, player_query_return_value, undo_input_return_value
        if not active_query and action_type == "undo":
             alerts.append(dbc.Alert("Player ID for undo is empty.", color="warning", duration=3000)); return alerts, new_trigger_val, player_query_return_value, undo_input_return_value
        
        target_pid, target_p_data = None, None
        if action_type in ["draft_opponent_enter", "draft_opponent_button", "draft_my_team"]:
            try:
                pid_candidate = int(active_query); p_data_cand = get_player_data(pid_candidate)
                if p_data_cand: target_pid, target_p_data = pid_candidate, p_data_cand
                else: alerts.append(dbc.Alert(f"ID '{active_query}' not found.", color="danger", duration=3000))
            except ValueError:
                matched = find_player_flexible(active_query)
                if not matched: alerts.append(dbc.Alert(f"Name '{active_query}' not found.", color="danger", duration=3000))
                elif len(matched) == 1: target_pid, target_p_data = matched[0][0], matched[0]
                else: 
                    ambiguous_display = [f"{p[1]} ({p[2]}, ID: {p[0]})" for p in matched[:5]]
                    alerts.append(dbc.Alert(f"Ambiguous: '{active_query}'. Matches: {', '.join(ambiguous_display)}. Use ID.", color="warning", duration=6000))
        
        if target_pid and target_p_data and action_type in ["draft_opponent_enter", "draft_opponent_button"]:
            if target_pid in GLOBALLY_DRAFTED_PLAYER_IDS and target_pid not in [p[0] for p in USER_DRAFTED_PLAYERS_DATA]:
                 alerts.append(dbc.Alert(f"{target_p_data[1]} already drafted by other.", color="info", duration=3000))
            else:
                GLOBALLY_DRAFTED_PLAYER_IDS.add(target_pid); removed_from_user = False
                if any(p[0] == target_pid for p in USER_DRAFTED_PLAYERS_DATA):
                    USER_DRAFTED_PLAYERS_DATA = [p for p in USER_DRAFTED_PLAYERS_DATA if p[0] != target_pid]
                    if target_pid in USER_PLAYER_SLOT_ASSIGNMENTS: del USER_PLAYER_SLOT_ASSIGNMENTS[target_pid]
                    removed_from_user = True
                alerts.append(dbc.Alert(f"{target_p_data[1]} opponent draft." + (" Removed." if removed_from_user else ""), color="success", duration=3000))
                new_trigger_val, player_query_return_value = trigger_val + 1, "" # Clear input
        elif target_pid and target_p_data and action_type == "draft_my_team":
            if any(p[0] == target_pid for p in USER_DRAFTED_PLAYERS_DATA): alerts.append(dbc.Alert(f"{target_p_data[1]} already on your team.", color="info", duration=3000))
            elif target_pid in GLOBALLY_DRAFTED_PLAYER_IDS: alerts.append(dbc.Alert(f"{target_p_data[1]} already drafted by other.", color="danger", duration=3000))
            elif len(USER_DRAFTED_PLAYERS_DATA) >= TOTAL_ROSTER_SPOTS: alerts.append(dbc.Alert("Roster full.", color="danger", duration=3000))
            else:
                USER_DRAFTED_PLAYERS_DATA.append(target_p_data); GLOBALLY_DRAFTED_PLAYER_IDS.add(target_pid)
                alerts.append(dbc.Alert(f"You drafted {target_p_data[1]}!", color="success", duration=3000))
                new_trigger_val, player_query_return_value = trigger_val + 1, "" # Clear input
        elif action_type == "undo":
            try:
                pid_undo = int(active_query); p_data_undo = get_player_data(pid_undo)
                if not p_data_undo: alerts.append(dbc.Alert(f"ID '{pid_undo}' not found for undo.", color="danger", duration=3000))
                else:
                    actions_performed_undo = []
                    if pid_undo in GLOBALLY_DRAFTED_PLAYER_IDS: GLOBALLY_DRAFTED_PLAYER_IDS.remove(pid_undo); actions_performed_undo.append("global")
                    user_len_before = len(USER_DRAFTED_PLAYERS_DATA)
                    USER_DRAFTED_PLAYERS_DATA = [p for p in USER_DRAFTED_PLAYERS_DATA if p[0] != pid_undo]
                    if len(USER_DRAFTED_PLAYERS_DATA) < user_len_before: actions_performed_undo.append("your team")
                    if pid_undo in USER_PLAYER_SLOT_ASSIGNMENTS: del USER_PLAYER_SLOT_ASSIGNMENTS[pid_undo]
                    if actions_performed_undo:
                        alerts.append(dbc.Alert(f"{p_data_undo[1]} removed from {', '.join(actions_performed_undo)}.", color="info", duration=3000))
                        new_trigger_val, undo_input_return_value = trigger_val + 1, "" # Clear input
                    else: alerts.append(dbc.Alert(f"{p_data_undo[1]} not in drafted lists.", color="info", duration=3000))
            except ValueError: alerts.append(dbc.Alert(f"Invalid ID for undo: '{active_query}'. Must be number.", color="danger", duration=3000))
        if not (target_pid and target_p_data) and action_type not in ["undo", None] and not alerts: # If lookup failed for a draft action
            alerts.append(dbc.Alert("Player not found or processed for drafting. Check input.", color="warning", duration=3000))
        
        return alerts, new_trigger_val, player_query_return_value, undo_input_return_value

    @app.callback(
        [Output('restart-draft-modal', 'is_open'),
         Output('action-messages-div', 'children', allow_duplicate=True),
         Output('ui-update-trigger', 'data', allow_duplicate=True)],
        [Input('restart-draft-open-modal-btn', 'n_clicks'), Input('restart-draft-confirm-btn', 'n_clicks'), Input('restart-draft-cancel-btn', 'n_clicks')],
        [State('restart-draft-modal', 'is_open'), State('ui-update-trigger', 'data')],
        prevent_initial_call=True
    )
    def manage_restart_draft_modal(open_clicks, confirm_clicks, cancel_clicks, current_modal_is_open, current_trigger_val): # Unchanged
        global GLOBALLY_DRAFTED_PLAYER_IDS, USER_DRAFTED_PLAYERS_DATA, USER_PLAYER_SLOT_ASSIGNMENTS, CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA, CURRENT_PLAYERS_BY_POSITION_FOR_GA
        triggered_id, new_modal_state, action_message, new_trigger_val = ctx.triggered_id, current_modal_is_open, [], current_trigger_val
        if triggered_id == 'restart-draft-open-modal-btn': new_modal_state = True
        elif triggered_id == 'restart-draft-confirm-btn':
            GLOBALLY_DRAFTED_PLAYER_IDS, USER_DRAFTED_PLAYERS_DATA, USER_PLAYER_SLOT_ASSIGNMENTS = set(), [], {}
            CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA, CURRENT_PLAYERS_BY_POSITION_FOR_GA = [], {}
            action_message = dbc.Alert("Draft restarted!", color="warning", duration=4000, dismissable=True)
            new_trigger_val, new_modal_state = current_trigger_val + 1, False
        elif triggered_id == 'restart-draft-cancel-btn': new_modal_state = False
        return new_modal_state, action_message, new_trigger_val

    @app.callback( # Run GA
        Output('ga-results-display', 'children'), Input('run-ga-btn', 'n_clicks'), prevent_initial_call=True
    )
    def run_ga_callback(n_clicks): # Unchanged
        if len(USER_DRAFTED_PLAYERS_DATA) >= TOTAL_ROSTER_SPOTS: return [dbc.Alert("Roster full.", color="info")]
        prepare_for_ga_run()
        overall_picks = len(GLOBALLY_DRAFTED_PLAYER_IDS)
        curr_overall_rd = math.floor(overall_picks / PICKS_PER_ROUND) + 1 if PICKS_PER_ROUND > 0 else 1
        best_ids, best_fit, ga_msgs = genetic_algorithm_adp_lineup(curr_overall_rd)
        next_pick_suggestion_data = None
        if best_ids and best_fit > -PENALTY_VIOLATION * 70 :
            user_picked_ids = {p[0] for p in USER_DRAFTED_PLAYERS_DATA}
            next_pick_suggestion_data = suggest_next_best_pick(best_ids, user_picked_ids)
        return format_ga_results_display(best_ids, best_fit, ga_msgs, next_pick_suggestion_data)

    @app.callback( # Scoring Mode Change
        [Output('ui-update-trigger', 'data', allow_duplicate=True),
         Output('action-messages-div', 'children', allow_duplicate=True)],
        Input('scoring-mode-selector', 'value'),
        State('ui-update-trigger', 'data'),
        prevent_initial_call=True 
    )
    def handle_scoring_mode_change(selected_mode, current_trigger_val): # Unchanged
        global CURRENT_SCORING_MODE, USER_PLAYER_SLOT_ASSIGNMENTS, CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA, CURRENT_PLAYERS_BY_POSITION_FOR_GA
        if selected_mode == CURRENT_SCORING_MODE: return dash.no_update, dash.no_update
        previous_mode = CURRENT_SCORING_MODE
        update_player_data_for_scoring_mode(selected_mode)
        USER_PLAYER_SLOT_ASSIGNMENTS = {}; CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA, CURRENT_PLAYERS_BY_POSITION_FOR_GA = [], {}
        alert_msg = dbc.Alert(f"Scoring mode: {previous_mode} → {selected_mode}. Player data updated.", color="info", duration=5000, dismissable=True)
        return current_trigger_val + 1, alert_msg
        
    @app.callback( # Update All Displays
        [Output('my-team-starters-table', 'children'), Output('my-team-starters-summary', 'children'),
         Output('my-team-bench-table', 'children'), Output('my-team-bench-summary', 'children'),
         Output('my-team-surplus-players', 'children'), Output('current-round-info', 'children'),
         Output('drafted-players-display', 'children'), Output('available-players-display', 'children'),
         Output('roster-structure-summary-display', 'children', allow_duplicate=True)],
        [Input('ui-update-trigger', 'data'), Input('available-pos-filter-dd', 'value'), Input('scoring-mode-selector', 'value')],
        prevent_initial_call=True 
    )
    def update_all_displays(trigger_val, avail_pos_filter, scoring_mode_val_unused): # Unchanged
        headers, sd, spg, ns, ts, _, bd, nb, tb, surplus = format_team_display_data()
        tbl_args = {"style_cell": {'textAlign': 'left', 'padding': '5px', 'fontFamily': 'sans-serif', 'fontSize': '0.85rem'},
                    "style_header": {'backgroundColor': 'rgba(0,0,0,0.03)', 'fontWeight': 'bold', 'borderBottom': '1px solid #dee2e6'},
                    "style_data_conditional": [{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgba(0,0,0,0.015)'}],
                    "style_table": {'border': '1px solid #dee2e6', 'borderRadius': '3px'}}
        starters_tbl = dash_table.DataTable(columns=headers, data=sd, **tbl_args) if sd else html.P("No starters.", className="text-muted")
        bench_tbl = dash_table.DataTable(columns=headers, data=bd, **tbl_args) if bd else html.P("No bench.", className="text-muted")
        surplus_disp = [html.H6("Surplus:", className="mt-3")] + [dbc.ListGroup(surplus, flush=True, style={'fontSize': '0.85rem'})] if surplus else []
        overall_picks = len(GLOBALLY_DRAFTED_PLAYER_IDS)
        curr_overall_rd = math.floor(overall_picks / PICKS_PER_ROUND) + 1 if PICKS_PER_ROUND > 0 else 1
        round_info_txt = f"Draft: Rd {curr_overall_rd} (Pick {overall_picks + 1}) | Mode: {CURRENT_SCORING_MODE}"
        drafted_disp, avail_disp = [], []
        if not GLOBALLY_DRAFTED_PLAYER_IDS: drafted_disp.append(html.P("None yet.", className="text-muted"))
        else:
            drafted_sorted = sorted([(get_player_data(pid), pid in [p[0] for p in USER_DRAFTED_PLAYERS_DATA])
                                  for pid in GLOBALLY_DRAFTED_PLAYER_IDS if get_player_data(pid)], key=lambda x: x[0][4])
            items = []
            for p_data, is_user in drafted_sorted:
                badge = dbc.Badge("You", color="success", pill=True, className="ms-auto") if is_user else None
                bye_str = f" Bye:{p_data[5]}" if p_data[5] > 0 else ""
                items.append(dbc.ListGroupItem([f"Rd {p_data[4]:>2}: {p_data[1]} ({p_data[2]}) PPG:{p_data[3]:.1f}{bye_str}", badge], className="d-flex justify-content-between align-items-center", style={'fontSize': '0.85rem'}))
            drafted_disp = dbc.ListGroup(items, flush=True)
        avail_items = []
        avail_pool_sorted = sorted([p for p in INITIAL_PLAYER_POOL_DATA if p[0] not in GLOBALLY_DRAFTED_PLAYER_IDS], key=lambda x: (x[4], -x[3]))
        count = 0; limit = 75
        for p_av in avail_pool_sorted:
            if avail_pos_filter != 'ALL' and p_av[2] != avail_pos_filter: continue
            bye_str_av = f" Bye:{p_av[5]}" if p_av[5] > 0 else ""
            avail_items.append(dbc.ListGroupItem(f"{p_av[1]}({p_av[2]}) PPG:{p_av[3]:.1f} Rd:{p_av[4]}{bye_str_av} (ID:{p_av[0]})", style={'fontSize': '0.85rem'}))
            count +=1;
            if count >= limit: break
        if not avail_items: avail_items.append(dbc.ListGroupItem("None available.", className="text-muted"))
        avail_disp = dbc.ListGroup(avail_items, flush=True)
        roster_summary_children = get_roster_structure_info()
        return starters_tbl, f"Starters ({ns}/{ts}) PPG: {spg:.2f}", bench_tbl, f"Bench ({nb}/{tb})", surplus_disp, round_info_txt, drafted_disp, avail_disp, roster_summary_children

    # --- Run the App ---
    if __name__ == '__main__':
        if INITIAL_SETUP_SUCCESS:
            app.run(debug=False)
        else:
            print("Dash application cannot start due to initialization errors. See console.")
            if 'app' in locals() and app: app.run(debug=True)