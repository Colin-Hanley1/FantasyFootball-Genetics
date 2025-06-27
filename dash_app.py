import dash
from dash import dcc, html, dash_table, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

import random
import math
import csv
import os
from collections import Counter
import numpy as np
import copy

# --- Configuration Constants ---
GAMES_IN_SEASON = 17
DEFAULT_CSV_FILENAME = "player_pool.csv"
GA_LOG_FILENAME = "ga_training_log.csv"

# --- Roster Configuration (Flex Eligibility is a global constant) ---
FLEX_ELIGIBILITY = {
    "Flex": ("WR", "RB", "TE"), "W/R": ("WR", "RB"), "R/T": ("RB", "TE"),
    "SUPERFLEX": ("QB", "WR", "RB", "TE"), "BN_SUPERFLEX": ("QB", "WR", "RB", "TE"),
    "BN_FLX": ("WR", "RB", "TE")
}

# --- Global, Read-Only Data ---
MASTER_PLAYER_POOL_RAW = []

# --- GA Parameters (Global constants) ---
POPULATION_SIZE = 100
N_GENERATIONS = 250
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5
PENALTY_VIOLATION = 10000
STARTER_PPG_MULTIPLIER = 3.0
BENCH_PPG_MULTIPLIER = 1.5
BENCH_VALUE_SCALER = 0.1
BENCH_ADP_PENALTY_SCALER = 1
STARTER_ADP_WEAKNESS_THRESHOLD = 2
EARLY_ROUND_ADP_BENCH_PENALTY = 0.5
BYE_WEEK_CONFLICT_PENALTY_FACTOR = 0.001
BACKUP_POSITION_PENALTY_SCALER = 0.4
NEXT_PICK_ADP_LOOKAHEAD_ROUNDS = 2
UNDRAFTABLE_ADP_PENALTY_SCALER = 0.1
ALLOWED_REACH_ROUNDS = 1
REACH_PENALTY_SCALER = 0.0

# --- Data Loading and Session Management ---

def load_master_player_pool_from_csv(filename=DEFAULT_CSV_FILENAME):
    player_pool_list = []
    expected_headers = ["ID", "Name", "Position", "PPRPoints", "PPRADP", "STDPoints", "STDADP", "STDSFADP", "PPRSFADP", "ByeWeek", "Team"]
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames or not all(h in reader.fieldnames for h in expected_headers):
                missing = [h for h in expected_headers if h not in (reader.fieldnames or [])]
                print(f"Error: CSV '{filename}' missing headers: {missing}. Got: {reader.fieldnames}")
                return []
            for row_num, row in enumerate(reader, 1):
                try:
                    player_data = {
                        "ID": int(row["ID"]),
                        "Name": row["Name"],
                        "Position": row["Position"].upper(),
                        "PPRPoints": float(row["PPRPoints"]),
                        "STDPoints": float(row["STDPoints"]),
                        "PPRADP": int(float(row.get("PPRADP") or 999)),
                        "STDADP": int(float(row.get("STDADP") or 999)),
                        "PPRSFADP": int(float(row.get("PPRSFADP") or 999)),
                        "STDSFADP": int(float(row.get("STDSFADP") or 999)),
                        "ByeWeek": int(row["ByeWeek"]) if row["ByeWeek"] and row["ByeWeek"].strip() else 0,
                        "Team": row.get("Team", "N/A")
                    }
                    player_pool_list.append(player_data)
                except (KeyError, ValueError) as e:
                    print(f"Warning: Skipping row {row_num} in '{filename}': {e}. Row: {row}")
    except FileNotFoundError:
        print(f"Error: CSV file '{filename}' not found."); return []
    except Exception as e:
        print(f"Unexpected error reading '{filename}': {e}"); return []
    return player_pool_list

def get_initial_session_data():
    return {
        'scoring_mode': "PPR",
        'picks_per_round': 8,
        'roster_structure': {
            "QB": 1, "RB": 2, "WR": 2, "TE": 1, "Flex": 1,
            "SUPERFLEX": 0,
            "BN_SUPERFLEX": 1, "BN_FLX": 4
        },
        'globally_drafted_player_ids': [],
        'user_drafted_player_ids': [],
    }

def is_superflex_mode(roster_structure):
    return (
        roster_structure.get("SUPERFLEX", 0) > 0 or
        roster_structure.get("QB", 0) >= 2
    )

def get_processed_player_pool_for_session(session_data):
    scoring_mode = session_data.get('scoring_mode', 'PPR')
    picks_per_round = session_data.get('picks_per_round', 8)
    roster_structure = session_data.get('roster_structure', {})
    superflex_active = is_superflex_mode(roster_structure)
    processed_pool_tuples = []
    for p_raw in MASTER_PLAYER_POOL_RAW:
        active_points = p_raw['PPRPoints'] if scoring_mode == "PPR" else p_raw['STDPoints']
        if superflex_active:
            active_adp = p_raw['PPRSFADP'] if scoring_mode == "PPR" else p_raw['STDSFADP']
        else:
            active_adp = p_raw['PPRADP'] if scoring_mode == "PPR" else p_raw['STDADP']
        ppg = active_points / GAMES_IN_SEASON if GAMES_IN_SEASON > 0 else 0
        calc_round = max(1, math.ceil(active_adp / picks_per_round)) if picks_per_round > 0 else 1
        processed_pool_tuples.append(
            (p_raw['ID'], p_raw['Name'], p_raw['Position'], ppg, calc_round, p_raw['ByeWeek'], p_raw['Team'])
        )
    processed_id_map = {p[0]: p for p in processed_pool_tuples}
    return processed_pool_tuples, processed_id_map

def get_roster_derived_details(roster_structure):
    position_order, total_spots = [], 0
    starters = {k: v for k, v in roster_structure.items() if not k.startswith("BN_")}
    bench = {k: v for k, v in roster_structure.items() if k.startswith("BN_")}
    starter_display_order = ["QB", "RB", "WR", "TE", "Flex", "W/R", "R/T", "SUPERFLEX"]
    sorted_starters = sorted(
        starters.items(),
        key=lambda item: starter_display_order.index(item[0]) if item[0] in starter_display_order else float('inf')
    )
    for slot, count in sorted_starters:
        if count > 0:
            position_order.extend([slot] * count)
            total_spots += count
    for slot, count in sorted(bench.items()):
        if count > 0:
            position_order.extend([slot] * count)
            total_spots += count
    return position_order, total_spots

# --- Stateless Helper Functions ---
def get_roster_structure_info_component(session_data):
    roster_structure = session_data['roster_structure']
    scoring_mode = session_data['scoring_mode']
    items = [dbc.ListGroupItem(html.H5("Current Roster Settings", className="mb-0"), className="bg-secondary text-white")]
    for slot, count in roster_structure.items():
        elig_str = f" (Eligible: {', '.join(FLEX_ELIGIBILITY[slot])})" if slot in FLEX_ELIGIBILITY and slot in roster_structure and roster_structure[slot] > 0 else ""
        items.append(dbc.ListGroupItem(f"{slot}: {count}{elig_str}"))
    items.append(dbc.ListGroupItem(f"Starter PPGs x{STARTER_PPG_MULTIPLIER} in fitness.", color="info"))
    items.append(dbc.ListGroupItem(f"Active Scoring Mode: {scoring_mode}", className="text-primary fw-bold"))
    return dbc.ListGroup(items, className="mb-3")

def format_ga_results_display(best_lineup_ids, best_fitness, ga_messages, next_pick_suggestion, processed_id_map, position_order, user_drafted_pids):
    children = [html.H5("GA Suggestions", className="card-title")]
    if next_pick_suggestion:
        p = next_pick_suggestion
        children.append(dbc.Alert([
            html.H5("🎯 Next Pick Suggestion:", className="alert-heading"),
            html.P(f"Consider: {p[1]} ({p[2]})", className="mb-1"),
            html.P(f"PPG: {p[3]:.2f} | ADP Rd: {p[4]} | Bye: {p[5] if p[5] > 0 else '-'}", style={'fontSize': '0.9rem'})
        ], color="primary", className="mt-1 mb-3 shadow-sm"))
    children.extend(ga_messages)
    if best_lineup_ids and best_fitness > -PENALTY_VIOLATION * 50:
            items = []
            for i, pid in enumerate(best_lineup_ids):
                p_data = processed_id_map.get(pid)
                slot_type = position_order[i]
                user_pick_str = " (Your Pick)" if pid in user_drafted_pids else ""
                if p_data:
                    txt = f"S{i:02d}({slot_type:<10}): {p_data[1]:<20}({p_data[2]:<2}) PPG:{p_data[3]:>5.2f} Bye:{p_data[5] if p_data[5] > 0 else '-'} Rd:{p_data[4]:>2}{user_pick_str}"
                else:
                    txt = f"S{i:02d}({slot_type:<10}): EMPTY"
                items.append(dbc.ListGroupItem(txt, style={'fontSize': '0.8rem', 'padding': '0.25rem 0.5rem'}))
            children.append(html.H6("📋 GA's Optimal Full Lineup:", className="mt-3"))
            children.append(dbc.ListGroup(items, flush=True, className="mt-2"))
    return children

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
        first_initials, last_name = name_part[:i]
        last_name = name_part[i:]
        if len(first_initials) > 2 and not last_name: last_name, first_initials = first_initials, ""
        return first_initials.lower(), last_name.lower()
    except Exception: return "", ""

def find_player_flexible(query_str, master_pool_raw):
    user_query_normalized_filn = normalize_name_for_search(query_str)
    user_query_simple_lower = query_str.lower().replace('.', '').replace("'", "").strip()
    POSSIBLE_QUERY_POSITIONS = {"qb", "wr", "rb", "te", "k", "def", "dst"}
    query_pos_part, name_only_query_words_for_ln_match = None, []
    for word in user_query_simple_lower.split():
        if word in POSSIBLE_QUERY_POSITIONS: query_pos_part = word.upper()
        else: name_only_query_words_for_ln_match.append(word)
    potential_query_lastname_norm = normalize_name_for_search(" ".join(name_only_query_words_for_ln_match[-1:])) if name_only_query_words_for_ln_match else ""
    exact_csv_matches, filn_format_matches, lastname_matches, strong_partial_matches, partial_matches = [], [], [], [], []
    for p_data_dict in master_pool_raw:
        pid, csv_name, csv_pos_col = p_data_dict["ID"], p_data_dict["Name"], p_data_dict["Position"]
        csv_name_lower = csv_name.lower()
        if user_query_simple_lower == csv_name_lower: exact_csv_matches.append(p_data_dict); continue
        fi_csv, ln_csv = parse_csv_name_to_fi_ln(csv_name)
        csv_name_normalized_filn = fi_csv + ln_csv
        if user_query_normalized_filn == csv_name_normalized_filn:
            if query_pos_part and query_pos_part == csv_pos_col: filn_format_matches.insert(0, p_data_dict)
            elif not query_pos_part: filn_format_matches.append(p_data_dict)
            else: partial_matches.append(p_data_dict)
            continue
        if ln_csv and user_query_normalized_filn == ln_csv:
             if query_pos_part and query_pos_part == csv_pos_col: lastname_matches.insert(0,p_data_dict)
             elif not query_pos_part: lastname_matches.append(p_data_dict)
             else: partial_matches.append(p_data_dict)
             continue
        if ln_csv and potential_query_lastname_norm and len(potential_query_lastname_norm) >= 3 and ln_csv.startswith(potential_query_lastname_norm):
            if query_pos_part and query_pos_part == csv_pos_col: strong_partial_matches.append(p_data_dict)
            elif not query_pos_part: partial_matches.append(p_data_dict)
            continue
        csv_name_part_after_underscore = csv_name_lower.split('_',1)[-1]
        if len(user_query_simple_lower) >= 3 and user_query_simple_lower in csv_name_part_after_underscore:
            partial_matches.append(p_data_dict); continue
    final_results, seen_ids = [], set()
    for p_list in [exact_csv_matches, filn_format_matches, lastname_matches, strong_partial_matches, partial_matches]:
        for p_item_dict in p_list:
            if p_item_dict['ID'] not in seen_ids:
                final_results.append(p_item_dict); seen_ids.add(p_item_dict['ID'])
    return final_results

# --- Main GA Logic ---
def prepare_for_ga_run(session_data, processed_player_pool, processed_id_map):
    """
    Assigns drafted players to roster slots based on a logical order.
    Now with constraints on bench positions.
    """
    roster_structure = session_data['roster_structure']
    user_drafted_ids = set(session_data['user_drafted_player_ids'])
    position_order, total_roster_spots = get_roster_derived_details(roster_structure)
    user_player_data = [processed_id_map[pid] for pid in user_drafted_ids if pid in processed_id_map]
    user_slot_assignments = {}

    player_groups = {}
    for p_data in user_player_data:
        player_groups.setdefault(p_data[2], []).append(p_data)
    for pos in player_groups:
        player_groups[pos].sort(key=lambda x: x[3], reverse=True)

    temp_player_groups = copy.deepcopy(player_groups)
    unassigned_starter_slots = [i for i, p_type in enumerate(position_order) if not p_type.startswith("BN_")]
    
    for slot_idx in unassigned_starter_slots:
        slot_type = position_order[slot_idx]
        best_player_to_assign = None
        if slot_type in temp_player_groups and temp_player_groups[slot_type]:
            best_player_to_assign = temp_player_groups[slot_type][0]
        elif slot_type in FLEX_ELIGIBILITY:
            candidate_players = []
            for pos in FLEX_ELIGIBILITY[slot_type]:
                if pos in temp_player_groups and temp_player_groups[pos]:
                    candidate_players.append(temp_player_groups[pos][0])
            if candidate_players:
                best_player_to_assign = max(candidate_players, key=lambda p: p[3])
        if best_player_to_assign:
            user_slot_assignments[best_player_to_assign[0]] = slot_idx
            temp_player_groups[best_player_to_assign[2]].pop(0)

    # --- NEW BENCH ASSIGNMENT LOGIC ---
    assigned_starter_pids = set(user_slot_assignments.keys())
    remaining_players = [p for p in user_player_data if p[0] not in assigned_starter_pids]
    remaining_players.sort(key=lambda x: x[3], reverse=True)

    assigned_starter_slots = set(user_slot_assignments.values())
    available_bench_slots = [i for i, p_type in enumerate(position_order) if p_type.startswith("BN_") and i not in assigned_starter_slots]

    # Calculate bench limits
    num_starter_qb = roster_structure.get("QB", 0) + roster_structure.get("SUPERFLEX", 0)
    num_starter_rb = roster_structure.get("RB", 0)
    num_starter_wr = roster_structure.get("WR", 0)
    num_starter_te = roster_structure.get("TE", 0)
    num_flex_spots = roster_structure.get("Flex", 0) + roster_structure.get("W/R", 0) + roster_structure.get("R/T", 0) + roster_structure.get("W/R/T", 0)

    max_bench_qb = num_starter_qb
    max_bench_te = num_starter_te
    max_bench_rb = num_starter_rb + num_flex_spots
    max_bench_wr = num_starter_wr + num_flex_spots
    bench_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}

    # Prioritize QBs for BN_SUPERFLEX
    bn_superflex_slots = [i for i in available_bench_slots if position_order[i] == 'BN_SUPERFLEX']
    other_bench_slots = [i for i in available_bench_slots if position_order[i] != 'BN_SUPERFLEX']
    
    players_to_place_on_bench = list(remaining_players)
    for player in list(players_to_place_on_bench):
        pos = player[2]
        if pos == 'QB' and bn_superflex_slots and bench_counts['QB'] < max_bench_qb:
            slot_idx = bn_superflex_slots.pop(0)
            user_slot_assignments[player[0]] = slot_idx
            bench_counts['QB'] += 1
            players_to_place_on_bench.remove(player)

    # Fill remaining bench slots respecting limits
    remaining_bench_slots = other_bench_slots + bn_superflex_slots
    remaining_bench_slots.sort()

    for player in players_to_place_on_bench:
        if not remaining_bench_slots: break
        pos = player[2]
        limit_reached = (pos == 'RB' and bench_counts['RB'] >= max_bench_rb) or \
                        (pos == 'WR' and bench_counts['WR'] >= max_bench_wr) or \
                        (pos == 'TE' and bench_counts['TE'] >= max_bench_te)
        if not limit_reached:
            slot_idx = remaining_bench_slots.pop(0)
            user_slot_assignments[player[0]] = slot_idx
            bench_counts[pos] += 1
    
    globally_drafted_ids = set(session_data['globally_drafted_player_ids'])
    available_pool = [p for p in processed_player_pool if p[0] not in globally_drafted_ids]
    players_by_pos = {pos: sorted([p for p in available_pool if p[2] == pos], key=lambda x: (x[4], -x[3])) for pos in set(p[2] for p in available_pool)}
    return {"available_pool": available_pool, "players_by_pos": players_by_pos, "user_slot_assignments": user_slot_assignments, "position_order": position_order, "total_roster_spots": total_roster_spots}

def genetic_algorithm_adp_lineup(curr_round, ga_context, processed_id_map, session_data):
    ui_messages = []
    total_roster_spots = ga_context['total_roster_spots']
    position_order = ga_context['position_order']
    user_slot_assignments = ga_context['user_slot_assignments']
    roster_structure = session_data['roster_structure']
    def get_player_data(pid): return processed_id_map.get(pid)
    def get_eligible_players_for_slot_type(slot_type):
        eligible, processed_ids = [], set()
        if slot_type in FLEX_ELIGIBILITY:
            for actual_pos in FLEX_ELIGIBILITY[slot_type]:
                for player in ga_context['players_by_pos'].get(actual_pos, []):
                    if player[0] not in processed_ids: eligible.append(player); processed_ids.add(player[0])
        elif slot_type in ga_context['players_by_pos']:
            eligible.extend(ga_context['players_by_pos'][slot_type])
        return eligible
    def calculate_fitness(individual_ids, current_draft_round):
        if not individual_ids or len(individual_ids) != total_roster_spots: return -float('inf'), 0, set(), []
        if any(pid is None or (not isinstance(pid, int)) or (pid <= 0 and pid != -99) for pid in individual_ids):
            inv_spots = sum(1 for pid in individual_ids if pid != -99 and (pid is None or not isinstance(pid, int) or pid <= 0))
            if inv_spots > 0: return -PENALTY_VIOLATION * (inv_spots + 20), 0, set(), []
        lineup_player_objects = [None] * total_roster_spots
        for i, pid in enumerate(individual_ids):
            slot_type = position_order[i]
            is_starting_slot = not slot_type.startswith("BN_")
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
            slot_type, actual_pos = position_order[i], p_data_current[2]
            is_valid = (slot_type == actual_pos) or (slot_type in FLEX_ELIGIBILITY and actual_pos in FLEX_ELIGIBILITY.get(slot_type, []))
            if not is_valid: return -PENALTY_VIOLATION * 10, 0, set(), valid_player_data_for_checks
        player_ids_for_dup_check = [p[0] for p in valid_player_data_for_checks]
        if len(set(player_ids_for_dup_check)) != len(player_ids_for_dup_check): return -PENALTY_VIOLATION * 5, 0, set(), valid_player_data_for_checks
        raw_ppg_sum, fitness_ppg_component, bench_value_score = 0, 0, 0
        for i, p_data_calc in enumerate(lineup_player_objects):
            if p_data_calc is None: continue
            ppg, adp_round = p_data_calc[3], p_data_calc[4]
            raw_ppg_sum += ppg
            if not position_order[i].startswith("BN_"):
                fitness_ppg_component += ppg * STARTER_PPG_MULTIPLIER
            else:
                fitness_ppg_component += ppg * BENCH_PPG_MULTIPLIER
                bench_value_score += ppg * adp_round
        fitness_score = fitness_ppg_component + (bench_value_score * BENCH_VALUE_SCALER)
        bench_adp_mismanagement_penalty = 0
        starters_adp_data = [p[4] for i, p in enumerate(lineup_player_objects) if p and not position_order[i].startswith("BN_")]
        if starters_adp_data:
            avg_s_adp_rnd, min_s_adp_rnd_val = sum(starters_adp_data) / len(starters_adp_data), min(starters_adp_data)
            for i_bench, p_data_bench in enumerate(lineup_player_objects):
                if p_data_bench and position_order[i_bench].startswith("BN_"):
                    bench_p_adp_rnd = p_data_bench[4]
                    if bench_p_adp_rnd < (avg_s_adp_rnd - STARTER_ADP_WEAKNESS_THRESHOLD):
                        bench_adp_mismanagement_penalty += (PENALTY_VIOLATION * BENCH_ADP_PENALTY_SCALER * (avg_s_adp_rnd - bench_p_adp_rnd) / 10.0)
                    if bench_p_adp_rnd <= 3 and min_s_adp_rnd_val > (bench_p_adp_rnd + STARTER_ADP_WEAKNESS_THRESHOLD):
                        bench_adp_mismanagement_penalty += (PENALTY_VIOLATION * EARLY_ROUND_ADP_BENCH_PENALTY * (4 - bench_p_adp_rnd) / 5.0)
        fitness_score -= bench_adp_mismanagement_penalty
        bye_weeks_on_roster = [p[5] for p in valid_player_data_for_checks if p[5] is not None and 0 < p[5] < 20]
        num_bye_week_conflicts = sum(count - 1 for count in Counter(bye_weeks_on_roster).values() if count >= 2)
        if num_bye_week_conflicts > 0: fitness_score -= (PENALTY_VIOLATION * BYE_WEEK_CONFLICT_PENALTY_FACTOR * num_bye_week_conflicts)
        missing_backup_penalty_points = 0
        core_starter_positions_defined = {slot for slot, count in roster_structure.items() if count > 0 and not slot.startswith("BN_") and slot not in FLEX_ELIGIBILITY}
        if core_starter_positions_defined:
            bench_player_actual_positions = Counter(p[2] for i, p in enumerate(lineup_player_objects) if p is not None and position_order[i].startswith("BN_"))
            for core_pos in core_starter_positions_defined:
                if any(p and not position_order[i].startswith("BN_") and p[2] == core_pos for i, p in enumerate(lineup_player_objects)) and bench_player_actual_positions[core_pos] == 0:
                    missing_backup_penalty_points += 1
            if missing_backup_penalty_points > 0: fitness_score -= (PENALTY_VIOLATION * BACKUP_POSITION_PENALTY_SCALER * missing_backup_penalty_points)
        if total_roster_spots > 0:
            draft_round_threshold = total_roster_spots + 2
            undraftable_adp_penalty = sum(p[4] - draft_round_threshold for p in valid_player_data_for_checks if p[4] > draft_round_threshold)
            if undraftable_adp_penalty > 0: fitness_score -= (PENALTY_VIOLATION * UNDRAFTABLE_ADP_PENALTY_SCALER * undraftable_adp_penalty)
        total_reach_penalty_points = sum(p[4] - current_draft_round - ALLOWED_REACH_ROUNDS for p in valid_player_data_for_checks if (p[4] - current_draft_round) > ALLOWED_REACH_ROUNDS)
        if total_reach_penalty_points > 0: fitness_score -= (PENALTY_VIOLATION * REACH_PENALTY_SCALER * total_reach_penalty_points)
        player_adp_rounds_in_lineup = [p[4] for p in valid_player_data_for_checks]
        adp_round_counts = Counter(player_adp_rounds_in_lineup)
        num_future_round_stacking_violations = sum(count - 1 for adp_r, count in adp_round_counts.items() if count > 1 and adp_r >= current_draft_round)
        if num_future_round_stacking_violations > 0: fitness_score -= (PENALTY_VIOLATION * num_future_round_stacking_violations * 1.5)
        return fitness_score, raw_ppg_sum, set(player_adp_rounds_in_lineup), valid_player_data_for_checks

    def repair_lineup(lineup_ids):
        repaired = list(lineup_ids)
        if not repaired or len(repaired) != total_roster_spots: return create_individual()
        for pid, slot_idx in user_slot_assignments.items():
            if 0 <= slot_idx < total_roster_spots and repaired[slot_idx] != pid:
                try:
                    current_occupant, other_idx = repaired[slot_idx], repaired.index(pid)
                    repaired[slot_idx], repaired[other_idx] = pid, current_occupant
                except ValueError: repaired[slot_idx] = pid
        counts = Counter(pid for pid in repaired if pid not in [-99, None])
        for pid, count in counts.items():
            if count > 1:
                indices = [i for i, x in enumerate(repaired) if x == pid and i not in user_slot_assignments.values()]
                for i in indices[1:]: repaired[i] = -99
        for i in range(total_roster_spots):
            if repaired[i] in [-99, None] and i not in user_slot_assignments.values():
                slot_type = position_order[i]
                current_pids = set(p for p in repaired if p not in [-99, None])
                candidates = [p for p in get_eligible_players_for_slot_type(slot_type) if p[0] not in current_pids]
                repaired[i] = random.choice(candidates)[0] if candidates else -99
        return repaired

    def create_individual():
        individual = [None] * total_roster_spots
        used_pids = set()
        for pid, slot_idx in user_slot_assignments.items():
            individual[slot_idx], used_pids = pid, used_pids.union({pid})
        for i in range(total_roster_spots):
            if individual[i] is None:
                slot_type = position_order[i]
                candidates = [p for p in get_eligible_players_for_slot_type(slot_type) if p[0] not in used_pids]
                if candidates:
                    chosen = random.choice(candidates)
                    individual[i], used_pids = chosen[0], used_pids.union({chosen[0]})
                else: individual[i] = -99
        return individual

    def tournament_selection(population, fitness_scores):
        if not population: return create_individual()
        tourn_size = min(TOURNAMENT_SIZE, len(population))
        if tourn_size <= 0: return random.choice(population)
        indices = random.sample(range(len(population)), tourn_size)
        return max([(population[i], fitness_scores[i]) for i in indices], key=lambda x: x[1])[0]

    def crossover(parent1, parent2):
        child1, child2 = list(parent1), list(parent2)
        if random.random() < CROSSOVER_RATE and total_roster_spots > 1:
            pt = random.randint(1, total_roster_spots - 1)
            child1, child2 = parent1[:pt] + parent2[pt:], parent2[:pt] + parent1[pt:]
        return repair_lineup(child1), repair_lineup(child2)

    def mutate(individual):
        if random.random() < MUTATION_RATE:
            ga_controlled_indices = [i for i in range(total_roster_spots) if i not in user_slot_assignments.values()]
            if not ga_controlled_indices: return repair_lineup(individual)
            idx_to_mutate = random.choice(ga_controlled_indices)
            slot_type = position_order[idx_to_mutate]
            current_pids = set(p for p in individual if p not in [-99, None] and p != individual[idx_to_mutate])
            candidates = [p for p in get_eligible_players_for_slot_type(slot_type) if p[0] not in current_pids]
            if candidates: individual[idx_to_mutate] = random.choice(candidates)[0]
        return repair_lineup(individual)

    if total_roster_spots <= len(user_slot_assignments):
        ui_messages.append(dbc.Alert("Roster full. No GA needed.", color="info"))
        ids = [-99] * total_roster_spots
        for pid, slot_idx in user_slot_assignments.items(): ids[slot_idx] = pid
        fit, ppg, _, _ = calculate_fitness(ids, curr_round)
        return ids, fit, ui_messages
    if not ga_context['available_pool']:
        ui_messages.append(dbc.Alert("No available players for GA.", color="warning"))
        return [], -1, ui_messages
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    best_overall_fitness, best_overall_individual = -float('inf'), population[0]
    for gen in range(N_GENERATIONS):
        fitness_results = [calculate_fitness(ind, curr_round) for ind in population]
        fitness_scores = [res[0] for res in fitness_results]
        gen_best_idx = np.argmax(fitness_scores)
        if fitness_scores[gen_best_idx] > best_overall_fitness:
            best_overall_fitness, best_overall_individual = fitness_scores[gen_best_idx], population[gen_best_idx]
        elites = sorted([(population[i], fitness_scores[i]) for i in range(len(population)) if fitness_scores[i] > -PENALTY_VIOLATION], key=lambda x: x[1], reverse=True)
        next_population = [e[0] for i, e in enumerate(elites) if i < max(1, int(0.05*POPULATION_SIZE))]
        while len(next_population) < POPULATION_SIZE:
            p1, p2 = tournament_selection(population, fitness_scores), tournament_selection(population, fitness_scores)
            c1, c2 = crossover(p1, p2)
            next_population.append(mutate(c1))
            if len(next_population) < POPULATION_SIZE: next_population.append(mutate(c2))
        population = next_population[:POPULATION_SIZE]
    final_fitness, final_ppg, _, _ = calculate_fitness(best_overall_individual, curr_round)
    ui_messages.extend([dbc.Alert("--- GA Finished ---", color="success"), dbc.Alert(f"🏆 Best Lineup (Fit: {final_fitness:.2f})", color="success", className="fw-bold"), dbc.Alert(f"Proj. True PPG: {final_ppg:.2f}", color="info")])
    return best_overall_individual, final_fitness, ui_messages

def suggest_next_best_pick(best_lineup_ids, user_drafted_pids, current_round, processed_id_map, position_order):
    ga_suggestions = []
    for i, pid in enumerate(best_lineup_ids):
        if pid not in user_drafted_pids and pid in processed_id_map:
            p_data = processed_id_map[pid]
            is_starter = not position_order[i].startswith("BN_")
            ga_suggestions.append({"data": p_data, "adp_round": p_data[4], "ppg": p_data[3], "is_starter": is_starter})
    if not ga_suggestions: return None
    timely_candidates = [p for p in ga_suggestions if p["adp_round"] <= (current_round + NEXT_PICK_ADP_LOOKAHEAD_ROUNDS)]
    pool = timely_candidates if timely_candidates else ga_suggestions
    pool.sort(key=lambda p: (not p["is_starter"], p["adp_round"], -p["ppg"]))
    return pool[0]["data"] if pool else None

# --- Initial Setup ---
try:
    MASTER_PLAYER_POOL_RAW = load_master_player_pool_from_csv(DEFAULT_CSV_FILENAME)
    INITIAL_SETUP_SUCCESS = bool(MASTER_PLAYER_POOL_RAW)
except Exception as e:
    print(f"CRITICAL ERROR during initial setup: {e}")
    INITIAL_SETUP_SUCCESS = False

# --- Dash App ---
DBC_THEME = dbc.themes.DARKLY
app = dash.Dash(__name__, external_stylesheets=[DBC_THEME], suppress_callback_exceptions=True)
load_figure_template(DBC_THEME)
server = app.server

if not INITIAL_SETUP_SUCCESS:
    app.layout = dbc.Container([dbc.Alert("Application Initialization Failed. Check CSV file and logs.", color="danger", className="mt-5")], fluid=True)
else:
    app.layout = dbc.Container([
        dcc.Store(id='session-store', storage_type='memory'),
        dcc.Store(id='notification-store', data={}),
        dcc.Interval(id='alert-interval', disabled=True, n_intervals=0, max_intervals=1, interval=4000),
        dbc.Row(dbc.Col(html.H1("🏈 Evolve Draft: Multi-User Fantasy Football Advisor", className="text-center my-4"))),
        dbc.Row(dbc.Col(id='current-round-info', className="text-center mb-3 fw-bold fs-5")),
        dbc.Row(dbc.Col(id='action-messages-div')),
        dbc.Row([
            dbc.Col(md=4, children=[
                dbc.Card(id='settings-and-actions-card'),
                dbc.Card([dbc.CardHeader(html.H4("Available Players", className="mb-0")), dbc.CardBody([dcc.Dropdown(id='available-pos-filter-dd', options=[{'label': 'All', 'value': 'ALL'}] + [{'label': p, 'value': p} for p in sorted(list(set(pl['Position'] for pl in MASTER_PLAYER_POOL_RAW)))], value='ALL', clearable=False, className="mb-2 dbc"), dcc.Loading(html.Div(id='available-players-display', style={'maxHeight': '300px', 'overflowY': 'auto'}))])], className="mt-3"),
                dbc.Card(id='roster-structure-summary-display', className="mt-3")
            ]),
            dbc.Col(md=5, children=[
                dbc.Card(id='my-team-card'),
                dbc.Card([dbc.CardHeader(html.H4("Algorithm Suggestions", className="mb-0")), dbc.CardBody([dbc.Button('Run GA Suggestions 🚀', id='run-ga-btn', color='primary', className="mb-3 w-100"), dcc.Loading(id="loading-ga-spinner", type="default", children=[html.Div(id='ga-results-display')])])], className="mt-3")
            ]),
            dbc.Col(md=3, children=[dbc.Card([dbc.CardHeader(html.H4("Globally Drafted Players", className="mb-0")), dcc.Loading(dbc.CardBody(html.Div(id='drafted-players-display', style={'maxHeight': 'calc(100vh - 160px)', 'overflowY': 'auto'})))])])
        ], className="mb-5")
    ], fluid=True)

    # --- CALLBACKS ---
    @app.callback(Output('session-store', 'data'), Input('session-store', 'data'))
    def initialize_session_data(session_data):
        if session_data is None: return get_initial_session_data()
        return dash.no_update

    @app.callback(
        [Output('session-store', 'data', allow_duplicate=True), Output('notification-store', 'data', allow_duplicate=True)],
        [Input('teams-slider', 'value'), Input('scoring-mode-selector', 'value'), Input('restart-draft-btn', 'n_clicks'), Input('apply-roster-changes-btn', 'n_clicks')],
        [State('session-store', 'data')] + [State(f"roster-input-{key.replace('/', '-')}", "value") for key in get_initial_session_data()['roster_structure'].keys()],
        prevent_initial_call=True)
    def handle_session_modifiers(new_team_count, selected_mode, restart_clicks, apply_roster_clicks, session_data, *roster_values):
        if session_data is None: return dash.no_update, dash.no_update
        triggered_id, new_session, notification = ctx.triggered_id, copy.deepcopy(session_data), {}
        if triggered_id == 'teams-slider' and new_team_count != new_session.get('picks_per_round'):
            new_session.update({'picks_per_round': new_team_count, 'globally_drafted_player_ids': [], 'user_drafted_player_ids': []})
            notification = {'message': f"League size set to {new_team_count}. Draft has been reset.", 'color': 'info'}
        elif triggered_id == 'scoring-mode-selector' and selected_mode != new_session.get('scoring_mode'):
            new_session.update({'scoring_mode': selected_mode, 'globally_drafted_player_ids': [], 'user_drafted_player_ids': []})
            notification = {'message': f"Scoring mode set to {selected_mode}. Draft has been reset.", 'color': 'info'}
        elif triggered_id == 'restart-draft-btn':
            new_session = get_initial_session_data()
            notification = {'message': "Draft has been successfully restarted.", 'color': 'warning'}
        elif triggered_id == 'apply-roster-changes-btn':
            new_roster, valid_update = {}, True
            for i, key in enumerate(get_initial_session_data()['roster_structure'].keys()):
                val = roster_values[i]
                if val is None or not isinstance(val, int) or val < 0:
                    notification = {'message': f"Invalid count for {key}. Must be a non-negative integer.", 'color': 'danger'}; valid_update = False; break
                new_roster[key] = val
            if valid_update:
                if sum(new_roster.values()) == 0:
                    notification = {'message': "Roster cannot have zero total spots.", 'color': 'danger'}
                else:
                    new_session.update({'roster_structure': new_roster, 'globally_drafted_player_ids': [], 'user_drafted_player_ids': []})
                    notification = {'message': "Roster structure updated. Draft has been reset.", 'color': 'success'}
        if notification: return new_session, notification
        return new_session, dash.no_update

    @app.callback(
        [Output('session-store', 'data', allow_duplicate=True), Output('notification-store', 'data', allow_duplicate=True),
         Output('player-query-input', 'value'), Output('undo-player-query-input', 'value')],
        [Input('draft-opponent-btn', 'n_clicks'), Input('draft-my-team-btn', 'n_clicks'), Input('undo-draft-btn', 'n_clicks'), Input('player-query-input', 'n_submit')],
        [State('player-query-input', 'value'), State('undo-player-query-input', 'value'), State('session-store', 'data')],
        prevent_initial_call=True)
    def handle_draft_actions(opp_clicks, my_clicks, undo_clicks, n_submit, query_val, undo_val, session_data):
        if not ctx.triggered_id or session_data is None: return dash.no_update, dash.no_update, query_val, undo_val
        triggered_id, new_session, notification = ctx.triggered_id, copy.deepcopy(session_data), {}
        query_out, undo_out, is_successful_action = query_val, undo_val, False
        active_query, action_type = (query_val, 'opponent') if triggered_id in ['player-query-input', 'draft-opponent-btn'] else (query_val, 'my_team') if triggered_id == 'draft-my-team-btn' else (undo_val, 'undo') if triggered_id == 'undo-draft-btn' else (None, None)
        if not active_query:
            notification = {'message': "Player name/ID cannot be empty.", 'color': 'warning'}
            return dash.no_update, notification, query_val, undo_val
        matched, p_data_dict = find_player_flexible(active_query, MASTER_PLAYER_POOL_RAW), None
        if not matched: notification = {'message': f"Player '{active_query}' not found.", 'color': 'danger'}
        elif len(matched) > 1: notification = {'message': f"Ambiguous name '{active_query}'. Use a more specific name or ID.", 'color': 'warning'}
        else: p_data_dict = matched[0]
        if p_data_dict:
            pid, pname = p_data_dict['ID'], p_data_dict['Name']
            drafted_set, my_team_set = set(new_session['globally_drafted_player_ids']), set(new_session['user_drafted_player_ids'])
            if action_type in ['opponent', 'my_team']:
                if pid in drafted_set: notification = {'message': f"{pname} has already been drafted.", 'color': 'info'}
                else:
                    new_session['globally_drafted_player_ids'].append(pid)
                    if action_type == 'my_team':
                        new_session['user_drafted_player_ids'].append(pid); notification = {'message': f"You drafted {pname}!", 'color': 'success'}
                    else:
                        if pid in my_team_set: new_session['user_drafted_player_ids'] = [p for p in new_session['user_drafted_player_ids'] if p != pid]
                        notification = {'message': f"Opponent drafted {pname}.", 'color': 'secondary'}
                    is_successful_action = True
            elif action_type == 'undo':
                if pid not in drafted_set: notification = {'message': f"{pname} was not on the draft board to undo.", 'color': 'info'}
                else:
                    new_session['globally_drafted_player_ids'] = [p for p in new_session['globally_drafted_player_ids'] if p != pid]
                    if pid in my_team_set: new_session['user_drafted_player_ids'] = [p for p in new_session['user_drafted_player_ids'] if p != pid]
                    notification = {'message': f"Removed {pname} from the draft board.", 'color': 'info'}
                    is_successful_action = True
        if is_successful_action:
            if action_type in ['opponent', 'my_team']: query_out = ''
            elif action_type == 'undo': undo_out = ''
        return new_session, notification, query_out, undo_out

    @app.callback(
        [Output('action-messages-div', 'children'), Output('alert-interval', 'disabled')],
        Input('notification-store', 'data'), prevent_initial_call=True)
    def display_notification(notification_data):
        if notification_data and notification_data.get('message'):
            message, color = notification_data['message'], notification_data.get('color', 'primary')
            return dbc.Alert(message, color=color, dismissable=True, duration=4000), False
        return None, True

    @app.callback(
        Output('action-messages-div', 'children', allow_duplicate=True),
        Input('alert-interval', 'n_intervals'), prevent_initial_call=True)
    def clear_alert(n_intervals):
        return None

    @app.callback(Output('ga-results-display', 'children'), Input('run-ga-btn', 'n_clicks'), State('session-store', 'data'), prevent_initial_call=True)
    def run_ga_callback(n_clicks, session_data):
        if n_clicks is None or session_data is None: return dbc.Alert("Session not ready.", color="warning")
        processed_pool, processed_id_map = get_processed_player_pool_for_session(session_data)
        ga_context = prepare_for_ga_run(session_data, processed_pool, processed_id_map)
        overall_picks = len(session_data['globally_drafted_player_ids'])
        curr_round = math.floor(overall_picks / session_data['picks_per_round']) + 1 if session_data['picks_per_round'] > 0 else 1
        best_ids, best_fit, ga_msgs = genetic_algorithm_adp_lineup(curr_round, ga_context, processed_id_map, session_data)
        next_pick = None
        if best_ids: next_pick = suggest_next_best_pick(best_ids, set(session_data['user_drafted_player_ids']), curr_round, processed_id_map, ga_context['position_order'])
        return format_ga_results_display(best_ids, best_fit, ga_msgs, next_pick, processed_id_map, ga_context['position_order'], set(session_data['user_drafted_player_ids']))

    @app.callback(
        [Output('my-team-card', 'children'), Output('settings-and-actions-card', 'children'), Output('current-round-info', 'children'),
         Output('drafted-players-display', 'children'), Output('available-players-display', 'children'), Output('roster-structure-summary-display', 'children')],
        [Input('session-store', 'data'), Input('available-pos-filter-dd', 'value')])
    def update_all_displays(session_data, avail_pos_filter):
        if session_data is None: return [dash.no_update] * 6
        processed_pool, processed_id_map = get_processed_player_pool_for_session(session_data)
        roster_structure, picks_per_round = session_data['roster_structure'], session_data['picks_per_round']
        position_order, total_roster_spots = get_roster_derived_details(roster_structure)
        ga_context = prepare_for_ga_run(session_data, processed_pool, processed_id_map)
        user_slot_assignments, display_rows, starters_ppg = ga_context['user_slot_assignments'], [], 0
        for i, slot_type in enumerate(position_order):
            pid = next((p for p, s_idx in user_slot_assignments.items() if s_idx == i), None)
            p_data, row = processed_id_map.get(pid), {'slot': i, 'pos': slot_type, 'player': '-', 'ppg': '-', 'adp_rd': '-', 'bye': '-'}
            if p_data:
                row.update({'player': f"{p_data[1]} ({p_data[2]})", 'ppg': f"{p_data[3]:.2f}", 'adp_rd': p_data[4], 'bye': p_data[5] if p_data[5] > 0 else "-"})
                if not slot_type.startswith("BN_"): starters_ppg += p_data[3]
            display_rows.append(row)
        starters_rows, bench_rows = [r for r in display_rows if not r['pos'].startswith("BN_")], [r for r in display_rows if r['pos'].startswith("BN_")]
        hover_style = {'if': {'state': 'hover'}, 'color': 'black', 'backgroundColor': 'rgba(255, 255, 255, 0.9)'}
        my_team_card = [
            dbc.CardHeader(html.H4("My Team", className="mb-0")),
            dbc.CardBody([
                html.H5("Starters"),
                dash_table.DataTable(data=starters_rows, columns=[{"name": i.upper(), "id": i} for i in ['pos', 'player', 'ppg', 'adp_rd', 'bye']], style_as_list_view=True, style_cell={'backgroundColor': 'transparent', 'color': 'white'}, style_header={'fontWeight': 'bold'}, style_data_conditional=[hover_style]),
                html.P(f"Total Starter PPG: {starters_ppg:.2f}", className="fw-bold mt-2 text-end"),
                html.H5("Bench", className="mt-3"),
                dash_table.DataTable(data=bench_rows, columns=[{"name": i.upper(), "id": i} for i in ['pos', 'player', 'ppg', 'adp_rd', 'bye']], style_as_list_view=True, style_cell={'backgroundColor': 'transparent', 'color': 'white'}, style_header={'fontWeight': 'bold'}, style_data_conditional=[hover_style]),
            ])]
        all_roster_keys = get_initial_session_data()['roster_structure'].keys()
        settings_card = [
             dbc.CardHeader(html.H5("Settings & Actions", className="mb-0")),
             dbc.CardBody([
                 dbc.Label("Scoring Mode:"),
                 dcc.RadioItems(id='scoring-mode-selector', options=[{'label': 'PPR', 'value': 'PPR'}, {'label': 'Standard', 'value': 'STD'}], value=session_data['scoring_mode'], labelStyle={'display': 'inline-block', 'marginRight': '20px'}, inputClassName="me-1"),
                 html.Hr(),
                 dbc.Label("Number of Teams:"),
                 dcc.Slider(id='teams-slider', min=8, max=16, step=2, value=picks_per_round, marks={i: str(i) for i in range(8, 17, 2)}),
                 html.Hr(),
                 dbc.Accordion([dbc.AccordionItem([dbc.Row([dbc.Col(dbc.Label(key, html_for=f"roster-input-{key.replace('/', '-')}", className="text-end"), width=5), dbc.Col(dbc.Input(id=f"roster-input-{key.replace('/', '-')}", type="number", value=roster_structure.get(key, 0), min=0, step=1, size="sm"), width=7)], className="mb-2 align-items-center") for key in all_roster_keys] + [dbc.Button("Apply Roster Changes", id="apply-roster-changes-btn", color="success", outline=True, className="w-100 mt-3")], title="Customize Roster Structure")], start_collapsed=True),
                 html.Hr(),
                 html.Div("Enter player and press Enter to draft for an opponent.", className="form-text mb-2"),
                 dbc.Input(id='player-query-input', placeholder='Player Name or ID', className='mb-2'),
                 dbc.ButtonGroup([dbc.Button('Draft for Opponent', id='draft-opponent-btn', color='warning', outline=True), dbc.Button('Draft for My Team', id='draft-my-team-btn', color='success', outline=True)], className="w-100 mb-3"),
                 dbc.Input(id='undo-player-query-input', placeholder='Player to Undo', className='mb-2'),
                 dbc.Button('Undo Draft', id='undo-draft-btn', color='danger', outline=True, className="w-100 mb-3"),
                 html.Hr(),
                 dbc.Button("Restart Entire Draft", id="restart-draft-btn", color="danger", className="w-100")])]
        superflex_indicator = " | SF Mode" if is_superflex_mode(roster_structure) else ""
        overall_picks = len(session_data['globally_drafted_player_ids'])
        curr_round = math.floor(overall_picks / picks_per_round) + 1 if picks_per_round > 0 else 1
        round_info_txt = f"Draft: Rd {curr_round} (Pick {overall_picks + 1}) | Mode: {session_data['scoring_mode']}{superflex_indicator}"
        drafted_pids, user_pids = set(session_data['globally_drafted_player_ids']), set(session_data['user_drafted_player_ids'])
        drafted_list_tuples = sorted([processed_id_map[pid] for pid in drafted_pids if pid in processed_id_map], key=lambda x: session_data['globally_drafted_player_ids'].index(x[0]))
        drafted_disp_items = [dbc.ListGroupItem([f"{i+1}. {p_data[1]} ({p_data[2]})", dbc.Badge("You", color="success", pill=True, className="ms-auto") if p_data[0] in user_pids else None], className="d-flex justify-content-between align-items-center", style={'fontSize': '0.85rem'}) for i, p_data in enumerate(drafted_list_tuples)]
        drafted_disp = dbc.ListGroup(drafted_disp_items, flush=True)
        avail_pool = [p for p in processed_pool if p[0] not in drafted_pids]
        if avail_pos_filter and avail_pos_filter != 'ALL': avail_pool = [p for p in avail_pool if p[2] == avail_pos_filter]
        avail_pool.sort(key=lambda p: (p[4], -p[3]))
        avail_disp = dbc.ListGroup([dbc.ListGroupItem(f"{p[1]} ({p[2]}) Rd:{p[4]}", style={'fontSize': '0.8rem', 'padding': '0.4rem 0.75rem'}) for p in avail_pool[:100]], flush=True)
        roster_summary = get_roster_structure_info_component(session_data)
        return my_team_card, settings_card, round_info_txt, drafted_disp, avail_disp, roster_summary

    # --- Run the App ---
if __name__ == '__main__':
    if INITIAL_SETUP_SUCCESS:
        app.run(debug=True)
    else:
        print("Dash application cannot start due to initialization errors.")
