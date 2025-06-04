import random
import math
import numpy as np
from collections import Counter
import csv
import os

# --- Configuration Constants ---
GAMES_IN_SEASON = 17
PICKS_PER_ROUND = 12
DEFAULT_CSV_FILENAME = "player_pool.csv"
GA_LOG_FILENAME = "ga_training_log.csv"

# --- Roster Configuration ---
FLEX_ELIGIBILITY = {
    "W/R/T": ("WR", "RB", "TE"),
    "W/R": ("WR", "RB"),
    "R/T": ("RB", "TE"),
    "SUPERFLEX": ("QB", "WR", "RB", "TE"),
    "BN_SUPERFLEX": ("QB", "WR", "RB", "TE"),
    "BN_FLX": ("WR", "RB", "TE")
}

ROSTER_STRUCTURE = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "W/R/T": 1,
    "BN_SUPERFLEX": 1,
    "BN_FLX": 4
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

# --- GA Parameters ---
POPULATION_SIZE = 100
N_GENERATIONS = 100
MUTATION_RATE = 0.20
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5
PENALTY_VIOLATION = 10000
STARTER_PPG_MULTIPLIER = 1.2  # <<< New constant: e.g., 20% bonus for starter PPG in fitness calculation

# --- 0. Data Loading and Initialization ---
def load_player_pool_from_csv(filename=DEFAULT_CSV_FILENAME):
    player_pool_list = []
    expected_headers = ["ID", "Name", "Position", "TotalPoints", "ADP"]
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames or not all(field in reader.fieldnames for field in expected_headers):
                missing_headers = [h for h in expected_headers if h not in (reader.fieldnames or [])]
                print(f"Error: CSV '{filename}' missing headers: {', '.join(missing_headers)}")
                return []
            for row_num, row in enumerate(reader, 1):
                try:
                    player_id = int(row["ID"])
                    name = row["Name"]
                    position = row["Position"].upper()
                    total_points = float(row["TotalPoints"])
                    if GAMES_IN_SEASON <= 0:
                        ppg = 0
                        # This warning will now be handled by initial_setup's exit condition
                    else:
                        ppg = total_points / GAMES_IN_SEASON
                    if PICKS_PER_ROUND <= 0:
                        calculated_round = 1
                        # This warning will now be handled by initial_setup's exit condition
                    else:
                        adp = int(row["ADP"])
                        calculated_round = max(1, math.ceil(adp / PICKS_PER_ROUND))
                    player_pool_list.append((player_id, name, position, ppg, calculated_round))
                except (ValueError, KeyError, ZeroDivisionError) as e:
                    print(f"Warning: Skipping row {row_num} in '{filename}': {e}. Row: {row}")
    except FileNotFoundError:
        print(f"Error: CSV file '{filename}' not found.")
    except Exception as e:
        print(f"Unexpected error reading '{filename}': {e}")
    if not player_pool_list: print(f"Warning: No players loaded from '{filename}'.")
    return player_pool_list

def initial_setup(csv_filename=DEFAULT_CSV_FILENAME):
    global INITIAL_PLAYER_POOL_DATA, MASTER_PLAYER_ID_TO_DATA
    global POSITION_ORDER, TOTAL_ROSTER_SPOTS

    if GAMES_IN_SEASON <= 0:
        print("Critical Error: GAMES_IN_SEASON must be positive. Exiting.")
        exit()
    if PICKS_PER_ROUND <= 0:
        print("Critical Error: PICKS_PER_ROUND must be positive. Exiting.")
        exit()

    INITIAL_PLAYER_POOL_DATA = load_player_pool_from_csv(csv_filename)
    if not INITIAL_PLAYER_POOL_DATA:
        print("Critical Error: Initial player pool is empty. Exiting.")
        exit()

    MASTER_PLAYER_ID_TO_DATA = {p[0]: p for p in INITIAL_PLAYER_POOL_DATA}

    temp_position_order = []
    starters = {k: v for k, v in ROSTER_STRUCTURE.items() if not k.startswith("BN_")}
    bench = {k: v for k, v in ROSTER_STRUCTURE.items() if k.startswith("BN_")}

    for slot_type, count in starters.items():
        for _ in range(count):
            temp_position_order.append(slot_type)
    for slot_type, count in bench.items():
        for _ in range(count):
            temp_position_order.append(slot_type)

    POSITION_ORDER = temp_position_order
    TOTAL_ROSTER_SPOTS = len(POSITION_ORDER)
    if TOTAL_ROSTER_SPOTS == 0:
        print("Critical Error: ROSTER_STRUCTURE is empty, resulting in zero roster spots. Exiting.")
        exit()

    unique_initial_positions = set(p[2] for p in INITIAL_PLAYER_POOL_DATA)
    all_roster_keys = set(ROSTER_STRUCTURE.keys())
    for slot_key in all_roster_keys:
        if slot_key not in FLEX_ELIGIBILITY and slot_key not in unique_initial_positions:
            print(f"Warning: Roster slot key '{slot_key}' is not a base position found in players AND not a defined FLEX type.")
    for flex_key, eligible_list in FLEX_ELIGIBILITY.items():
        for pos in eligible_list:
            if pos not in unique_initial_positions:
                 print(f"Warning: Position '{pos}' in FLEX_ELIGIBILITY for '{flex_key}' not in any player's position data.")
    print("Initial data loaded.")
    print(f"Full POSITION_ORDER (starters then bench): {POSITION_ORDER}")
    print(f"TOTAL_ROSTER_SPOTS: {TOTAL_ROSTER_SPOTS}")

def prepare_for_ga_run():
    global CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA, CURRENT_PLAYERS_BY_POSITION_FOR_GA
    global USER_PLAYER_SLOT_ASSIGNMENTS

    CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA = [
        p for p in INITIAL_PLAYER_POOL_DATA if p[0] not in GLOBALLY_DRAFTED_PLAYER_IDS
    ]
    CURRENT_PLAYERS_BY_POSITION_FOR_GA = {}
    all_avail_positions = set(p[2] for p in CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA)
    for pos_key in all_avail_positions:
        CURRENT_PLAYERS_BY_POSITION_FOR_GA[pos_key] = [
            p for p in CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA if p[2] == pos_key
        ]
        if CURRENT_PLAYERS_BY_POSITION_FOR_GA[pos_key]:
            CURRENT_PLAYERS_BY_POSITION_FOR_GA[pos_key].sort(key=lambda x: (x[4], -x[3]))

    USER_PLAYER_SLOT_ASSIGNMENTS = {}
    available_slots_indices = list(range(TOTAL_ROSTER_SPOTS))
    sorted_user_players = sorted(USER_DRAFTED_PLAYERS_DATA, key=lambda x: x[4])
    processed_player_ids_for_assignment = set()

    for player_data in sorted_user_players:
        player_id, _, actual_player_pos, _, _ = player_data
        if player_id in processed_player_ids_for_assignment: continue
        processed_player_ids_for_assignment.add(player_id)
        assigned_this_player = False

        # Try to assign to specific starter slots first
        for slot_idx in available_slots_indices:
            slot_type = POSITION_ORDER[slot_idx]
            if not slot_type.startswith("BN_") and slot_type == actual_player_pos:
                USER_PLAYER_SLOT_ASSIGNMENTS[player_id] = slot_idx
                available_slots_indices.remove(slot_idx)
                assigned_this_player = True; break
        if assigned_this_player: continue
        # Then flex starter slots
        for slot_idx in available_slots_indices:
            slot_type = POSITION_ORDER[slot_idx]
            if not slot_type.startswith("BN_") and slot_type in FLEX_ELIGIBILITY and actual_player_pos in FLEX_ELIGIBILITY[slot_type]:
                USER_PLAYER_SLOT_ASSIGNMENTS[player_id] = slot_idx
                available_slots_indices.remove(slot_idx)
                assigned_this_player = True; break
        if assigned_this_player: continue
        # Then specific bench slots
        for slot_idx in available_slots_indices:
            slot_type = POSITION_ORDER[slot_idx]
            if slot_type.startswith("BN_") and slot_type == actual_player_pos: # e.g. BN_QB (if defined)
                 USER_PLAYER_SLOT_ASSIGNMENTS[player_id] = slot_idx
                 available_slots_indices.remove(slot_idx)
                 assigned_this_player = True; break
        if assigned_this_player: continue
        # Then flex bench slots
        for slot_idx in available_slots_indices:
            slot_type = POSITION_ORDER[slot_idx]
            if slot_type.startswith("BN_") and slot_type in FLEX_ELIGIBILITY and actual_player_pos in FLEX_ELIGIBILITY[slot_type]:
                USER_PLAYER_SLOT_ASSIGNMENTS[player_id] = slot_idx
                available_slots_indices.remove(slot_idx)
                assigned_this_player = True; break
        if not assigned_this_player:
            print(f"Warning: Could not auto-assign your drafted player {player_data[1]} ({actual_player_pos}) to any open roster slot.")
    num_ga_slots = TOTAL_ROSTER_SPOTS - len(USER_PLAYER_SLOT_ASSIGNMENTS)
    print(f"GA will attempt to fill {num_ga_slots} open roster slots.")

# --- Helper Functions ---
def get_player_data(player_id):
    return MASTER_PLAYER_ID_TO_DATA.get(player_id)

def get_player_round(player_id):
    player_data = get_player_data(player_id)
    return player_data[4] if player_data else -1

def get_slot_type_for_index(index):
    if 0 <= index < len(POSITION_ORDER):
        return POSITION_ORDER[index]
    return None

def get_eligible_players_for_slot_type_for_ga(slot_type_value):
    eligible_players = []
    processed_ids = set()
    if slot_type_value in FLEX_ELIGIBILITY:
        for actual_pos in FLEX_ELIGIBILITY[slot_type_value]:
            for player in CURRENT_PLAYERS_BY_POSITION_FOR_GA.get(actual_pos, []):
                if player[0] not in processed_ids:
                    eligible_players.append(player)
                    processed_ids.add(player[0])
    elif slot_type_value in CURRENT_PLAYERS_BY_POSITION_FOR_GA: # Handles specific positions like "QB"
        for player in CURRENT_PLAYERS_BY_POSITION_FOR_GA[slot_type_value]:
            if player[0] not in processed_ids:
                eligible_players.append(player)
                processed_ids.add(player[0])
    return eligible_players

def find_player_by_name(name_query):
    name_query_lower = name_query.lower().strip()
    exact_matches = []
    partial_matches = []
    for player_data_tuple in INITIAL_PLAYER_POOL_DATA:
        player_name_lower = player_data_tuple[1].lower().strip()
        if name_query_lower == player_name_lower:
            exact_matches.append(player_data_tuple)
    if exact_matches: return exact_matches
    for player_data_tuple in INITIAL_PLAYER_POOL_DATA:
        player_name_lower = player_data_tuple[1].lower().strip()
        normalized_player_name = player_name_lower.replace(".", "").replace("-", " ").replace("'", "")
        normalized_query = name_query_lower.replace(".", "").replace("-", " ").replace("'", "")
        if normalized_query in normalized_player_name:
            partial_matches.append(player_data_tuple)
    return partial_matches

# --- GA Core Functions ---
def create_individual():
    individual_ids = [None] * TOTAL_ROSTER_SPOTS
    used_player_ids_in_this_individual = set()
    for player_id, assigned_slot_index in USER_PLAYER_SLOT_ASSIGNMENTS.items():
        if 0 <= assigned_slot_index < TOTAL_ROSTER_SPOTS:
            individual_ids[assigned_slot_index] = player_id
            used_player_ids_in_this_individual.add(player_id)
    for i in range(TOTAL_ROSTER_SPOTS):
        if individual_ids[i] is not None: continue
        slot_type_to_fill = POSITION_ORDER[i]
        candidate_pool = get_eligible_players_for_slot_type_for_ga(slot_type_to_fill)
        possible_players = [p for p in candidate_pool if p[0] not in used_player_ids_in_this_individual]
        if possible_players:
            chosen_player_data = random.choice(possible_players)
            individual_ids[i] = chosen_player_data[0]
            used_player_ids_in_this_individual.add(chosen_player_data[0])
        else:
            individual_ids[i] = -99 # Placeholder for unfillable slot
    for i in range(len(individual_ids)): # Ensure any remaining None are also -99 for GA slots
        if individual_ids[i] is None:
             is_user_assigned_slot = any(s_idx == i for s_idx in USER_PLAYER_SLOT_ASSIGNMENTS.values())
             if not is_user_assigned_slot: individual_ids[i] = -99
    return individual_ids

def create_initial_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

def calculate_fitness(individual_ids, curr_round):
    # 1. Initial validity checks
    if not individual_ids or len(individual_ids) != TOTAL_ROSTER_SPOTS:
        return -float('inf'), 0, set(), []
    if any(pid is None or (not isinstance(pid, int)) or (pid <= 0 and pid != -99) for pid in individual_ids):
        num_truly_invalid_spots = sum(1 for pid in individual_ids if pid != -99 and (pid is None or not isinstance(pid, int) or pid <= 0))
        if num_truly_invalid_spots > 0:
            return -PENALTY_VIOLATION * (num_truly_invalid_spots + 20), 0, set(), []

    # 2. Check for empty (-99) or invalid (positive but unresolvable) STARTING slots
    #    and simultaneously build initial player data lists.
    lineup_player_objects = [None] * TOTAL_ROSTER_SPOTS # Stores p_data or None for -99 bench

    for i, pid in enumerate(individual_ids):
        slot_type = POSITION_ORDER[i]
        is_starting_slot = not slot_type.startswith("BN_")

        if pid == -99:
            if is_starting_slot:
                return -PENALTY_VIOLATION * 75, 0, set(), [] # Must-fill starter penalty!
            # lineup_player_objects[i] remains None for -99 bench, which is fine.
        else: # pid is not -99 (so should be a positive integer after initial check)
            p_data = get_player_data(pid)
            if p_data is None: # Positive PID but no data found
                if is_starting_slot:
                    return -PENALTY_VIOLATION * 65, 0, set(), [] # Invalid player in starter
                else: # Invalid player in bench slot (e.g. bad ID from repair/crossover)
                    return -PENALTY_VIOLATION * 30, 0, set(), []
            lineup_player_objects[i] = p_data

    # 3. Filter out None entries (empty bench slots) for subsequent checks
    valid_player_data_for_checks = [p_obj for p_obj in lineup_player_objects if p_obj is not None]

    # 4. Roster position validation (on actual players placed in slots)
    for i, p_data_tuple_current in enumerate(lineup_player_objects):
        if p_data_tuple_current is None: continue # Skip empty bench slots

        slot_type = POSITION_ORDER[i]
        actual_player_pos = p_data_tuple_current[2]
        is_valid_for_slot = False
        if slot_type in FLEX_ELIGIBILITY:
            if actual_player_pos in FLEX_ELIGIBILITY[slot_type]:
                is_valid_for_slot = True
        elif slot_type == actual_player_pos:
            is_valid_for_slot = True
        if not is_valid_for_slot:
            return -PENALTY_VIOLATION * 10, 0, set(), valid_player_data_for_checks

    # 5. Check for duplicate players (among valid players)
    player_id_list_for_duplicate_check = [p_data[0] for p_data in valid_player_data_for_checks]
    if len(set(player_id_list_for_duplicate_check)) != len(player_id_list_for_duplicate_check):
        return -PENALTY_VIOLATION * 5, 0, set(), valid_player_data_for_checks

    # 6. Calculate PPG components
    raw_ppg_sum_for_display = 0
    fitness_ppg_component = 0
    for i, p_data_calc in enumerate(lineup_player_objects):
        if p_data_calc is None: continue # Skip empty bench slot

        player_ppg = p_data_calc[3]
        raw_ppg_sum_for_display += player_ppg # True PPG sum

        slot_type = POSITION_ORDER[i]
        is_starting_slot_for_bonus = not slot_type.startswith("BN_")

        if is_starting_slot_for_bonus:
            fitness_ppg_component += (player_ppg * STARTER_PPG_MULTIPLIER) # Bonus for starters
        else: # Bench player
            fitness_ppg_component += player_ppg
            
    # 7. fitness_score starts with the (potentially boosted) PPG component
    fitness_score = fitness_ppg_component

    # 8. Apply ADP Round conflict penalty
    player_adp_rounds_in_lineup = [p_data[4] for p_data in valid_player_data_for_checks]
    adp_round_counts = Counter(player_adp_rounds_in_lineup)
    num_future_round_stacking_violations = 0
    for adp_round_of_player_in_lineup, count in adp_round_counts.items():
        if count > 1:
            if adp_round_of_player_in_lineup >= curr_round: # Penalize stacking from current or future ADP
                num_future_round_stacking_violations += (count - 1)
    if num_future_round_stacking_violations > 0:
        fitness_score -= (PENALTY_VIOLATION * num_future_round_stacking_violations)

    # 9. Return
    return fitness_score, raw_ppg_sum_for_display, set(player_adp_rounds_in_lineup), valid_player_data_for_checks

def repair_lineup(lineup_ids_to_repair):
    repaired_ids = list(lineup_ids_to_repair)
    if not repaired_ids or len(repaired_ids) != TOTAL_ROSTER_SPOTS:
        return create_individual()

    user_assigned_slots_indices = set(USER_PLAYER_SLOT_ASSIGNMENTS.values())
    for player_id, assigned_slot_index in USER_PLAYER_SLOT_ASSIGNMENTS.items():
        if 0 <= assigned_slot_index < TOTAL_ROSTER_SPOTS:
            if repaired_ids[assigned_slot_index] != player_id:
                for j_idx, j_pid in enumerate(repaired_ids):
                    if j_pid == player_id and j_idx != assigned_slot_index:
                        repaired_ids[j_idx] = -99
                repaired_ids[assigned_slot_index] = player_id

    # Prioritize fixing starter slots, then bench slots for GA-controlled part
    ga_controlled_starter_indices = [i for i, st in enumerate(POSITION_ORDER) if not st.startswith("BN_") and i not in user_assigned_slots_indices]
    ga_controlled_bench_indices = [i for i, st in enumerate(POSITION_ORDER) if st.startswith("BN_") and i not in user_assigned_slots_indices]
    prioritized_ga_indices_to_repair = ga_controlled_starter_indices + ga_controlled_bench_indices

    max_repair_loops = TOTAL_ROSTER_SPOTS + 5
    for _ in range(max_repair_loops):
        current_valid_pids = set(pid for pid in repaired_ids if pid is not None and pid > 0 and pid != -99)
        id_counts = Counter(current_valid_pids) # Count only valid PIDs for duplication checks
        made_change_in_pass = False

        for i in prioritized_ga_indices_to_repair:
            current_player_id_at_slot_i = repaired_ids[i]
            needs_replacement = False
            if current_player_id_at_slot_i == -99 or \
               current_player_id_at_slot_i is None or \
               (current_player_id_at_slot_i <= 0 and current_player_id_at_slot_i != -99) or \
               (current_player_id_at_slot_i != -99 and id_counts.get(current_player_id_at_slot_i, 0) > 1):
                needs_replacement = True

            if needs_replacement:
                slot_type_at_i = POSITION_ORDER[i]
                eligible_pool = get_eligible_players_for_slot_type_for_ga(slot_type_at_i)
                
                # IDs to avoid: those already validly placed in other slots
                ids_to_avoid_for_new_pick = set(current_valid_pids)
                if current_player_id_at_slot_i in ids_to_avoid_for_new_pick and id_counts.get(current_player_id_at_slot_i, 0) <=1 : # if it was a unique valid id that we are replacing for other reason
                    pass # it's fine
                elif current_player_id_at_slot_i in ids_to_avoid_for_new_pick and id_counts.get(current_player_id_at_slot_i,0)>1: # if it was a duplicate
                     pass # its count will be decremented effectively when a new player is chosen

                options = [p for p in eligible_pool if p[0] not in ids_to_avoid_for_new_pick]

                if options:
                    new_player = random.choice(options)
                    repaired_ids[i] = new_player[0]
                    # Update current_valid_pids and id_counts
                    if current_player_id_at_slot_i is not None and current_player_id_at_slot_i > 0 and current_player_id_at_slot_i != -99:
                        id_counts[current_player_id_at_slot_i] = max(0, id_counts.get(current_player_id_at_slot_i,0)-1)
                        if id_counts[current_player_id_at_slot_i] == 0 : current_valid_pids.discard(current_player_id_at_slot_i)
                    current_valid_pids.add(new_player[0])
                    id_counts[new_player[0]] = id_counts.get(new_player[0], 0) + 1
                    made_change_in_pass = True
                else: # No unique, valid player found for this slot
                    is_starting_slot_for_repair = not slot_type_at_i.startswith("BN_")
                    if repaired_ids[i] != -99: # If it wasn't already -99, make it so.
                        if current_player_id_at_slot_i is not None and current_player_id_at_slot_i > 0 and current_player_id_at_slot_i != -99:
                            id_counts[current_player_id_at_slot_i] = max(0, id_counts.get(current_player_id_at_slot_i,0)-1)
                            if id_counts[current_player_id_at_slot_i] == 0: current_valid_pids.discard(current_player_id_at_slot_i)
                        repaired_ids[i] = -99
                        made_change_in_pass = True
        
        # Break if no changes and basic validity (no duplicates among non -99)
        final_valid_pids = [pid for pid in repaired_ids if pid is not None and pid > 0 and pid != -99]
        if not made_change_in_pass and len(set(final_valid_pids)) == len(final_valid_pids):
            break
    return repaired_ids

def tournament_selection(population, fitness_scores_only):
    if not population: return create_individual()
    actual_tournament_size = min(TOURNAMENT_SIZE, len(population))
    if actual_tournament_size <= 0: return random.choice(population) if population else create_individual()
    selected_indices = random.sample(range(len(population)), actual_tournament_size)
    tournament_individuals_options = [population[i] for i in selected_indices]
    tournament_fitnesses_options = [fitness_scores_only[i] for i in selected_indices]
    return tournament_individuals_options[np.argmax(tournament_fitnesses_options)]

def crossover(parent1_ids, parent2_ids):
    child1_ids, child2_ids = list(parent1_ids), list(parent2_ids)
    if random.random() < CROSSOVER_RATE and TOTAL_ROSTER_SPOTS > 1:
        pt = random.randint(1, TOTAL_ROSTER_SPOTS - 1)
        temp_child1 = parent1_ids[:pt] + parent2_ids[pt:]
        temp_child2 = parent2_ids[:pt] + parent1_ids[pt:]
        for player_id, assigned_slot_index in USER_PLAYER_SLOT_ASSIGNMENTS.items():
            if 0 <= assigned_slot_index < TOTAL_ROSTER_SPOTS:
                temp_child1[assigned_slot_index] = player_id
                temp_child2[assigned_slot_index] = player_id
        child1_ids, child2_ids = temp_child1, temp_child2
    return repair_lineup(child1_ids), repair_lineup(child2_ids)

def mutate(individual_ids):
    mutated_ids = list(individual_ids)
    user_assigned_slots_indices = set(USER_PLAYER_SLOT_ASSIGNMENTS.values())
    if random.random() < MUTATION_RATE:
        if not mutated_ids or len(mutated_ids) != TOTAL_ROSTER_SPOTS: return repair_lineup(mutated_ids)
        ga_controlled_indices = [i for i in range(TOTAL_ROSTER_SPOTS) if i not in user_assigned_slots_indices]
        if not ga_controlled_indices: return repair_lineup(mutated_ids)
        mutation_idx = random.choice(ga_controlled_indices)
        original_player_id_at_mutation_idx = mutated_ids[mutation_idx]
        slot_type_to_mutate = POSITION_ORDER[mutation_idx]
        eligible_pool_for_mutation = get_eligible_players_for_slot_type_for_ga(slot_type_to_mutate)
        if not eligible_pool_for_mutation: return repair_lineup(mutated_ids)
        
        other_valid_pids_in_mutated = set(
            pid for idx, pid in enumerate(mutated_ids) if idx != mutation_idx and pid is not None and pid > 0 and pid != -99
        )
        options = [p for p in eligible_pool_for_mutation if p[0] != original_player_id_at_mutation_idx and p[0] not in other_valid_pids_in_mutated]
        if not options: options = [p for p in eligible_pool_for_mutation if p[0] not in other_valid_pids_in_mutated]
        if not options: options = [p for p in eligible_pool_for_mutation if p[0] != original_player_id_at_mutation_idx]
        if not options and eligible_pool_for_mutation: options = eligible_pool_for_mutation
        if options: mutated_ids[mutation_idx] = random.choice(options)[0]
    return repair_lineup(mutated_ids)

# --- Main GA Loop ---
def genetic_algorithm_adp_lineup(curr_round):
    open_ga_slots = TOTAL_ROSTER_SPOTS - len(USER_PLAYER_SLOT_ASSIGNMENTS)
    num_open_starter_slots = sum(1 for i, slot_type in enumerate(POSITION_ORDER) if not slot_type.startswith("BN_") and not any(s_idx == i for pid, s_idx in USER_PLAYER_SLOT_ASSIGNMENTS.items()))

    if open_ga_slots <= 0 :
        print("Roster is full based on your assigned players. Evaluating current team.")
    elif num_open_starter_slots == 0 and open_ga_slots > 0:
        print("All starting positions are filled by your picks. GA will optimize remaining bench spots.")
    
    if open_ga_slots <= 0:
        current_team_ids = [-99] * TOTAL_ROSTER_SPOTS # Default to -99 for empty
        for pid, slot_idx in USER_PLAYER_SLOT_ASSIGNMENTS.items(): current_team_ids[slot_idx] = pid
        fitness, ppg, _, _ = calculate_fitness(current_team_ids, curr_round)
        ppg_display = f"{ppg:.2f}" if fitness > -PENALTY_VIOLATION * 70 else "N/A (Invalid)"
        print(f"Current Team Fitness: {fitness:.2f}, True PPG: {ppg_display}")
        return current_team_ids, fitness

    if not CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA and open_ga_slots > 0:
        print("Warning: No available players in the pool for GA to select for open slots.")
        current_team_ids = [-99] * TOTAL_ROSTER_SPOTS
        for pid, slot_idx in USER_PLAYER_SLOT_ASSIGNMENTS.items(): current_team_ids[slot_idx] = pid
        fitness,ppg,_,_ = calculate_fitness(current_team_ids, curr_round)
        ppg_display = f"{ppg:.2f}" if fitness > -PENALTY_VIOLATION * 70 else "N/A (Invalid)"
        print(f"Current Team (no GA run due to empty pool): Fitness: {fitness:.2f}, True PPG: {ppg_display}")
        return current_team_ids, fitness
        
    population = create_initial_population()
    if not population or not all(ind and len(ind) == TOTAL_ROSTER_SPOTS for ind in population):
        print("Critical: Initial GA population is empty or malformed.")
        current_team_ids = [-99] * TOTAL_ROSTER_SPOTS
        for pid, slot_idx in USER_PLAYER_SLOT_ASSIGNMENTS.items(): current_team_ids[slot_idx] = pid
        fitness,ppg,_,_ = calculate_fitness(current_team_ids, curr_round)
        ppg_display = f"{ppg:.2f}" if fitness > -PENALTY_VIOLATION * 70 else "N/A (Invalid)"
        print(f"Current Team (GA init failed): Fitness: {fitness:.2f}, True PPG: {ppg_display}")
        return current_team_ids, fitness

    best_lineup_overall_ids = None
    best_fitness_overall = -float('inf')
    
    log_this_run = False # Set to True to enable CSV logging for this GA run
    csv_log_writer = None
    csv_log_file = None

    if log_this_run:
        file_exists = os.path.isfile(GA_LOG_FILENAME)
        try:
            # Open file here, keep it open for the duration of GA generations for this run
            csv_log_file = open(GA_LOG_FILENAME, mode='a', newline='', encoding='utf-8')
            csv_log_writer = csv.writer(csv_log_file)
            if not file_exists or os.path.getsize(GA_LOG_FILENAME) == 0:
                header = ["Generation", "Individual_Index", "Fitness", "TruePPG"] + \
                           [f"Slot_{j}_Player_ID" for j in range(TOTAL_ROSTER_SPOTS)]
                csv_log_writer.writerow(header)
        except IOError as e:
            print(f"Warning: Could not open GA log file {GA_LOG_FILENAME}: {e}. Logging disabled for this run.")
            log_this_run = False


    for generation in range(N_GENERATIONS):
        fitness_results = [calculate_fitness(ind, curr_round) for ind in population]
        
        if log_this_run and csv_log_writer:
            for i, individual_ids_log in enumerate(population):
                fitness_score_log = fitness_results[i][0] 
                true_ppg_score_log = fitness_results[i][1] # Log the true PPG
                logged_ids = [str(id_val) if id_val is not None else "-99" for id_val in individual_ids_log]
                row_to_log = [generation + 1, i, f"{fitness_score_log:.2f}", f"{true_ppg_score_log:.2f}"] + logged_ids
                csv_log_writer.writerow(row_to_log)

        fitness_scores_only = [res[0] for res in fitness_results] # Fitness (potentially boosted)
        current_gen_best_idx = np.argmax(fitness_scores_only)
        current_gen_best_fitness = fitness_scores_only[current_gen_best_idx]
        
        if current_gen_best_fitness > best_fitness_overall:
            best_fitness_overall = current_gen_best_fitness
            best_lineup_overall_ids = list(population[current_gen_best_idx]) 
            # best_true_ppg_overall = fitness_results[current_gen_best_idx][1] # True PPG of best

        next_population = []
        elite_count = max(1, int(0.05 * POPULATION_SIZE)) 
        sorted_indices_for_elitism = np.argsort(fitness_scores_only)[::-1] 
        elites_added_count = 0
        for elite_idx in sorted_indices_for_elitism:
            if elites_added_count >= elite_count: break
            if fitness_scores_only[elite_idx] > -PENALTY_VIOLATION * 70: # Must not have empty starters
                candidate_elite = list(population[elite_idx]) 
                if not any(Counter(existing_elite) == Counter(candidate_elite) for existing_elite in next_population):
                    next_population.append(candidate_elite)
                    elites_added_count +=1
        
        while len(next_population) < POPULATION_SIZE:
            p1 = tournament_selection(population, fitness_scores_only)
            p2 = tournament_selection(population, fitness_scores_only)
            c1, c2 = crossover(p1, p2)
            next_population.append(mutate(c1))
            if len(next_population) < POPULATION_SIZE: next_population.append(mutate(c2))
        population = next_population[:POPULATION_SIZE]

        if (generation + 1) % 20 == 0 or generation == N_GENERATIONS - 1:
            pop_best_fitness_disp = fitness_scores_only[current_gen_best_idx]
            pop_best_true_ppg_disp = fitness_results[current_gen_best_idx][1] # Display true PPG
            ppg_disp_str = f"{pop_best_true_ppg_disp:.2f}" if pop_best_fitness_disp > -PENALTY_VIOLATION * 70 else 'N/A (Inv)'
            print(f"Gen {generation+1}/{N_GENERATIONS}: Pop Best Fit={pop_best_fitness_disp:.2f}, True PPG={ppg_disp_str}")

    if log_this_run and csv_log_file and not csv_log_file.closed:
        csv_log_file.close() # Close log file at the end of GA run

    print("\n--- Genetic Algorithm Finished ---")
    if best_lineup_overall_ids and best_fitness_overall > -PENALTY_VIOLATION * 70 : 
        # Recalculate for final display, using the best lineup found
        final_fitness, final_true_points, final_rounds_set, _ = calculate_fitness(best_lineup_overall_ids, curr_round)
        
        print(f"🏆 Best Overall Lineup Found (Fitness Score: {final_fitness:.2f})")
        print(f"   Projected True PPG: {final_true_points:.2f}")
        
        print("\n📋 Best Lineup Details:")
        for i, player_id_in_slot in enumerate(best_lineup_overall_ids):
            p_data_final = get_player_data(player_id_in_slot) 
            slot_type_display = get_slot_type_for_index(i)
            is_user_pick_str = " (Your Pick)" if player_id_in_slot in USER_PLAYER_SLOT_ASSIGNMENTS and USER_PLAYER_SLOT_ASSIGNMENTS.get(player_id_in_slot) == i else ""
            
            if p_data_final:
                print(f"  Slot {i:2} ({slot_type_display:<12}): {p_data_final[1]:<25} ({p_data_final[2]:<2}) PPG: {p_data_final[3]:>5.2f} ADP Rd: {p_data_final[4]:>2}{is_user_pick_str}")
            elif player_id_in_slot == -99:
                 print(f"  Slot {i:2} ({slot_type_display:<12}): <EMPTY SLOT>")
            else:
                 print(f"  Slot {i:2} ({slot_type_display:<12}): Invalid Player ID {player_id_in_slot}")
        
        actual_ids_in_best = [pid for pid in best_lineup_overall_ids if pid is not None and pid > 0 and pid != -99]
        if len(set(actual_ids_in_best)) != len(actual_ids_in_best):
             print("🚨 WARNING: FINAL BEST LINEUP REPORTED CONTAINS DUPLICATE PLAYERS.")
        return best_lineup_overall_ids, best_fitness_overall
    else:
        print("⚠️ No significantly valid/improved solution found by GA. Consider your current team or top available players.")
        user_team_final_ids = [-99] * TOTAL_ROSTER_SPOTS
        for pid_user, slot_idx_user in USER_PLAYER_SLOT_ASSIGNMENTS.items():
            if 0 <= slot_idx_user < TOTAL_ROSTER_SPOTS: user_team_final_ids[slot_idx_user] = pid_user
        
        fit_user_final, ppg_user_final, _, _ = calculate_fitness(user_team_final_ids, curr_round)
        ppg_user_display = f"{ppg_user_final:.2f}" if fit_user_final > -PENALTY_VIOLATION * 70 else "N/A (Invalid)"
        print(f"Returning current user team state. Fitness: {fit_user_final:.2f}, True PPG: {ppg_user_display}")
        return user_team_final_ids, fit_user_final

# --- Main Interactive Live Draft Loop ---
if __name__ == "__main__":
    initial_setup(DEFAULT_CSV_FILENAME)
    if not INITIAL_PLAYER_POOL_DATA or not POSITION_ORDER or TOTAL_ROSTER_SPOTS == 0:
        print("Exiting due to critical initialization failure.")
        exit()
    
    print("\n🏈 Welcome to the Live Fantasy Football Draft Assistant! 🏈")
    print("Roster settings:")
    for slot_def, count in ROSTER_STRUCTURE.items():
        eligibility_str = f" (Eligible: {', '.join(FLEX_ELIGIBILITY[slot_def])})" if slot_def in FLEX_ELIGIBILITY else ""
        print(f"  - {slot_def}: {count}{eligibility_str}")
    print(f"Starter PPGs contribute x{STARTER_PPG_MULTIPLIER} to fitness score calculation.")
    print("-" * 30)

    while True:
        if len(GLOBALLY_DRAFTED_PLAYER_IDS) == 0: curr_round_est = 1
        else: curr_round_est = math.floor(len(GLOBALLY_DRAFTED_PLAYER_IDS) / PICKS_PER_ROUND) + 1
        max_draft_rounds = math.ceil(TOTAL_ROSTER_SPOTS * 1.2) if TOTAL_ROSTER_SPOTS > 0 else 15 
        curr_round_display = min(curr_round_est, max_draft_rounds)

        print(f"\n--- 🔔 Live Draft Options | Round {curr_round_display} (Pick {len(GLOBALLY_DRAFTED_PLAYER_IDS) + 1}) --- ")
        print("  'd <player_name_or_id>'   : Record player drafted (by another team)")
        print("  'my <player_name_or_id>'  : Record YOUR drafted player")
        print("  'undo <player_id>'        : Remove player from drafted lists")
        print("  'run'                     : Get draft suggestions (runs GA) 🚀")
        print("  'team'                    : View your current team 📊")
        print("  'drafted'                 : View all globally drafted players")
        print("  'available [pos]'         : View top available (e.g., 'available WR')")
        print("  'q'                       : Quit")
        
        action_input = input("Enter action: ").strip().lower().split()
        command = action_input[0] if action_input else ""
        args = action_input[1:]

        if command in ['d', 'my'] and args:
            query = " ".join(args); target_pid = None; target_p_data = None
            try:
                pid_candidate = int(query); p_data_cand = get_player_data(pid_candidate)
                if p_data_cand: target_pid, target_p_data = pid_candidate, p_data_cand
                else: print(f"⚠️ Player ID '{query}' not found."); continue
            except ValueError: 
                matched_players = find_player_by_name(query)
                if not matched_players: print(f"⚠️ Player '{query}' not found by name."); continue
                elif len(matched_players) == 1: target_p_data, target_pid = matched_players[0], matched_players[0][0]
                else: 
                    print(f"⚠️ Ambiguous name '{query}'. Matches:");
                    for p_id_m, name_m, pos_m, _, _ in matched_players[:5]: print(f"  ID: {p_id_m:<4} {name_m:<25} {pos_m}")
                    print(f"Use ID: '{command} <player_id>'."); continue
            if target_pid is None: continue
            if command == 'd': 
                if target_pid in GLOBALLY_DRAFTED_PLAYER_IDS: print(f"ℹ️ {target_p_data[1]} already drafted.")
                else: GLOBALLY_DRAFTED_PLAYER_IDS.add(target_pid); print(f"👍 {target_p_data[1]} marked globally drafted.")
                was_on_user = any(p[0] == target_pid for p in USER_DRAFTED_PLAYERS_DATA)
                USER_DRAFTED_PLAYERS_DATA = [p for p in USER_DRAFTED_PLAYERS_DATA if p[0] != target_pid]
                if target_pid in USER_PLAYER_SLOT_ASSIGNMENTS: del USER_PLAYER_SLOT_ASSIGNMENTS[target_pid]
                if was_on_user : print(f"   Removed from your team.")
            elif command == 'my': 
                is_on_my = any(ud_p[0] == target_pid for ud_p in USER_DRAFTED_PLAYERS_DATA)
                is_global_other = target_pid in GLOBALLY_DRAFTED_PLAYER_IDS and not is_on_my
                if is_global_other: print(f"🚫 {target_p_data[1]} already drafted by another team.")
                elif is_on_my: print(f"ℹ️ {target_p_data[1]} already on your team.")
                else: USER_DRAFTED_PLAYERS_DATA.append(target_p_data); GLOBALLY_DRAFTED_PLAYER_IDS.add(target_pid); print(f"✅ You drafted: {target_p_data[1]} ({target_p_data[2]})!")
        elif command == 'undo' and args:
            try:
                pid_undo = int(args[0]); p_data_undo = get_player_data(pid_undo)
                if not p_data_undo: print(f"⚠️ ID '{pid_undo}' not found."); continue
                actions = []
                if pid_undo in GLOBALLY_DRAFTED_PLAYER_IDS: GLOBALLY_DRAFTED_PLAYER_IDS.remove(pid_undo); actions.append("global drafted")
                user_len = len(USER_DRAFTED_PLAYERS_DATA)
                USER_DRAFTED_PLAYERS_DATA = [p for p in USER_DRAFTED_PLAYERS_DATA if p[0] != pid_undo]
                if len(USER_DRAFTED_PLAYERS_DATA) < user_len: actions.append("your team")
                if pid_undo in USER_PLAYER_SLOT_ASSIGNMENTS: del USER_PLAYER_SLOT_ASSIGNMENTS[pid_undo]
                if actions: print(f"⏪ {p_data_undo[1]} removed from: {', '.join(actions)}.")
                else: print(f"ℹ️ {p_data_undo[1]} not found in drafted lists.")
            except ValueError: print("⚠️ Invalid ID for 'undo'.")
            except Exception as e: print(f"💥 Error undoing: {e}")
        elif command == 'run':
            print("\n🔄 Preparing data for GA run..."); prepare_for_ga_run() 
            open_ga_slots_main = TOTAL_ROSTER_SPOTS - len(USER_PLAYER_SLOT_ASSIGNMENTS)
            if len(USER_PLAYER_SLOT_ASSIGNMENTS) >= TOTAL_ROSTER_SPOTS: command = 'team'; print("✅ Roster full! Showing team.")
            else: print("🧠 Running Genetic Algorithm..."); best_lineup_ids, best_fitness_val = genetic_algorithm_adp_lineup(curr_round_est)
        if command == 'team': 
            prepare_for_ga_run(); print("\n--- 📊 Your Current Team ---")
            if not USER_DRAFTED_PLAYERS_DATA: print("No players drafted yet.")
            else:
                roster_disp = ["<EMPTY SLOT>"] * TOTAL_ROSTER_SPOTS; starters_ppg_list = []; num_s, num_b = 0,0
                for pid_u, sid_u in USER_PLAYER_SLOT_ASSIGNMENTS.items():
                    pd_u = get_player_data(pid_u); slot_type_td = POSITION_ORDER[sid_u]
                    is_s_td = not slot_type_td.startswith("BN_")
                    if pd_u:
                        roster_disp[sid_u] = f"{pd_u[1]:<25} ({pd_u[2]:<2}) PPG: {pd_u[3]:>5.2f} ADP Rd: {pd_u[4]:>2}"
                        if is_s_td: starters_ppg_list.append(pd_u[3]); num_s+=1
                        else: num_b+=1
                    else: roster_disp[sid_u] = f"<UNKNOWN ID: {pid_u}>"
                total_s_slots = sum(1 for s in POSITION_ORDER if not s.startswith('BN_'))
                print(f"Starters ({num_s}/{total_s_slots} filled):")
                for i_s, s_type in enumerate(POSITION_ORDER): 
                    if not s_type.startswith("BN_"): print(f"  {i_s:2} ({s_type:<12}): {roster_disp[i_s]}")
                print(f"Total True PPG from Starters: {sum(starters_ppg_list):.2f}")
                total_b_slots = sum(1 for s in POSITION_ORDER if s.startswith('BN_'))
                print(f"\nBench ({num_b}/{total_b_slots} filled):")
                bench_empty_flag = True
                for i_b, b_type in enumerate(POSITION_ORDER):
                    if b_type.startswith("BN_"): print(f"  {i_b:2} ({b_type:<12}): {roster_disp[i_b]}"); bench_empty_flag = bench_empty_flag and roster_disp[i_b] == "<EMPTY SLOT>"
                if bench_empty_flag and total_b_slots > 0: print("  (No bench players assigned)")
                unassigned = [p for p in USER_DRAFTED_PLAYERS_DATA if p[0] not in USER_PLAYER_SLOT_ASSIGNMENTS]
                if unassigned:
                    print("\nSurplus (Drafted, Not in Roster Slots):")
                    for p_ua in sorted(unassigned, key=lambda x: (-x[3], x[4])): print(f"  - {p_ua[1]:<25} ({p_ua[2]:<2}) ID: {p_ua[0]:<4} PPG: {p_ua[3]:>5.2f} ADP Rd: {p_ua[4]:>2}")
        elif command == 'drafted':
            print(f"\n--- 📜 Globally Drafted Players ({len(GLOBALLY_DRAFTED_PLAYER_IDS)}) ---")
            if not GLOBALLY_DRAFTED_PLAYER_IDS: print("None yet.")
            else:
                drafted_disp = sorted([(get_player_data(pid_g), any(ud_p[0] == pid_g for ud_p in USER_DRAFTED_PLAYERS_DATA)) for pid_g in GLOBALLY_DRAFTED_PLAYER_IDS if get_player_data(pid_g)], key=lambda x: (x[0][4], x[0][0]))
                for pd_g_s, is_y_s in drafted_disp: print(f"Rd {pd_g_s[4]:>2}: {pd_g_s[1]:<25} ({pd_g_s[2]:<2}, ID: {pd_g_s[0]:<4}){' (Your Pick)' if is_y_s else ''}")
        elif command == 'available':
            avail_pool = sorted([p for p in INITIAL_PLAYER_POOL_DATA if p[0] not in GLOBALLY_DRAFTED_PLAYER_IDS], key=lambda x: (-x[3], x[4]))
            filter_pos = args[0].upper() if args else None
            print(f"\n--- ⭐ Top Available {('(' + filter_pos + ')' if filter_pos else '(All Pos)')} ---")
            count_disp = 0; limit_disp = 15 if filter_pos else 30
            for p_av in avail_pool:
                if filter_pos and p_av[2] != filter_pos: continue 
                print(f"{p_av[1]:<25} ({p_av[2]:<2}, ID: {p_av[0]:<4}) PPG: {p_av[3]:>5.2f} ADP Rd: {p_av[4]:>2}")
                count_disp += 1; 
                if count_disp >= limit_disp : break 
            if count_disp == 0: print(f"None available {'for ' + filter_pos if filter_pos else 'at all'}.")
        elif command == 'q': print("👋 Exiting. Good luck!"); break
        elif command == "": continue
        else: print(f"❓ Unknown command: '{command}'.")