import random
import math
import numpy as np
from collections import Counter
import csv

# --- Configuration Constants ---
GAMES_IN_SEASON = 17
PICKS_PER_ROUND = 12
DEFAULT_CSV_FILENAME = "player_pool.csv" # Make sure this CSV exists or is updated

# --- Roster Configuration ---
FLEX_ELIGIBILITY = {
    "W/R/T": ("WR", "RB", "TE"),
    "W/R": ("WR", "RB"),
    "R/T": ("RB", "TE"),
    "SUPERFLEX": ("QB", "WR", "RB", "TE")
}

ROSTER_STRUCTURE = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "W/R/T": 1
}

# --- Global Variables ---
# Initial State (Loaded once)
INITIAL_PLAYER_POOL_DATA = []
MASTER_PLAYER_ID_TO_DATA = {} # For all players ever loaded

# Draft State (Updated during live draft)
GLOBALLY_DRAFTED_PLAYER_IDS = set()
USER_DRAFTED_PLAYERS_DATA = [] # List of player data tuples for your team

# GA Run State (Rebuilt before each GA run)
CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA = []
CURRENT_PLAYERS_BY_POSITION_FOR_GA = {} # Based on CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA
USER_PLAYER_SLOT_ASSIGNMENTS = {} # {player_id: assigned_slot_index} for your drafted players

# Static Roster Definition (Set once by initial_setup)
POSITION_ORDER = []
TOTAL_ROSTER_SPOTS = 0

# --- GA Parameters ---
POPULATION_SIZE = 100
N_GENERATIONS = 100
MUTATION_RATE = 0.20
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5
PENALTY_VIOLATION = 10000

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
                    ppg = total_points / GAMES_IN_SEASON
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

    INITIAL_PLAYER_POOL_DATA = load_player_pool_from_csv(csv_filename)
    if not INITIAL_PLAYER_POOL_DATA:
        print("Critical Error: Initial player pool is empty. Exiting.")
        exit()

    MASTER_PLAYER_ID_TO_DATA = {p[0]: p for p in INITIAL_PLAYER_POOL_DATA}

    temp_position_order = []
    for slot_type, count in ROSTER_STRUCTURE.items():
        for _ in range(count):
            temp_position_order.append(slot_type)
    POSITION_ORDER = temp_position_order
    TOTAL_ROSTER_SPOTS = len(POSITION_ORDER)

    unique_initial_positions = set(p[2] for p in INITIAL_PLAYER_POOL_DATA)
    for slot_key in ROSTER_STRUCTURE.keys():
        if slot_key not in FLEX_ELIGIBILITY and slot_key not in unique_initial_positions:
            print(f"Warning: Roster slot key '{slot_key}' not in FLEX_ELIGIBILITY or player positions.")
    for flex_key, eligible_list in FLEX_ELIGIBILITY.items():
        for pos in eligible_list:
            if pos not in unique_initial_positions:
                 print(f"Warning: Position '{pos}' in FLEX_ELIGIBILITY for '{flex_key}' not in player positions.")
    print("Initial data loaded.")
    print(f"Full POSITION_ORDER: {POSITION_ORDER}")
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
            CURRENT_PLAYERS_BY_POSITION_FOR_GA[pos_key].sort(key=lambda x: x[4])

    USER_PLAYER_SLOT_ASSIGNMENTS = {}
    available_slots_indices = list(range(TOTAL_ROSTER_SPOTS))
    sorted_user_players = sorted(USER_DRAFTED_PLAYERS_DATA, key=lambda x: x[4]) # Sort by round

    for player_data in sorted_user_players:
        player_id, _, actual_player_pos, _, _ = player_data
        assigned_this_player = False
        # Try to assign to specific position slot first (iterate over a copy for safe removal)
        for slot_idx in list(available_slots_indices):
            slot_type = POSITION_ORDER[slot_idx]
            if slot_type == actual_player_pos:
                USER_PLAYER_SLOT_ASSIGNMENTS[player_id] = slot_idx
                available_slots_indices.remove(slot_idx)
                assigned_this_player = True
                break
        if assigned_this_player: continue
        # Then try to assign to flex slot (iterate over a copy)
        for slot_idx in list(available_slots_indices):
            slot_type = POSITION_ORDER[slot_idx]
            if slot_type in FLEX_ELIGIBILITY and actual_player_pos in FLEX_ELIGIBILITY[slot_type]:
                USER_PLAYER_SLOT_ASSIGNMENTS[player_id] = slot_idx
                available_slots_indices.remove(slot_idx)
                assigned_this_player = True
                break
        if not assigned_this_player:
            print(f"Warning: Could not auto-assign {player_data[1]} ({actual_player_pos}) to a roster slot.")
    
    print(f"User players assigned to slots: {USER_PLAYER_SLOT_ASSIGNMENTS}")
    num_ga_slots = TOTAL_ROSTER_SPOTS - len(USER_PLAYER_SLOT_ASSIGNMENTS)
    print(f"GA will attempt to fill {num_ga_slots} open slots.")

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
    elif slot_type_value in CURRENT_PLAYERS_BY_POSITION_FOR_GA:
        eligible_players = CURRENT_PLAYERS_BY_POSITION_FOR_GA[slot_type_value]
    return eligible_players

# --- GA Core Functions (Modified for Live Draft) ---

def create_individual():
    individual_ids = [None] * TOTAL_ROSTER_SPOTS
    used_player_ids = set()
    used_rounds = set()

    for player_id, assigned_slot_index in USER_PLAYER_SLOT_ASSIGNMENTS.items():
        if 0 <= assigned_slot_index < TOTAL_ROSTER_SPOTS:
            individual_ids[assigned_slot_index] = player_id
            player_data = get_player_data(player_id)
            if player_data:
                used_player_ids.add(player_data[0])
                used_rounds.add(player_data[4])
    
    for i in range(TOTAL_ROSTER_SPOTS):
        if individual_ids[i] is not None: continue

        slot_type_to_fill = POSITION_ORDER[i]
        candidate_pool = get_eligible_players_for_slot_type_for_ga(slot_type_to_fill)
        
        if not candidate_pool:
            individual_ids[i] = -99; continue

        possible_players = [p for p in candidate_pool if p[0] not in used_player_ids and p[4] not in used_rounds]
        
        chosen_player_data = None
        if possible_players:
            chosen_player_data = random.choice(possible_players)
        else:
            possible_players_no_round = [p for p in candidate_pool if p[0] not in used_player_ids]
            if possible_players_no_round:
                chosen_player_data = random.choice(possible_players_no_round)
        
        if chosen_player_data:
            individual_ids[i] = chosen_player_data[0]
            used_player_ids.add(chosen_player_data[0])
            used_rounds.add(chosen_player_data[4])
        else:
            individual_ids[i] = -1

    for i in range(len(individual_ids)):
        if individual_ids[i] is None or individual_ids[i] <= 0:
            is_user_slot = any(s_idx == i for s_idx in USER_PLAYER_SLOT_ASSIGNMENTS.values())
            if not is_user_slot:
                slot_type_fallback = POSITION_ORDER[i]
                fallback_pool = get_eligible_players_for_slot_type_for_ga(slot_type_fallback)
                if fallback_pool:
                    options = [p for p in fallback_pool if p[0] not in used_player_ids]
                    if not options and fallback_pool: options = fallback_pool 
                    if options: individual_ids[i] = random.choice(options)[0]
                    else: individual_ids[i] = -99
                else: individual_ids[i] = -99
    return individual_ids

def create_initial_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

def calculate_fitness(individual_ids):
    total_ppg = 0
    if not individual_ids or len(individual_ids) != TOTAL_ROSTER_SPOTS or \
       any(pid is None or not isinstance(pid, int) or pid <= 0 for pid in individual_ids):
        return -float('inf'), 0, set(), []

    lineup_players_data = []
    for pid in individual_ids:
        p_data = get_player_data(pid)
        if p_data is None: return -float('inf'), 0, set(), []
        lineup_players_data.append(p_data)

    for i, p_data_tuple in enumerate(lineup_players_data):
        slot_type = POSITION_ORDER[i]
        actual_player_pos = p_data_tuple[2]
        is_valid = False
        if slot_type in FLEX_ELIGIBILITY:
            if actual_player_pos in FLEX_ELIGIBILITY[slot_type]: is_valid = True
        elif slot_type == actual_player_pos:
            is_valid = True
        if not is_valid: return -PENALTY_VIOLATION * 10, 0, set(), lineup_players_data

    actual_player_count = sum(1 for pid in individual_ids if pid is not None and pid > 0)
    player_id_set = set(p_data[0] for p_data in lineup_players_data) # lineup_players_data only contains valid players
    if len(player_id_set) != actual_player_count: # Check if unique IDs match count of actual players
        return -PENALTY_VIOLATION * 5, 0, set(), lineup_players_data

    player_rounds = [p_data[4] for p_data in lineup_players_data]
    round_counts = Counter(player_rounds)
    round_violations = sum(count - 1 for count in round_counts.values() if count > 1)
    if round_violations > 0:
        return -PENALTY_VIOLATION * round_violations, 0, set(player_rounds), lineup_players_data

    for p_data in lineup_players_data: total_ppg += p_data[3]
    return total_ppg, total_ppg, set(player_rounds), lineup_players_data

def repair_lineup(lineup_ids_to_repair):
    repaired_ids = list(lineup_ids_to_repair)
    if not repaired_ids or len(repaired_ids) != TOTAL_ROSTER_SPOTS:
        return create_individual()

    user_assigned_slots_indices = set(USER_PLAYER_SLOT_ASSIGNMENTS.values())
    max_repair_iterations = TOTAL_ROSTER_SPOTS * 2

    for iteration in range(max_repair_iterations):
        id_counts = Counter(pid for pid in repaired_ids if pid is not None and pid > 0)
        made_change_player = False
        for i in range(TOTAL_ROSTER_SPOTS):
            if i in user_assigned_slots_indices: continue
            current_player_id = repaired_ids[i]
            slot_type = POSITION_ORDER[i]
            if current_player_id is None or current_player_id <= 0 or id_counts.get(current_player_id, 0) > 1:
                eligible_pool = get_eligible_players_for_slot_type_for_ga(slot_type)
                if not eligible_pool: repaired_ids[i] = -99; continue
                current_lineup_ids_for_check = set(pid for pid in repaired_ids if pid is not None and pid > 0 and pid != current_player_id)
                options = [p for p in eligible_pool if p[0] not in current_lineup_ids_for_check]
                if options:
                    new_player = random.choice(options)
                    if current_player_id is not None and current_player_id > 0: id_counts[current_player_id] -=1
                    repaired_ids[i] = new_player[0]
                    id_counts[new_player[0]] = id_counts.get(new_player[0],0) +1
                    made_change_player = True
                elif eligible_pool:
                    new_player = random.choice(eligible_pool)
                    if current_player_id is not None and current_player_id > 0: id_counts[current_player_id] -=1
                    repaired_ids[i] = new_player[0]
                    id_counts[new_player[0]] = id_counts.get(new_player[0],0) +1
                    made_change_player = True
                else: repaired_ids[i] = -99
        if not made_change_player: break
    
    for i in range(TOTAL_ROSTER_SPOTS):
        if i in user_assigned_slots_indices: continue
        if repaired_ids[i] is None or repaired_ids[i] <=0:
            slot_type = POSITION_ORDER[i]
            pool = get_eligible_players_for_slot_type_for_ga(slot_type)
            if pool: 
                temp_ids_for_fill = set(pid for pid in repaired_ids if pid is not None and pid > 0)
                options = [p for p in pool if p[0] not in temp_ids_for_fill]
                if not options: options = pool
                if options: repaired_ids[i] = random.choice(options)[0]
                else: repaired_ids[i] = -99
            else: repaired_ids[i] = -99

    for iteration in range(max_repair_iterations):
        current_player_details = []
        all_ids_valid_for_round_check = True
        for idx, pid in enumerate(repaired_ids):
            p_data = get_player_data(pid)
            if not p_data:
                all_ids_valid_for_round_check = False; break
            current_player_details.append({'id':pid, 'round':p_data[4], 'slot_idx':idx, 'is_user_pick': (idx in user_assigned_slots_indices)})
        if not all_ids_valid_for_round_check: break

        round_counts = Counter(detail['round'] for detail in current_player_details)
        made_change_round = False
        for i in range(TOTAL_ROSTER_SPOTS):
            player_at_slot_i = current_player_details[i]
            if player_at_slot_i['is_user_pick']: continue
            current_round_of_player_i = player_at_slot_i['round']
            if round_counts[current_round_of_player_i] <= 1: continue
            slot_type = POSITION_ORDER[i]
            eligible_pool_for_round_fix = get_eligible_players_for_slot_type_for_ga(slot_type)
            other_players_rounds = set()
            other_player_ids = set()
            for detail in current_player_details:
                if detail['slot_idx'] == i: continue
                other_players_rounds.add(detail['round'])
                other_player_ids.add(detail['id'])
            options = [p for p in eligible_pool_for_round_fix if p[0] not in other_player_ids and p[4] not in other_players_rounds]
            if options:
                new_player = random.choice(options)
                round_counts[current_round_of_player_i] -= 1
                repaired_ids[i] = new_player[0]
                round_counts[new_player[4]] = round_counts.get(new_player[4],0) + 1
                player_at_slot_i['id'] = new_player[0]
                player_at_slot_i['round'] = new_player[4]
                made_change_round = True
        if not made_change_round: break
    return repaired_ids

def tournament_selection(population, fitness_scores_only):
    if not population: return create_individual()
    actual_tournament_size = min(TOURNAMENT_SIZE, len(population))
    if actual_tournament_size <= 0 : return random.choice(population) if population else create_individual()
    selected_indices = random.sample(range(len(population)), actual_tournament_size)
    tournament_individuals = [population[i] for i in selected_indices]
    tournament_fitnesses = [fitness_scores_only[i] for i in selected_indices]
    return tournament_individuals[np.argmax(tournament_fitnesses)]

def crossover(parent1_ids, parent2_ids):
    child1_ids, child2_ids = list(parent1_ids), list(parent2_ids)
    if random.random() < CROSSOVER_RATE and len(parent1_ids) > 1:
        pt = random.randint(1, TOTAL_ROSTER_SPOTS - 1)
        child1_ids_temp = parent1_ids[:pt] + parent2_ids[pt:]
        child2_ids_temp = parent2_ids[:pt] + parent1_ids[pt:]
        for pid, slot_idx in USER_PLAYER_SLOT_ASSIGNMENTS.items():
            child1_ids_temp[slot_idx] = pid
            child2_ids_temp[slot_idx] = pid
        child1_ids, child2_ids = child1_ids_temp, child2_ids_temp
    return repair_lineup(child1_ids), repair_lineup(child2_ids)

def mutate(individual_ids):
    mutated_ids = list(individual_ids)
    user_assigned_slots_indices = set(USER_PLAYER_SLOT_ASSIGNMENTS.values())
    if random.random() < MUTATION_RATE:
        if not mutated_ids or len(mutated_ids) != TOTAL_ROSTER_SPOTS: return repair_lineup(mutated_ids)
        ga_controlled_indices = [i for i in range(TOTAL_ROSTER_SPOTS) if i not in user_assigned_slots_indices]
        if not ga_controlled_indices: return repair_lineup(mutated_ids)
        mutation_idx = random.choice(ga_controlled_indices)
        original_player_id_at_idx = mutated_ids[mutation_idx]
        slot_type_to_mutate = POSITION_ORDER[mutation_idx]
        eligible_pool = get_eligible_players_for_slot_type_for_ga(slot_type_to_mutate)
        if not eligible_pool: return repair_lineup(mutated_ids)
        
        other_rounds, other_ids = set(), set()
        original_round = get_player_round(original_player_id_at_idx) if original_player_id_at_idx and original_player_id_at_idx > 0 else -1

        for i in range(TOTAL_ROSTER_SPOTS):
            if i == mutation_idx: continue
            pid = mutated_ids[i]
            if pid and pid > 0:
                p_data = get_player_data(pid)
                if p_data: other_rounds.add(p_data[4]); other_ids.add(p_data[0])
        
        options = [p for p in eligible_pool if p[0] not in other_ids and \
                   (p[4] not in other_rounds or (original_round != -1 and p[4] == original_round))]
        if not options: options = [p for p in eligible_pool if p[0] not in other_ids and p[0] != original_player_id_at_idx]
        if not options: options = [p for p in eligible_pool if p[0] != original_player_id_at_idx] # Ensure it's not the same player ID
        if not options and eligible_pool: options = eligible_pool # Last resort: pick any if all above fail

        if options: mutated_ids[mutation_idx] = random.choice(options)[0]
    return repair_lineup(mutated_ids)

# --- Main GA Loop ---
def genetic_algorithm_adp_lineup():
    open_ga_slots = TOTAL_ROSTER_SPOTS - len(USER_PLAYER_SLOT_ASSIGNMENTS)
    if open_ga_slots <= 0 :
        print("Roster is effectively full with your picks. Evaluating current team.")
        current_team_ids = [None] * TOTAL_ROSTER_SPOTS
        for pid, slot_idx in USER_PLAYER_SLOT_ASSIGNMENTS.items(): current_team_ids[slot_idx] = pid
        current_team_ids = [-1 if x is None else x for x in current_team_ids]
        fitness, ppg, _, _ = calculate_fitness(current_team_ids)
        ppg_display = f"{ppg:.2f}" if fitness > -PENALTY_VIOLATION else "N/A"
        print(f"Current Team Fitness: {fitness:.2f}, PPG: {ppg_display}")
        return current_team_ids, fitness

    if not CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA and open_ga_slots > 0:
        print("Warning: No available players for GA to select for open slots.")
        current_team_ids = [None] * TOTAL_ROSTER_SPOTS
        for pid, slot_idx in USER_PLAYER_SLOT_ASSIGNMENTS.items(): current_team_ids[slot_idx] = pid
        current_team_ids = [-1 if x is None else x for x in current_team_ids]
        fitness,_,_,_ = calculate_fitness(current_team_ids)
        return current_team_ids, fitness
        
    population = create_initial_population() # Call the defined function
    if not population or not all(ind and len(ind) == TOTAL_ROSTER_SPOTS for ind in population):
        print("Critical: Initial GA population is empty or malformed.")
        current_team_ids = [None] * TOTAL_ROSTER_SPOTS
        for pid, slot_idx in USER_PLAYER_SLOT_ASSIGNMENTS.items(): current_team_ids[slot_idx] = pid
        current_team_ids = [-1 if x is None else x for x in current_team_ids]
        fitness,_,_,_ = calculate_fitness(current_team_ids)
        return current_team_ids, fitness
    print(f"Initial GA population of {len(population)} lineups created.")

    best_lineup_overall_ids = None
    best_fitness_overall = -float('inf')

    for generation in range(N_GENERATIONS):
        fitness_results = [calculate_fitness(ind) for ind in population]
        fitness_scores_only = [res[0] for res in fitness_results]
        current_best_idx = np.argmax(fitness_scores_only)
        current_best_fitness = fitness_scores_only[current_best_idx]
        
        if current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_lineup_overall_ids = list(population[current_best_idx])
            best_ppg = fitness_results[current_best_idx][1] # res[1] is total_ppg

            # Corrected f-string logic
            ppg_display_string = f"{best_ppg:.2f}" if best_fitness_overall > -PENALTY_VIOLATION else "N/A"
            print(f"Gen {generation+1}: New Global Best! Fit={best_fitness_overall:.2f}, PPG={ppg_display_string}")

        next_population = []
        elite_count = max(1, int(0.05 * POPULATION_SIZE))
        sorted_indices = np.argsort(fitness_scores_only)[::-1]
        
        elites_added = 0
        for i in range(len(sorted_indices)):
            if elites_added >= elite_count: break
            idx = sorted_indices[i]
            if fitness_scores_only[idx] > -PENALTY_VIOLATION * 2:
                candidate = list(population[idx])
                if not any(Counter(ex_el) == Counter(candidate) for ex_el in next_population):
                    next_population.append(candidate)
                    elites_added +=1
        
        while len(next_population) < POPULATION_SIZE:
            p1 = tournament_selection(population, fitness_scores_only)
            p2 = tournament_selection(population, fitness_scores_only)
            c1, c2 = crossover(p1, p2)
            next_population.append(mutate(c1))
            if len(next_population) < POPULATION_SIZE: next_population.append(mutate(c2))
        population = next_population[:POPULATION_SIZE]

        if (generation + 1) % 10 == 0 or generation == N_GENERATIONS - 1:
            gen_fit_p, gen_ppg_p, gen_rds_p, _ = fitness_results[current_best_idx] # Use current_best_idx from this gen

            # Corrected f-string logic
            ppg_display_periodic = f"{gen_ppg_p:.2f}" if gen_fit_p > -PENALTY_VIOLATION else 'N/A'
            rounds_display_periodic = str(len(gen_rds_p)) if gen_fit_p > -PENALTY_VIOLATION else 'N/A' # Ensure string for N/A
            print(f"Gen {generation+1}: Pop Best Fit={gen_fit_p:.2f}, PPG={ppg_display_periodic}, Rds={rounds_display_periodic}")

    print("\n--- Genetic Algorithm Finished ---")
    if best_lineup_overall_ids and best_fitness_overall > -PENALTY_VIOLATION * 1.5 :
        final_fitness, final_points, final_rounds, final_data_tuples = calculate_fitness(best_lineup_overall_ids)
        print(f"Best Lineup (IDs): {best_lineup_overall_ids}")
        print(f"Stats: Fitness={final_fitness:.2f}, Total PPG={final_points:.2f}, Rounds Used={len(final_rounds)}")
        print("\nBest Lineup Details:")
        if final_data_tuples and len(final_data_tuples) == TOTAL_ROSTER_SPOTS:
            for i, p_id_in_slot in enumerate(best_lineup_overall_ids):
                p_dat_final = get_player_data(p_id_in_slot)
                slot_type_disp_final = get_slot_type_for_index(i)
                is_user_pick_str = " (Your Pick)" if p_id_in_slot in USER_PLAYER_SLOT_ASSIGNMENTS and USER_PLAYER_SLOT_ASSIGNMENTS.get(p_id_in_slot) == i else ""
                if p_dat_final:
                    print(f"  - Slot {i} ({slot_type_disp_final}): {p_dat_final[1]} ({p_dat_final[2]}, ID:{p_dat_final[0]}){is_user_pick_str} - PPG:{p_dat_final[3]:.2f}, Rd:{p_dat_final[4]}")
                else: print(f"  - Slot {i} ({slot_type_disp_final}): Invalid Player ID {p_id_in_slot}")
            
            actual_ids_in_best = [pid for pid in best_lineup_overall_ids if pid is not None and pid > 0]
            if len(set(actual_ids_in_best)) != len(actual_ids_in_best):
                 print("WARNING: FINAL BEST LINEUP CONTAINS DUPLICATE PLAYERS.")
        return best_lineup_overall_ids, best_fitness_overall
    else:
        print("No significantly valid solution found by GA. Consider your current team or top available.")
        user_team_final = [None] * TOTAL_ROSTER_SPOTS
        for pid_u, slot_idx_u in USER_PLAYER_SLOT_ASSIGNMENTS.items(): user_team_final[slot_idx_u] = pid_u
        user_team_final = [-1 if x is None else x for x in user_team_final]
        fit_u_final,_,_,_ = calculate_fitness(user_team_final)
        print(f"Returning current user team state. Fitness: {fit_u_final:.2f}, IDs: {user_team_final}")
        return user_team_final, fit_u_final

# --- Main Interactive Live Draft Loop ---
if __name__ == "__main__":
    initial_setup(DEFAULT_CSV_FILENAME)
    if not INITIAL_PLAYER_POOL_DATA or not POSITION_ORDER:
        print("Exiting due to initialization failure.")
        exit()
    
    print("\nWelcome to the Live Fantasy Football Draft Assistant!")
    print("Roster settings:")
    for slot_def, count in ROSTER_STRUCTURE.items():
        eligibility = f" (Eligible: {', '.join(FLEX_ELIGIBILITY[slot_def])})" if slot_def in FLEX_ELIGIBILITY else ""
        print(f"  - {slot_def}: {count}{eligibility}")
    print("-" * 30)

    while True:
        print("\n--- Live Draft Options ---")
        print("  'd <player_id>'   : Record player drafted globally")
        print("  'my <player_id>'  : Record YOUR drafted player")
        print("  'run'             : Get new draft suggestions (runs GA)")
        print("  'team'            : View your current team slots")
        print("  'drafted'         : View all globally drafted players")
        print("  'available [pos]' : View top available players (optional: by position filter, e.g. 'available WR')")
        print("  'q'               : Quit")
        
        action_input = input("Enter action: ").strip().lower().split()
        command = action_input[0] if action_input else ""

        if command == 'd' and len(action_input) > 1:
            try:
                pid = int(action_input[1])
                p_data = get_player_data(pid)
                if p_data:
                    GLOBALLY_DRAFTED_PLAYER_IDS.add(pid)
                    USER_DRAFTED_PLAYERS_DATA = [p for p in USER_DRAFTED_PLAYERS_DATA if p[0] != pid]
                    if pid in USER_PLAYER_SLOT_ASSIGNMENTS:
                        del USER_PLAYER_SLOT_ASSIGNMENTS[pid]
                    print(f"Player {p_data[1]} (ID: {pid}) marked globally drafted.")
                else: print(f"Player ID {pid} not found.")
            except ValueError: print("Invalid Player ID.")

        elif command == 'my' and len(action_input) > 1:
            try:
                pid = int(action_input[1])
                p_data = get_player_data(pid)
                if p_data:
                    if pid not in GLOBALLY_DRAFTED_PLAYER_IDS:
                        if not any(ud_p[0] == pid for ud_p in USER_DRAFTED_PLAYERS_DATA):
                             USER_DRAFTED_PLAYERS_DATA.append(p_data)
                        GLOBALLY_DRAFTED_PLAYER_IDS.add(pid)
                        print(f"You drafted: {p_data[1]} ({p_data[2]})")
                    else: print(f"Player {p_data[1]} (ID: {pid}) already drafted.")
                else: print(f"Player ID {pid} not found.")
            except ValueError: print("Invalid Player ID.")

        elif command == 'run':
            print("\nPreparing data for GA run...")
            prepare_for_ga_run()
            
            if len(USER_PLAYER_SLOT_ASSIGNMENTS) >= TOTAL_ROSTER_SPOTS:
                print("Your roster is full based on assigned players! Showing current team.")
                command = 'team' 
            else:
                print("Running Genetic Algorithm for suggestions...")
                best_lineup_ids, best_fitness_val = genetic_algorithm_adp_lineup()

        if command == 'team': 
            print("\n--- Your Current Team (Assigned Slots) ---")
            if not USER_DRAFTED_PLAYERS_DATA and not USER_PLAYER_SLOT_ASSIGNMENTS:
                print("You haven't drafted any players or no players are assigned slots.")
            else:
                display_roster = ["<EMPTY>"] * TOTAL_ROSTER_SPOTS
                for p_id_user, slot_idx_user in USER_PLAYER_SLOT_ASSIGNMENTS.items():
                    p_d_user = get_player_data(p_id_user)
                    if p_d_user: display_roster[slot_idx_user] = f"{p_d_user[1]} ({p_d_user[2]})"
                
                for i_slot_disp in range(TOTAL_ROSTER_SPOTS):
                    slot_type_disp = POSITION_ORDER[i_slot_disp]
                    print(f"Slot {i_slot_disp} ({slot_type_disp}): {display_roster[i_slot_disp]}")

                unassigned_user_players = [p_un for p_un in USER_DRAFTED_PLAYERS_DATA if p_un[0] not in USER_PLAYER_SLOT_ASSIGNMENTS]
                if unassigned_user_players:
                    print("\nUser-Drafted Players Not Assigned to Specific Slots (e.g., Bench/Surplus):")
                    for p_d_unassigned in unassigned_user_players:
                        print(f" - {p_d_unassigned[1]} ({p_d_unassigned[2]}) - ID: {p_d_unassigned[0]}")

        elif command == 'drafted':
            print("\n--- Globally Drafted Players ---")
            if not GLOBALLY_DRAFTED_PLAYER_IDS: print("None yet.")
            else:
                sorted_drafted_ids = sorted(list(GLOBALLY_DRAFTED_PLAYER_IDS))
                for pid_glob in sorted_drafted_ids:
                    p_d_glob = get_player_data(pid_glob)
                    is_yours_str = " (Your Pick)" if any(ud_p[0] == pid_glob for ud_p in USER_DRAFTED_PLAYERS_DATA) else ""
                    if p_d_glob: print(f"- {p_d_glob[1]} ({p_d_glob[2]}, ID: {pid_glob}){is_yours_str}")
        
        elif command == 'available':
            temp_available_pool_view = [p for p in INITIAL_PLAYER_POOL_DATA if p[0] not in GLOBALLY_DRAFTED_PLAYER_IDS]
            temp_available_pool_view.sort(key=lambda x: x[3], reverse=True)

            filter_pos_view = action_input[1].upper() if len(action_input) > 1 else None
            
            print("\n--- Top Available Players ---")
            count_view = 0
            for p_avail_view in temp_available_pool_view:
                if filter_pos_view and p_avail_view[2] != filter_pos_view: continue
                print(f"- {p_avail_view[1]} ({p_avail_view[2]}, ID: {p_avail_view[0]}) - PPG: {p_avail_view[3]:.2f}, Rd: {p_avail_view[4]}")
                count_view += 1
                if count_view >= 20 and not filter_pos_view : break
                if count_view >=10 and filter_pos_view: break
            if count_view == 0: print("None available for this filter.")

        elif command == 'q':
            print("Exiting live draft tool.")
            break
        elif command == "": continue
        else:
            if command not in ['run', 'team']:
                 print("Invalid command.")