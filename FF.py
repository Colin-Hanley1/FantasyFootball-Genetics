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
    "W/R/T": ("WR", "RB", "TE"),      # Standard WR/RB/TE Flex
    "W/R": ("WR", "RB"),              # WR/RB Flex
    "R/T": ("RB", "TE"),              # RB/TE Flex
    "SUPERFLEX": ("QB", "WR", "RB", "TE") # Allows QB in a flex spot
    # Add other custom flex types here, e.g., "RB/TE": ("RB", "TE")
}

# Define your roster using basic positions and keys from FLEX_ELIGIBILITY
# Example: 1 QB, 2 RB, 2 WR, 1 TE, 1 W/R/T Flex
ROSTER_STRUCTURE = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "W/R/T": 1  # This key "W/R/T" must exist in FLEX_ELIGIBILITY
    # If you want more positions like DST, add "DST":1 and ensure "DST" players are in your CSV
    # and PLAYERS_BY_POSITION gets populated with "DST".
}

# --- Global Variables (to be initialized) ---
PLAYER_POOL = []
PLAYERS_BY_POSITION = {}  # Stores lists of players for each *basic* position (QB, RB, WR, TE)
PLAYER_ID_TO_DATA = {}
POSITION_ORDER = []       # Defines the type of each slot in the roster (e.g., "QB", "W/R/T")
TOTAL_ROSTER_SPOTS = 0

# --- 0. Define Player Data and Roster Constraints ---

def load_player_pool_from_csv(filename=DEFAULT_CSV_FILENAME):
    player_pool_list = []
    expected_headers = ["ID", "Name", "Position", "TotalPoints", "ADP"]
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames or not all(field in reader.fieldnames for field in expected_headers):
                missing_headers = [h for h in expected_headers if h not in (reader.fieldnames or [])]
                print(f"Error: CSV file '{filename}' is missing required headers: {', '.join(missing_headers)} (or CSV is empty).")
                print(f"Expected headers are: {', '.join(expected_headers)}")
                return []

            for row_num, row in enumerate(reader, 1):
                try:
                    player_id = int(row["ID"])
                    name = row["Name"]
                    position = row["Position"].upper() # Standardize position to uppercase
                    total_points = float(row["TotalPoints"])
                    ppg = total_points / GAMES_IN_SEASON
                    adp = int(row["ADP"])
                    calculated_round = max(1, math.ceil(adp / PICKS_PER_ROUND))
                    player_pool_list.append((player_id, name, position, ppg, calculated_round))
                except ValueError as e:
                    print(f"Warning: Skipping row {row_num} in {filename} due to data conversion error: {e}. Row content: {row}")
                except KeyError as e:
                    print(f"Warning: Skipping row {row_num} in {filename} due to missing column: {e}. Row content: {row}")
                except ZeroDivisionError:
                    print(f"Warning: Skipping row {row_num} in {filename} due to ZeroDivisionError. Row: {row}")
    except FileNotFoundError:
        print(f"Error: Player pool CSV file '{filename}' not found.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading '{filename}': {e}")
        return []

    if not player_pool_list:
        print(f"Warning: No players loaded from '{filename}'. The CSV might be empty or all rows had errors.")
    return player_pool_list

def initialize_global_data(csv_filename=DEFAULT_CSV_FILENAME):
    global PLAYER_POOL, PLAYERS_BY_POSITION, PLAYER_ID_TO_DATA, POSITION_ORDER, TOTAL_ROSTER_SPOTS

    PLAYER_POOL = load_player_pool_from_csv(csv_filename)
    if not PLAYER_POOL:
        print("Critical Error: Player pool is empty after loading. Exiting.")
        exit()

    all_actual_positions = set(p[2] for p in PLAYER_POOL) # p[2] is actual player position
    for pos_key in all_actual_positions:
        PLAYERS_BY_POSITION[pos_key] = [p for p in PLAYER_POOL if p[2] == pos_key]
        if PLAYERS_BY_POSITION[pos_key]: # Only sort if list is not empty
            PLAYERS_BY_POSITION[pos_key].sort(key=lambda x: x[4]) # Sort by round (x[4])

    PLAYER_ID_TO_DATA = {p[0]: p for p in PLAYER_POOL} # p[0] is player_id

    current_position_order = []
    for slot_type, count in ROSTER_STRUCTURE.items():
        for _ in range(count):
            current_position_order.append(slot_type)
    POSITION_ORDER = current_position_order
    TOTAL_ROSTER_SPOTS = len(POSITION_ORDER)

    # Validate ROSTER_STRUCTURE slot types
    for slot_type_key in ROSTER_STRUCTURE.keys():
        is_basic_pos = slot_type_key in PLAYERS_BY_POSITION # Check if it's a known basic position from CSV
        is_flex_pos = slot_type_key in FLEX_ELIGIBILITY
        
        if not is_basic_pos and not is_flex_pos:
            print(f"Critical Error: Slot type '{slot_type_key}' in ROSTER_STRUCTURE is not a recognized basic position found in the player pool, nor a defined FLEX type. Exiting.")
            exit()
        if is_flex_pos: # Also validate that positions within FLEX_ELIGIBILITY are known basic positions
            for eligible_pos in FLEX_ELIGIBILITY[slot_type_key]:
                if eligible_pos not in PLAYERS_BY_POSITION:
                    print(f"Critical Error: Position '{eligible_pos}' (part of FLEX_ELIGIBILITY for '{slot_type_key}') not found as a basic position in the player pool. Exiting.")
                    exit()
    
    print("Global data initialized.")
    print(f"POSITION_ORDER: {POSITION_ORDER}")
    print(f"TOTAL_ROSTER_SPOTS: {TOTAL_ROSTER_SPOTS}")

# --- Helper Functions ---
def get_player_data(player_id):
    return PLAYER_ID_TO_DATA.get(player_id)

def get_player_round(player_id):
    player_data = get_player_data(player_id)
    return player_data[4] if player_data else -1

def get_slot_type_for_index(index):
    if 0 <= index < len(POSITION_ORDER):
        return POSITION_ORDER[index]
    print(f"Error: Invalid slot index {index} requested in get_slot_type_for_index.")
    return None

def get_eligible_players_for_slot_type(slot_type_value):
    eligible_players = []
    processed_ids = set() # To ensure unique players in the combined list for flex

    if slot_type_value in FLEX_ELIGIBILITY:
        for actual_pos in FLEX_ELIGIBILITY[slot_type_value]:
            for player in PLAYERS_BY_POSITION.get(actual_pos, []):
                if player[0] not in processed_ids:
                    eligible_players.append(player)
                    processed_ids.add(player[0])
    elif slot_type_value in PLAYERS_BY_POSITION:
        eligible_players = PLAYERS_BY_POSITION[slot_type_value]
    else:
        # This case should ideally be caught by initialization checks
        print(f"Warning: Unknown slot type '{slot_type_value}' in get_eligible_players_for_slot_type.")
    return eligible_players

# --- 1. Genetic Algorithm Parameters ---
POPULATION_SIZE = 150
N_GENERATIONS = 150
MUTATION_RATE = 0.20
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 7
PENALTY_VIOLATION = 10000

# --- 2. Chromosome Representation & Initialization ---
def create_individual():
    individual_ids = [None] * TOTAL_ROSTER_SPOTS
    used_player_ids = set()
    used_rounds = set()

    for i in range(TOTAL_ROSTER_SPOTS):
        slot_type_to_fill = POSITION_ORDER[i]
        candidate_pool = get_eligible_players_for_slot_type(slot_type_to_fill)

        if not candidate_pool:
            individual_ids[i] = -99 # Mark as highly invalid if no players for this slot type
            continue

        possible_players = [
            p for p in candidate_pool
            if p[0] not in used_player_ids and p[4] not in used_rounds
        ]
        
        if not possible_players: # Try relaxing round constraint
            possible_players = [p for p in candidate_pool if p[0] not in used_player_ids]
            if not possible_players: # If still no one, pick any from candidate_pool (might be used)
                if candidate_pool: # Should always be true if we didn't continue above
                    chosen_player = random.choice(candidate_pool)
                else: # Should not be reached
                    individual_ids[i] = -1 
                    continue
            else:
                chosen_player = random.choice(possible_players)
        else:
            chosen_player = random.choice(possible_players)

        individual_ids[i] = chosen_player[0]
        used_player_ids.add(chosen_player[0])
        # Only add round if it wasn't already used by a higher priority pick for this slot type
        # This logic is complex; current approach adds round if player is chosen.
        # Fitness and repair will handle round violations.
        used_rounds.add(chosen_player[4])

    for i in range(len(individual_ids)): # Fill any remaining None/-1/-99 slots
        if individual_ids[i] is None or individual_ids[i] <= 0:
            slot_type_fallback = POSITION_ORDER[i]
            fallback_pool = get_eligible_players_for_slot_type(slot_type_fallback)
            if fallback_pool:
                # Try to pick one not already in individual_ids
                current_ids_in_roster = set(id_val for id_val in individual_ids if id_val is not None and id_val > 0)
                options = [p for p in fallback_pool if p[0] not in current_ids_in_roster]
                if not options: # if all are used, pick any
                    options = fallback_pool
                if options:
                     individual_ids[i] = random.choice(options)[0]
                else: # No players at all for this slot type
                    individual_ids[i] = -99 
            else:
                individual_ids[i] = -99
    return individual_ids

def create_initial_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

# --- 3. Fitness Function ---
def calculate_fitness(individual_ids):
    total_ppg = 0
    if not individual_ids or len(individual_ids) != TOTAL_ROSTER_SPOTS or \
       any(pid is None or not isinstance(pid, int) or pid <= 0 for pid in individual_ids):
        return -float('inf'), 0, set(), []

    lineup_players_data = []
    for pid in individual_ids:
        p_data = get_player_data(pid)
        if p_data is None:
            return -float('inf'), 0, set(), [] # Player ID not found
        lineup_players_data.append(p_data)

    # Constraint 0: Player position validity for the slot
    for i, player_data_tuple in enumerate(lineup_players_data):
        slot_type = get_slot_type_for_index(i)
        actual_player_position = player_data_tuple[2] # Player's actual position (e.g., "WR")

        if slot_type is None:
            return -PENALTY_VIOLATION * 20, 0, set(), lineup_players_data # Should not happen

        is_valid_position_for_slot = False
        if slot_type in FLEX_ELIGIBILITY:
            if actual_player_position in FLEX_ELIGIBILITY[slot_type]:
                is_valid_position_for_slot = True
        elif slot_type == actual_player_position: # Basic position check
            is_valid_position_for_slot = True
        
        if not is_valid_position_for_slot:
            return -PENALTY_VIOLATION * 10, 0, set(), lineup_players_data # Invalid position for slot

    # Constraint 1: No duplicate players (IDs)
    player_id_set = set(p_data[0] for p_data in lineup_players_data)
    if len(player_id_set) != TOTAL_ROSTER_SPOTS:
        return -PENALTY_VIOLATION * 5, 0, set(), lineup_players_data

    # Constraint 2: No duplicate rounds
    player_rounds = [p_data[4] for p_data in lineup_players_data]
    round_counts = Counter(player_rounds)
    round_violations = sum(count - 1 for count in round_counts.values() if count > 1)
    if round_violations > 0:
        return -PENALTY_VIOLATION * round_violations, 0, set(player_rounds), lineup_players_data

    for p_data in lineup_players_data:
        total_ppg += p_data[3]
    
    return total_ppg, total_ppg, set(player_rounds), lineup_players_data

# --- Repair Function (Crucial) ---
def repair_lineup(lineup_ids_to_repair):
    # Create a working copy
    repaired_ids = list(lineup_ids_to_repair)

    # Ensure basic validity before attempting complex repair
    if not repaired_ids or len(repaired_ids) != TOTAL_ROSTER_SPOTS:
        return create_individual() # Return a brand new valid individual

    # Iteration limit for repair passes to prevent infinite loops
    max_repair_iterations = TOTAL_ROSTER_SPOTS * 2 

    # Pass 1: Fix invalid player IDs (<=0) and duplicate player IDs
    for iteration in range(max_repair_iterations):
        id_counts = Counter(pid for pid in repaired_ids if pid is not None and pid > 0)
        made_change_in_player_pass = False
        for i in range(TOTAL_ROSTER_SPOTS):
            current_player_id = repaired_ids[i]
            slot_type = get_slot_type_for_index(i)
            if slot_type is None: continue # Should not happen

            # Check if current player is invalid or duplicated
            if current_player_id is None or current_player_id <= 0 or id_counts.get(current_player_id, 0) > 1:
                eligible_pool = get_eligible_players_for_slot_type(slot_type)
                if not eligible_pool: continue

                # IDs currently in the lineup, excluding the one being replaced (if it was valid)
                current_lineup_ids_excluding_self = set(pid for idx, pid in enumerate(repaired_ids) if idx != i and pid is not None and pid > 0)
                
                replacement_options = [p for p in eligible_pool if p[0] not in current_lineup_ids_excluding_self]

                if replacement_options:
                    new_player = random.choice(replacement_options)
                    # Update counts if the old player was valid and duplicated
                    if current_player_id is not None and current_player_id > 0:
                        id_counts[current_player_id] = id_counts.get(current_player_id, 0) -1
                    
                    repaired_ids[i] = new_player[0]
                    id_counts[new_player[0]] = id_counts.get(new_player[0], 0) + 1
                    made_change_in_player_pass = True
                else: # No unique replacement found, try picking any valid player for the slot (might cause new duplicate)
                    if eligible_pool:
                        last_resort_player = random.choice(eligible_pool)
                        if current_player_id is not None and current_player_id > 0: id_counts[current_player_id] -=1
                        repaired_ids[i] = last_resort_player[0]
                        id_counts[last_resort_player[0]] = id_counts.get(last_resort_player[0],0) + 1
                        # This might create a new duplicate that the next iteration or fitness will catch
                        made_change_in_player_pass = True


        if not made_change_in_player_pass: # If a full pass made no changes
            break
        if iteration == max_repair_iterations -1 :
            pass # print(f"Warning: Max iterations reached in player ID repair for {lineup_ids_to_repair}")


    # Ensure all slots have a player ID (even if it creates temporary duplicates to be resolved by fitness or next pass)
    for i in range(TOTAL_ROSTER_SPOTS):
        if repaired_ids[i] is None or repaired_ids[i] <=0:
            slot_type = get_slot_type_for_index(i)
            eligible_pool = get_eligible_players_for_slot_type(slot_type)
            if eligible_pool:
                repaired_ids[i] = random.choice(eligible_pool)[0] # Could be a duplicate
            else: # No players for this slot type AT ALL
                # This is a critical issue with player pool or roster config
                # Forcing a highly penalized lineup
                return [-99] * TOTAL_ROSTER_SPOTS # Indicates fundamental issue

    # Pass 2: Fix duplicate rounds (assuming player IDs are now unique and valid)
    # First, verify player position validity again after player ID repairs
    temp_lineup_player_data = []
    valid_indices_for_round_check = []
    for i_check in range(TOTAL_ROSTER_SPOTS):
        pid_check = repaired_ids[i_check]
        if pid_check is None or pid_check <=0: # Should be filled by now, but as a safeguard
            return repaired_ids # Give up if still fundamentally broken
        
        p_data_check = get_player_data(pid_check)
        if not p_data_check: return repaired_ids # Player ID somehow invalid

        slot_type_check = get_slot_type_for_index(i_check)
        actual_pos_check = p_data_check[2]
        
        pos_ok = False
        if slot_type_check in FLEX_ELIGIBILITY:
            if actual_pos_check in FLEX_ELIGIBILITY[slot_type_check]: pos_ok = True
        elif slot_type_check == actual_pos_check:
            pos_ok = True
        
        if not pos_ok: # Player in slot is not of correct type (e.g. QB in RB slot)
             # This indicates a deeper issue or a very constrained pool.
             # Attempt to fix this specific slot before round repair.
            eligible_pool_pos_fix = get_eligible_players_for_slot_type(slot_type_check)
            current_ids_in_lineup = set(pid for pid in repaired_ids if pid is not None and pid > 0 and pid != pid_check)
            options_pos_fix = [p for p in eligible_pool_pos_fix if p[0] not in current_ids_in_lineup]
            if options_pos_fix:
                repaired_ids[i_check] = random.choice(options_pos_fix)[0]
                p_data_check = get_player_data(repaired_ids[i_check]) # Get new data
            else: # Cannot fix position, this lineup is likely to be penalized
                pass # Allow fitness to penalize, or return as is.
        
        if p_data_check: # Re-check p_data_check if it was changed
            temp_lineup_player_data.append(p_data_check)
            valid_indices_for_round_check.append(i_check)


    if len(temp_lineup_player_data) != TOTAL_ROSTER_SPOTS : # If not a full roster of valid players
        return repaired_ids # Give up detailed round repair

    for iteration in range(max_repair_iterations):
        # Rebuild player_rounds_map for current state of repaired_ids
        player_rounds_map = {}
        current_player_details_for_rounds = [] # Store (idx, player_data)
        for r_idx in range(TOTAL_ROSTER_SPOTS):
            r_pid = repaired_ids[r_idx]
            r_p_data = get_player_data(r_pid)
            if r_p_data: # Ensure player data exists
                player_rounds_map[r_idx] = r_p_data[4] # player_data[4] is round
                current_player_details_for_rounds.append({'id':r_pid, 'round':r_p_data[4], 'original_index': r_idx, 'slot_type': get_slot_type_for_index(r_idx)})
            else: # Should not happen if previous steps filled all slots correctly
                # This lineup is broken beyond simple round repair for this slot
                continue

        round_counts = Counter(player_rounds_map.values())
        made_change_in_round_pass = False

        # Sort players by their round count (desc) then by round (asc) to prioritize fixing worst offenders
        # or players in earlier rounds that might free up options for later rounds.
        # For simplicity, iterate through slots.
        for i in range(TOTAL_ROSTER_SPOTS):
            player_detail = next((item for item in current_player_details_for_rounds if item['original_index'] == i), None)
            if not player_detail: continue

            current_round_of_player_i = player_detail['round']
            
            if round_counts[current_round_of_player_i] <= 1: # This player's round is not duplicated
                continue

            # This player's round is duplicated, try to change THIS player
            slot_type = player_detail['slot_type']
            eligible_pool_for_round_fix = get_eligible_players_for_slot_type(slot_type)

            # Rounds and IDs of OTHER players in the lineup
            other_players_rounds_set = set()
            other_player_ids_set = set()
            for k_detail in current_player_details_for_rounds:
                if k_detail['original_index'] == i: continue # Skip self
                other_players_rounds_set.add(k_detail['round'])
                other_player_ids_set.add(k_detail['id'])
            
            replacement_options = [
                p for p in eligible_pool_for_round_fix
                if p[0] not in other_player_ids_set and p[4] not in other_players_rounds_set
            ]
            
            if replacement_options:
                new_player = random.choice(replacement_options)
                
                round_counts[current_round_of_player_i] -= 1 # Decrement count for old round
                repaired_ids[i] = new_player[0] # Update ID in lineup
                round_counts[new_player[4]] = round_counts.get(new_player[4], 0) + 1 # Increment for new
                # Update player_rounds_map for consistency within this pass (or rebuild next iteration)
                player_rounds_map[i] = new_player[4]
                # Update current_player_details_for_rounds as well
                player_detail['id'] = new_player[0]
                player_detail['round'] = new_player[4]

                made_change_in_round_pass = True
        
        if not made_change_in_round_pass: # If a full pass made no changes to rounds
            break
        if iteration == max_repair_iterations -1:
            pass # print(f"Warning: Max iterations reached in round repair for {lineup_ids_to_repair}")
            
    return repaired_ids

# --- 4. Selection Operator ---
def tournament_selection(population, fitness_scores_only):
    if not population: return create_individual() # Should not happen if init pop is good
    actual_tournament_size = min(TOURNAMENT_SIZE, len(population))
    if actual_tournament_size == 0 : return random.choice(population) if population else create_individual()


    selected_indices = random.sample(range(len(population)), actual_tournament_size)
    tournament_individuals = [population[i] for i in selected_indices]
    tournament_fitnesses = [fitness_scores_only[i] for i in selected_indices]
    
    winner_index_in_tournament = np.argmax(tournament_fitnesses)
    return tournament_individuals[winner_index_in_tournament]

# --- 5. Crossover Operator ---
def crossover(parent1_ids, parent2_ids):
    child1_ids, child2_ids = list(parent1_ids), list(parent2_ids)
    if random.random() < CROSSOVER_RATE:
        if len(parent1_ids) > 1 and len(parent1_ids) == TOTAL_ROSTER_SPOTS:
            pt = random.randint(1, len(parent1_ids) - 1)
            child1_ids = parent1_ids[:pt] + parent2_ids[pt:]
            child2_ids = parent2_ids[:pt] + parent1_ids[pt:]
    
    child1_ids = repair_lineup(child1_ids)
    child2_ids = repair_lineup(child2_ids)
    return child1_ids, child2_ids

# --- 6. Mutation Operator ---
def mutate(individual_ids):
    mutated_ids = list(individual_ids)
    if random.random() < MUTATION_RATE:
        if not mutated_ids or len(mutated_ids) != TOTAL_ROSTER_SPOTS:
            return repair_lineup(mutated_ids)

        mutation_idx = random.randrange(len(mutated_ids))
        original_player_id_at_idx = mutated_ids[mutation_idx] # Could be None or <=0 if lineup is broken
        
        slot_type_to_mutate = get_slot_type_for_index(mutation_idx)
        if not slot_type_to_mutate: return repair_lineup(mutated_ids)

        eligible_player_pool_for_mutation = get_eligible_players_for_slot_type(slot_type_to_mutate)
        if not eligible_player_pool_for_mutation: return repair_lineup(mutated_ids)

        other_players_rounds = set()
        other_player_ids = set()
        original_player_data = get_player_data(original_player_id_at_idx) if original_player_id_at_idx else None
        original_player_round = original_player_data[4] if original_player_data else -1


        for i in range(len(mutated_ids)):
            if i == mutation_idx: continue
            pid_at_i = mutated_ids[i]
            if pid_at_i is None or pid_at_i <=0 : continue # Skip invalid slots
            p_data = get_player_data(pid_at_i)
            if p_data:
                other_players_rounds.add(p_data[4])
                other_player_ids.add(p_data[0])
        
        candidate_replacements = [
            p for p in eligible_player_pool_for_mutation
            if p[0] not in other_player_ids and \
               (p[4] not in other_players_rounds or (original_player_round != -1 and p[4] == original_player_round))
        ]
        
        if not candidate_replacements:
            candidate_replacements = [p for p in eligible_player_pool_for_mutation if p[0] not in other_player_ids and p[0] != original_player_id_at_idx]
        
        if not candidate_replacements: # Fallback: any player from eligible pool not the same ID (can create round duplicates)
             candidate_replacements = [p for p in eligible_player_pool_for_mutation if p[0] != original_player_id_at_idx]


        if candidate_replacements:
            mutated_ids[mutation_idx] = random.choice(candidate_replacements)[0]
    
    return repair_lineup(mutated_ids)

# --- 7. Main Genetic Algorithm Loop ---
def genetic_algorithm_adp_lineup():
    if not PLAYER_POOL or not POSITION_ORDER: 
        print("Critical: GA called with uninitialized or empty PLAYER_POOL/POSITION_ORDER.")
        return None, -float('inf')

    population = create_initial_population()
    if not population or not all(len(ind) == TOTAL_ROSTER_SPOTS for ind in population):
        print("Critical: Initial population is empty or malformed. Check create_individual.")
        return None, -float('inf')
    print(f"Initial population of {len(population)} lineups created.")

    best_lineup_overall_ids = None
    best_fitness_overall = -float('inf')
    best_points_overall = 0

    for generation in range(N_GENERATIONS):
        fitness_results = [calculate_fitness(ind) for ind in population]
        fitness_scores_only = [res[0] for res in fitness_results]

        current_best_idx_in_gen = np.argmax(fitness_scores_only)
        current_best_fitness_in_gen = fitness_scores_only[current_best_idx_in_gen]
        
        if current_best_fitness_in_gen > best_fitness_overall:
            best_fitness_overall = current_best_fitness_in_gen
            best_lineup_overall_ids = list(population[current_best_idx_in_gen])
            # Get points from the fitness_results directly for the best one
            # res[1] is total_ppg from calculate_fitness
            best_points_overall = fitness_results[current_best_idx_in_gen][1] if best_fitness_overall > -PENALTY_VIOLATION else 0 
            
            points_display = f"{best_points_overall:.2f}" if best_fitness_overall > -PENALTY_VIOLATION else "N/A (invalid)"
            print(f"Gen {generation + 1}: New Global Best! Fitness={best_fitness_overall:.2f}, Actual PPG={points_display}")

        next_population_ids = []
        elite_count = max(1, int(0.05 * POPULATION_SIZE))
        sorted_indices_current_gen = np.argsort(fitness_scores_only)[::-1]
        
        elites_added_this_gen = 0
        for i in range(len(sorted_indices_current_gen)):
            if elites_added_this_gen >= elite_count: break
            elite_candidate_idx = sorted_indices_current_gen[i]
            if fitness_scores_only[elite_candidate_idx] > -PENALTY_VIOLATION * 2: 
                candidate_lineup_for_elite = list(population[elite_candidate_idx])
                is_duplicate_elite = any(Counter(existing_elite) == Counter(candidate_lineup_for_elite) for existing_elite in next_population_ids)
                if not is_duplicate_elite:
                    next_population_ids.append(candidate_lineup_for_elite)
                    elites_added_this_gen +=1
        
        while len(next_population_ids) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitness_scores_only)
            parent2 = tournament_selection(population, fitness_scores_only)
            child1, child2 = crossover(parent1, parent2)
            next_population_ids.append(mutate(child1))
            if len(next_population_ids) < POPULATION_SIZE:
                next_population_ids.append(mutate(child2))
        
        population = next_population_ids[:POPULATION_SIZE]

        if (generation + 1) % 10 == 0 or generation == N_GENERATIONS - 1:
            # For periodic print, re-evaluate fitness of current population's apparent best
            # as population might have changed slightly after selection/crossover/mutation
            temp_fitness_results = [calculate_fitness(ind) for ind in population]
            temp_fitness_scores = [res[0] for res in temp_fitness_results]
            idx_best_in_current_pop = np.argmax(temp_fitness_scores)
            
            gen_fitness, gen_points, gen_rounds_set, _ = temp_fitness_results[idx_best_in_current_pop]
            points_display_periodic = f"{gen_points:.2f}" if gen_fitness > -PENALTY_VIOLATION else 'N/A (invalid)'
            rounds_display_periodic = len(gen_rounds_set) if gen_fitness > -PENALTY_VIOLATION else 'N/A (invalid)'
            print(f"Gen {generation + 1}: Current Pop Best Fitness={gen_fitness:.2f}, Actual PPG={points_display_periodic}, Unique Rounds Used={rounds_display_periodic}")

    print("\n--- Genetic Algorithm Finished ---")
    if best_lineup_overall_ids and best_fitness_overall > -PENALTY_VIOLATION / 2:
        final_fitness, final_points, final_rounds_set, final_player_data_list = calculate_fitness(best_lineup_overall_ids)
        print(f"Best Lineup Found (Player IDs): {best_lineup_overall_ids}")
        print(f"Stats: Final Fitness={final_fitness:.2f}, Total Actual PPG={final_points:.2f}, Unique Rounds Used={len(final_rounds_set)}")
        print("\nBest Lineup Details:")
        
        if final_player_data_list and len(final_player_data_list) == TOTAL_ROSTER_SPOTS:
            best_lineup_data_map = {p_data[0]: p_data for p_data in final_player_data_list}
            displayed_count = 0
            for i, player_id_in_slot in enumerate(best_lineup_overall_ids):
                p_data = best_lineup_data_map.get(player_id_in_slot)
                slot_type_display = get_slot_type_for_index(i)
                if p_data:
                    print(f"  - Slot {i} ({slot_type_display}): {p_data[1]} ({p_data[2]}, ID: {p_data[0]}) - PPG: {p_data[3]:.2f}, Calc Round: {p_data[4]}")
                    displayed_count +=1
                else:
                    print(f"  - Slot {i} ({slot_type_display}): Error - Player ID {player_id_in_slot} data not found in final list.")
            if displayed_count != TOTAL_ROSTER_SPOTS:
                print("Warning: Not all player details displayed for best lineup.")

            if len(final_rounds_set) != TOTAL_ROSTER_SPOTS:
                print(f"WARNING: Final lineup does not use {TOTAL_ROSTER_SPOTS} unique rounds! (Used: {len(final_rounds_set)})")
            player_ids_in_final = [p_data[0] for p_data in final_player_data_list]
            if len(set(player_ids_in_final)) != TOTAL_ROSTER_SPOTS:
                print(f"WARNING: Final lineup contains DUPLICATE players! IDs: {player_ids_in_final}")
            else:
                print("Best lineup appears valid based on final constraint checks.")
        else:
            print("Best lineup data malformed or incomplete for detailed printout.")
        return best_lineup_overall_ids, best_fitness_overall
    else:
        print("No significantly valid solution found, or solution did not improve beyond penalties.")
        if best_lineup_overall_ids:
            print(f"Last best attempt (may be highly penalized): {best_lineup_overall_ids} with fitness {best_fitness_overall:.2f}")
            _, _, _, last_attempt_data = calculate_fitness(best_lineup_overall_ids)
            if last_attempt_data:
                print("Details of last attempt:")
                for p_d in last_attempt_data: print(f"  - {p_d[2]} {p_d[1]}: PPG={p_d[3]:.2f}, Round={p_d[4]}")
        else:
            print("No lineup was ever selected as best_lineup_overall_ids.")
        return None, -float('inf')

# --- 8. Run the Genetic Algorithm ---
if __name__ == "__main__":
    print(f"Attempting to load player data from: {DEFAULT_CSV_FILENAME}")
    print(f"PPG will be calculated assuming {GAMES_IN_SEASON} games per season.")
    print(f"Draft rounds will be calculated assuming {PICKS_PER_ROUND} picks per round.")
    print("Roster settings:")
    for slot_def, count in ROSTER_STRUCTURE.items():
        eligibility = ""
        if slot_def in FLEX_ELIGIBILITY:
            eligibility = f" (Eligible: {', '.join(FLEX_ELIGIBILITY[slot_def])})"
        print(f"  - {slot_def}: {count}{eligibility}")
    print("-" * 30)

    initialize_global_data(DEFAULT_CSV_FILENAME)

    if not PLAYER_POOL or not POSITION_ORDER:
        print("Failed to initialize critical data. Cannot run genetic algorithm. Exiting.")
        exit()

    best_lineup_ids, best_fitness_val = genetic_algorithm_adp_lineup()
    
    print("\n" + "=" * 30)
    print("--- Main Program Output ---")
    if best_lineup_ids:
        print(f"Returned Best Lineup IDs: {best_lineup_ids}")
        print(f"Returned Best Fitness: {best_fitness_val:.2f}")
    else:
        print("The genetic algorithm did not return a valid best lineup.")
    print("=" * 30)