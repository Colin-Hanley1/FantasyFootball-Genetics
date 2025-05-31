import random
import math
import numpy as np
from collections import Counter
import csv
# Ensure 'os' is imported if you are using the version that writes to ga_training_log.csv
# import os

# --- Configuration Constants ---
GAMES_IN_SEASON = 17
PICKS_PER_ROUND = 12
DEFAULT_CSV_FILENAME = "player_pool.csv"
# GA_LOG_FILENAME = "ga_training_log.csv" # If using the logging version

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

# --- 0. Data Loading and Initialization --- ( 그대로 유지 )
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
                    ppg = total_points / GAMES_IN_SEASON # Make sure GAMES_IN_SEASON is not zero
                    adp = int(row["ADP"])
                    calculated_round = max(1, math.ceil(adp / PICKS_PER_ROUND)) # Ensure PICKS_PER_ROUND is not zero
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
    for slot_type, count in ROSTER_STRUCTURE.items():
        for _ in range(count):
            temp_position_order.append(slot_type)
    POSITION_ORDER = temp_position_order
    TOTAL_ROSTER_SPOTS = len(POSITION_ORDER)
    if TOTAL_ROSTER_SPOTS == 0:
        print("Critical Error: ROSTER_STRUCTURE is empty, resulting in zero roster spots. Exiting.")
        exit()

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
            # Sort by ADP round primarily (ascending), then by PPG descending as a tie-breaker
            CURRENT_PLAYERS_BY_POSITION_FOR_GA[pos_key].sort(key=lambda x: (x[4], -x[3]))


    USER_PLAYER_SLOT_ASSIGNMENTS = {}
    available_slots_indices = list(range(TOTAL_ROSTER_SPOTS))
    # Sort user players by their ADP round to try and fit earlier ADP players first
    sorted_user_players = sorted(USER_DRAFTED_PLAYERS_DATA, key=lambda x: x[4])


    for player_data in sorted_user_players:
        player_id, _, actual_player_pos, _, _ = player_data
        assigned_this_player = False
        # Try to assign to a specific position slot first
        for slot_idx in list(available_slots_indices): # Iterate over a copy for safe removal
            slot_type = POSITION_ORDER[slot_idx]
            if slot_type == actual_player_pos:
                USER_PLAYER_SLOT_ASSIGNMENTS[player_id] = slot_idx
                available_slots_indices.remove(slot_idx)
                assigned_this_player = True
                break
        if assigned_this_player: continue

        # If not assigned to a specific slot, try flex slots
        for slot_idx in list(available_slots_indices): # Iterate over a copy
            slot_type = POSITION_ORDER[slot_idx]
            if slot_type in FLEX_ELIGIBILITY and actual_player_pos in FLEX_ELIGIBILITY[slot_type]:
                USER_PLAYER_SLOT_ASSIGNMENTS[player_id] = slot_idx
                available_slots_indices.remove(slot_idx)
                assigned_this_player = True
                break
        
        if not assigned_this_player:
            # This can happen if all specific and eligible flex slots are already taken by other user players
            print(f"Warning: Could not auto-assign your drafted player {player_data[1]} ({actual_player_pos}) to a roster slot. Player remains on your team but might be considered 'bench'.")

    # print(f"User players assigned to slots: {USER_PLAYER_SLOT_ASSIGNMENTS}") # Verbose, can be enabled for debugging
    num_ga_slots = TOTAL_ROSTER_SPOTS - len(USER_PLAYER_SLOT_ASSIGNMENTS)
    print(f"GA will attempt to fill {num_ga_slots} open roster slots.")


# --- Helper Functions --- ( 그대로 유지 )
def get_player_data(player_id):
    return MASTER_PLAYER_ID_TO_DATA.get(player_id)

def get_player_round(player_id):
    player_data = get_player_data(player_id)
    return player_data[4] if player_data else -1 # Return ADP round or -1 if not found

def get_slot_type_for_index(index):
    if 0 <= index < len(POSITION_ORDER):
        return POSITION_ORDER[index]
    return None

def get_eligible_players_for_slot_type_for_ga(slot_type_value):
    eligible_players = []
    processed_ids = set() # To avoid duplicates if a player fits multiple flex categories for one slot type
    if slot_type_value in FLEX_ELIGIBILITY:
        # For flex slots, gather all players from eligible positions
        for actual_pos in FLEX_ELIGIBILITY[slot_type_value]:
            # Use CURRENT_PLAYERS_BY_POSITION_FOR_GA which is already filtered for availability
            for player in CURRENT_PLAYERS_BY_POSITION_FOR_GA.get(actual_pos, []):
                if player[0] not in processed_ids:
                    eligible_players.append(player)
                    processed_ids.add(player[0])
    elif slot_type_value in CURRENT_PLAYERS_BY_POSITION_FOR_GA:
        # For specific position slots
        eligible_players = CURRENT_PLAYERS_BY_POSITION_FOR_GA[slot_type_value]
    
    # Ensure eligible_players are not already drafted or part of user's fixed assignments in GA context
    # This is implicitly handled because CURRENT_PLAYERS_BY_POSITION_FOR_GA is built from CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA
    return eligible_players # This list is already sorted by ADP round, then PPG (desc) from prepare_for_ga_run

def find_player_by_name(name_query):
    name_query_lower = name_query.lower().strip()
    exact_matches = []
    partial_matches = []

    # Exact match first
    for player_data_tuple in INITIAL_PLAYER_POOL_DATA: # Search in the master pool
        player_name_lower = player_data_tuple[1].lower().strip()
        if name_query_lower == player_name_lower:
            exact_matches.append(player_data_tuple)
    
    if exact_matches:
        return exact_matches # Prefer exact matches
    
    # Partial match if no exact found
    for player_data_tuple in INITIAL_PLAYER_POOL_DATA:
        player_name_lower = player_data_tuple[1].lower().strip()
        # A more robust partial match might involve checking if query is a substring
        # or using fuzzy matching if desired. For now, simple 'in'.
        normalized_player_name = player_name_lower.replace(".", "").replace("-", " ").replace("'", "")
        normalized_query = name_query_lower.replace(".", "").replace("-", " ").replace("'", "")
        if normalized_query in normalized_player_name:
            partial_matches.append(player_data_tuple)
            
    return partial_matches


# --- GA Core Functions (Modified for Live Draft) --- (create_individual, create_initial_population, repair_lineup, tournament_selection, crossover, mutate are unchanged from the previous version you provided that included ga_training_log.csv functionality)

def create_individual():
    individual_ids = [None] * TOTAL_ROSTER_SPOTS
    used_player_ids_in_this_individual = set()

    # 1. Place user's drafted and assigned players first
    for player_id, assigned_slot_index in USER_PLAYER_SLOT_ASSIGNMENTS.items():
        if 0 <= assigned_slot_index < TOTAL_ROSTER_SPOTS:
            individual_ids[assigned_slot_index] = player_id
            used_player_ids_in_this_individual.add(player_id)
    
    # 2. Fill remaining GA-controlled slots
    for i in range(TOTAL_ROSTER_SPOTS):
        if individual_ids[i] is not None:  # Slot already filled (likely by a user pick)
            continue

        slot_type_to_fill = POSITION_ORDER[i]
        # Get players eligible for this slot, EXCLUDING those already used in this individual
        candidate_pool = get_eligible_players_for_slot_type_for_ga(slot_type_to_fill)
        
        possible_players = [p for p in candidate_pool if p[0] not in used_player_ids_in_this_individual]
        
        if possible_players:
            # Prioritize players by ADP round, then PPG (desc) due to sorting in get_eligible_players_for_slot_type_for_ga
            # For random initial creation, can use random.choice. For more directed, could use p[0].
            chosen_player_data = random.choice(possible_players)
            individual_ids[i] = chosen_player_data[0]
            used_player_ids_in_this_individual.add(chosen_player_data[0])
        else:
            # No unique players available for this slot from the preferred pool.
            # This might mean the pool for this slot is exhausted or all remaining are already picked for other slots.
            individual_ids[i] = -99  # Mark as needing repair or unfillable

    # 3. Final check for any None or placeholder IDs in GA-controlled slots (e.g. if a slot couldn't be filled)
    # This step is crucial if the above loop leaves -99s that could potentially be filled by a player
    # who *was* available but would have been a duplicate if picked naively. Repair handles complex duplicates.
    # For creation, if a slot is -99, it implies a scarcity for that slot type given current constraints.
    # `repair_lineup` will later attempt to fix -99 or other invalid IDs.
    for i in range(len(individual_ids)):
        if individual_ids[i] is None or individual_ids[i] <= 0 : # if it's still None or a placeholder like -1, -99
            # Check if this slot was supposed to be a user pick; if so, it's an error.
            # However, USER_PLAYER_SLOT_ASSIGNMENTS should have filled these.
            # This focuses on GA-controlled slots that couldn't be filled.
            is_user_assigned_slot = any(s_idx == i for s_idx in USER_PLAYER_SLOT_ASSIGNMENTS.values())
            if not is_user_assigned_slot:
                 individual_ids[i] = -99 # Confirm placeholder if GA couldn't fill it.
    return individual_ids

def create_initial_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]

## MODIFIED FUNCTION ##
def calculate_fitness(individual_ids, curr_round):
    # Initial checks for individual validity (length, None, <=0 IDs)
    if not individual_ids or len(individual_ids) != TOTAL_ROSTER_SPOTS:
        return -float('inf'), 0, set(), [] # Malformed individual (length)
    
    # Check for placeholder/invalid player IDs
    # Using the penalty structure from the version that generated ga_training_log.csv for consistency
    if any(pid is None or not isinstance(pid, int) or pid <= 0 for pid in individual_ids):
        num_invalid_spots = sum(1 for pid in individual_ids if pid is None or not isinstance(pid, int) or pid <= 0)
        # Penalize heavily if there are unfillable slots (-99) or other invalid IDs (-1, None)
        return -PENALTY_VIOLATION * (num_invalid_spots + 20), 0, set(), [] 

    # Fetch player data
    lineup_players_data = []
    for pid in individual_ids:
        p_data = get_player_data(pid)
        if p_data is None: 
            # This case should ideally be caught by the (pid <= 0) check if -99 etc. are used,
            # but if a positive ID is somehow not in MASTER_PLAYER_ID_TO_DATA:
            return -PENALTY_VIOLATION * 30, 0, set(), [] # Player ID valid format, but not found
        lineup_players_data.append(p_data)

    # Roster position validation
    for i, p_data_tuple in enumerate(lineup_players_data):
        # p_data_tuple: (player_id, name, position, ppg, calculated_round)
        slot_type = POSITION_ORDER[i]
        actual_player_pos = p_data_tuple[2]
        is_valid_for_slot = False
        if slot_type in FLEX_ELIGIBILITY:
            if actual_player_pos in FLEX_ELIGIBILITY[slot_type]:
                is_valid_for_slot = True
        elif slot_type == actual_player_pos: # Specific position slot
            is_valid_for_slot = True
        
        if not is_valid_for_slot:
            return -PENALTY_VIOLATION * 10, 0, set(), lineup_players_data # Invalid position assignment

    # Check for duplicate players (essential)
    player_id_set = set(p_data[0] for p_data in lineup_players_data)
    if len(player_id_set) != TOTAL_ROSTER_SPOTS: # Each spot must be a unique player
        return -PENALTY_VIOLATION * 5, 0, set(), lineup_players_data # Duplicate player in lineup

    # If all checks above pass, calculate raw PPG sum
    current_lineup_raw_ppg = 0
    for p_data in lineup_players_data:
        current_lineup_raw_ppg += p_data[3]

    # Initialize fitness score with the raw PPG sum
    fitness_score = current_lineup_raw_ppg

    # ADP Round conflict penalty - NEW LOGIC
    player_adp_rounds_in_lineup = [p_data[4] for p_data in lineup_players_data] # Get ADP rounds of all players
    adp_round_counts = Counter(player_adp_rounds_in_lineup)
    
    num_future_round_stacking_violations = 0
    for adp_round_of_player_in_lineup, count in adp_round_counts.items():
        if count > 1:  # If multiple players are from the same ADP round
            # Check if this ADP round is a FUTURE round relative to the live draft's current round
            if adp_round_of_player_in_lineup >= curr_round:
                num_future_round_stacking_violations += (count - 1) # Each extra player from that future round is a violation

    if num_future_round_stacking_violations > 0:
        fitness_score -= (PENALTY_VIOLATION * num_future_round_stacking_violations)
        # The `current_lineup_raw_ppg` (second return value) remains the actual sum of player PPGs.
        # The penalty is applied only to `fitness_score` (first return value), which the GA uses for selection.
        # This allows logging/display of raw PPG even if fitness is penalized.

    return fitness_score, current_lineup_raw_ppg, set(player_adp_rounds_in_lineup), lineup_players_data


def repair_lineup(lineup_ids_to_repair):
    repaired_ids = list(lineup_ids_to_repair) 
    if not repaired_ids or len(repaired_ids) != TOTAL_ROSTER_SPOTS:
        return create_individual() 

    user_assigned_slots_indices = set(USER_PLAYER_SLOT_ASSIGNMENTS.values())
    
    # 1. Ensure user-assigned players are locked in their correct slots
    for player_id, assigned_slot_index in USER_PLAYER_SLOT_ASSIGNMENTS.items():
        if 0 <= assigned_slot_index < TOTAL_ROSTER_SPOTS:
            # If another player is in the user's locked slot, or it's empty, force user's player
            if repaired_ids[assigned_slot_index] != player_id:
                # If the player *was* elsewhere, nullify that to avoid duplicates before placing
                for j_idx, j_pid in enumerate(repaired_ids):
                    if j_pid == player_id and j_idx != assigned_slot_index:
                        repaired_ids[j_idx] = -99 # Mark for refill
                repaired_ids[assigned_slot_index] = player_id


    # 2. Iteratively fix invalid player IDs (<=0, especially -99) or duplicates in GA-controlled slots
    max_repair_loops = TOTAL_ROSTER_SPOTS * 2 
    for _ in range(max_repair_loops):
        current_roster_ids_for_uniqueness_check = set(pid for pid in repaired_ids if pid is not None and pid > 0)
        id_counts = Counter(pid for pid in repaired_ids if pid is not None and pid > 0)
        made_change_in_pass = False

        for i in range(TOTAL_ROSTER_SPOTS):
            if i in user_assigned_slots_indices: # Skip user-locked slots
                current_roster_ids_for_uniqueness_check.add(repaired_ids[i]) # Ensure user picks are part of uniqueness
                continue

            current_player_id_at_slot_i = repaired_ids[i]
            
            # Condition for replacement: Invalid ID, or a valid ID that's duplicated elsewhere
            needs_replacement = False
            if current_player_id_at_slot_i is None or current_player_id_at_slot_i <= 0:
                needs_replacement = True
            elif id_counts.get(current_player_id_at_slot_i, 0) > 1:
                needs_replacement = True
            
            if needs_replacement:
                slot_type = POSITION_ORDER[i]
                eligible_pool = get_eligible_players_for_slot_type_for_ga(slot_type)
                
                # Try to find a new player NOT currently in `current_roster_ids_for_uniqueness_check`
                # (excluding current_player_id_at_slot_i itself if it was a duplicate being removed)
                ids_to_avoid = set(current_roster_ids_for_uniqueness_check)
                if current_player_id_at_slot_i is not None and current_player_id_at_slot_i > 0 and needs_replacement: # if it was a duplicate
                     pass # it will be "removed" from counts implicitly by replacement. `ids_to_avoid` already has it.


                options = [p for p in eligible_pool if p[0] not in ids_to_avoid]
                
                if options:
                    new_player = random.choice(options)
                    
                    # Update counts and set for the player being removed (if it was valid)
                    if current_player_id_at_slot_i is not None and current_player_id_at_slot_i > 0:
                         id_counts[current_player_id_at_slot_i] -=1
                         # No need to remove from current_roster_ids_for_uniqueness_check explicitly here as we rebuild/recheck

                    repaired_ids[i] = new_player[0]
                    id_counts[new_player[0]] = id_counts.get(new_player[0], 0) + 1
                    current_roster_ids_for_uniqueness_check.add(new_player[0]) # Add new player to set for next iterations
                    made_change_in_pass = True
                else: # No unique options available from the pool
                    # If we absolutely must fill, and can't find a unique, this indicates a deep issue or very shallow pool.
                    # Forcing a pick from eligible_pool (even if it's a duplicate) is an option, but repair should aim for validity.
                    # Marking as -99 is safer if no valid unique pick can be made.
                    if current_player_id_at_slot_i is not None and current_player_id_at_slot_i > 0 and id_counts.get(current_player_id_at_slot_i,0)>0:
                         id_counts[current_player_id_at_slot_i] -=1

                    repaired_ids[i] = -99 
                    made_change_in_pass = True # A change was made (to -99)
        
        # If a full pass makes no changes and all IDs are valid and unique (for GA slots), break
        final_check_ids = [pid for idx, pid in enumerate(repaired_ids) if idx not in user_assigned_slots_indices and pid is not None and pid > 0]
        final_check_user_ids = [pid for idx, pid in enumerate(repaired_ids) if idx in user_assigned_slots_indices and pid is not None and pid > 0]
        
        all_ga_slots_valid = all(pid is not None and pid > 0 for idx, pid in enumerate(repaired_ids) if idx not in user_assigned_slots_indices) or not any(idx not in user_assigned_slots_indices for idx in range(TOTAL_ROSTER_SPOTS))
        
        combined_valid_ids = [pid for pid in repaired_ids if pid is not None and pid > 0]

        if not made_change_in_pass and len(set(combined_valid_ids)) == len(combined_valid_ids) and all(pid > 0 for pid in combined_valid_ids if pid is not None):
            # And all GA slots that *should* be filled are filled (or no GA slots exist)
            # This condition is tricky; if -99 is a valid state for "unfillable", then this break needs care.
            # For now, assume -99 means "still broken".
            # The fitness function will heavily penalize -99s.
            break
            
    # ADP round conflict repair is NOT done here; handled by fitness penalties.
    # Position validity is also primarily handled by fitness penalties. Repair focuses on ID validity and uniqueness.
    return repaired_ids

def tournament_selection(population, fitness_scores_only):
    if not population: # Should be caught by caller, but as a safeguard
        # print("Warning: Tournament selection called with empty population.")
        return create_individual() 
    
    actual_tournament_size = min(TOURNAMENT_SIZE, len(population))
    if actual_tournament_size <= 0 : 
        return random.choice(population) if population else create_individual()

    selected_indices = random.sample(range(len(population)), actual_tournament_size)
    
    tournament_individuals_options = [population[i] for i in selected_indices]
    tournament_fitnesses_options = [fitness_scores_only[i] for i in selected_indices]
    
    best_in_tournament_actual_idx = np.argmax(tournament_fitnesses_options)
    return tournament_individuals_options[best_in_tournament_actual_idx]


def crossover(parent1_ids, parent2_ids):
    child1_ids, child2_ids = list(parent1_ids), list(parent2_ids) 
    if random.random() < CROSSOVER_RATE:
        if TOTAL_ROSTER_SPOTS > 1: 
            # Ensure crossover point `pt` is at least 1 and less than TOTAL_ROSTER_SPOTS
            pt = random.randint(1, TOTAL_ROSTER_SPOTS - 1)
            
            temp_child1 = parent1_ids[:pt] + parent2_ids[pt:]
            temp_child2 = parent2_ids[:pt] + parent1_ids[pt:]

            # Re-enforce user's picks post-crossover as they are fixed
            for player_id, assigned_slot_index in USER_PLAYER_SLOT_ASSIGNMENTS.items():
                if 0 <= assigned_slot_index < TOTAL_ROSTER_SPOTS:
                    temp_child1[assigned_slot_index] = player_id
                    temp_child2[assigned_slot_index] = player_id
            child1_ids, child2_ids = temp_child1, temp_child2
        # If TOTAL_ROSTER_SPOTS is 1, no crossover happens, children are copies.
        
    return repair_lineup(child1_ids), repair_lineup(child2_ids)


def mutate(individual_ids):
    mutated_ids = list(individual_ids) 
    user_assigned_slots_indices = set(USER_PLAYER_SLOT_ASSIGNMENTS.values())

    if random.random() < MUTATION_RATE:
        if not mutated_ids or len(mutated_ids) != TOTAL_ROSTER_SPOTS:
            return repair_lineup(mutated_ids) 

        # Identify slots that GA can actually change
        ga_controlled_indices = [i for i in range(TOTAL_ROSTER_SPOTS) if i not in user_assigned_slots_indices]
        if not ga_controlled_indices: # No slots to mutate (e.g., roster is full of user picks)
            return repair_lineup(mutated_ids) # Effectively returns original (repaired)

        mutation_idx = random.choice(ga_controlled_indices)
        original_player_id_at_mutation_idx = mutated_ids[mutation_idx]
        
        slot_type_to_mutate = POSITION_ORDER[mutation_idx]
        eligible_pool_for_mutation = get_eligible_players_for_slot_type_for_ga(slot_type_to_mutate)

        if not eligible_pool_for_mutation: 
            return repair_lineup(mutated_ids) # No options to mutate to

        # Find a new player:
        # 1. Different from the original player in that slot (if original was valid)
        # 2. Not already present in OTHER slots of the individual
        other_player_ids_in_mutated = set(
            pid for idx, pid in enumerate(mutated_ids) if idx != mutation_idx and pid is not None and pid > 0
        )
        
        options = [p for p in eligible_pool_for_mutation if p[0] != original_player_id_at_mutation_idx and p[0] not in other_player_ids_in_mutated]
        
        if not options: # Fallback 1: Any player not in other_player_ids_in_mutated (might be same as original if original was invalid)
            options = [p for p in eligible_pool_for_mutation if p[0] not in other_player_ids_in_mutated]
        if not options: # Fallback 2: Any player different from original (might create duplicate elsewhere, repair will handle)
             options = [p for p in eligible_pool_for_mutation if p[0] != original_player_id_at_mutation_idx]
        if not options and eligible_pool_for_mutation: # Fallback 3: Any player from the pool (last resort)
            options = eligible_pool_for_mutation

        if options:
            mutated_ids[mutation_idx] = random.choice(options)[0]
            
    return repair_lineup(mutated_ids)


# --- Main GA Loop --- ( 그대로 유지, Assuming it uses the version that logs to CSV if GA_LOG_FILENAME is defined)
# This function would be the one from your previous version that includes the CSV logging.
# For brevity, I'm not repeating that entire function here, but calculate_fitness above would slot into it.
def genetic_algorithm_adp_lineup(curr_round):
    # This function body would be the one you have that includes:
    # - GA_LOG_FILENAME usage if you want to log iterations
    # - The main generation loop calling calculate_fitness, selection, crossover, mutation
    # - Printing of best lineups, etc.
    # The key is that `calculate_fitness` (defined above) is called within this loop.

    # --- Placeholder for the GA loop structure from your previous complete version ---
    open_ga_slots = TOTAL_ROSTER_SPOTS - len(USER_PLAYER_SLOT_ASSIGNMENTS)
    if open_ga_slots <= 0 :
        print("Roster is effectively full with your picks. Evaluating current team.")
        current_team_ids = [None] * TOTAL_ROSTER_SPOTS
        for pid, slot_idx in USER_PLAYER_SLOT_ASSIGNMENTS.items(): current_team_ids[slot_idx] = pid
        current_team_ids = [-1 if x is None else x for x in current_team_ids] # Ensure no Nones
        fitness, ppg, _, _ = calculate_fitness(current_team_ids, curr_round)
        ppg_display = f"{ppg:.2f}" if fitness > -PENALTY_VIOLATION * 1.5 else "N/A (Invalid)" # Adjusted penalty check
        print(f"Current Team Fitness: {fitness:.2f}, PPG: {ppg_display}")
        return current_team_ids, fitness

    if not CURRENT_AVAILABLE_PLAYER_POOL_FOR_GA and open_ga_slots > 0:
        print("Warning: No available players for GA to select for open slots.")
        current_team_ids = [None] * TOTAL_ROSTER_SPOTS
        for pid, slot_idx in USER_PLAYER_SLOT_ASSIGNMENTS.items(): current_team_ids[slot_idx] = pid
        current_team_ids = [-1 if x is None else x for x in current_team_ids]
        fitness,ppg,_,_ = calculate_fitness(current_team_ids, curr_round)
        ppg_display = f"{ppg:.2f}" if fitness > -PENALTY_VIOLATION * 1.5 else "N/A (Invalid)"
        print(f"Current Team (no GA run): Fitness: {fitness:.2f}, PPG: {ppg_display}")
        return current_team_ids, fitness
        
    population = create_initial_population()
    if not population or not all(ind and len(ind) == TOTAL_ROSTER_SPOTS for ind in population):
        print("Critical: Initial GA population is empty or malformed. Cannot run GA.")
        current_team_ids = [None] * TOTAL_ROSTER_SPOTS
        for pid, slot_idx in USER_PLAYER_SLOT_ASSIGNMENTS.items(): current_team_ids[slot_idx] = pid
        current_team_ids = [-1 if x is None else x for x in current_team_ids]
        fitness,ppg,_,_ = calculate_fitness(current_team_ids, curr_round)
        ppg_display = f"{ppg:.2f}" if fitness > -PENALTY_VIOLATION * 1.5 else "N/A (Invalid)"
        print(f"Current Team (GA init failed): Fitness: {fitness:.2f}, PPG: {ppg_display}")
        return current_team_ids, fitness
    # print(f"Initial GA population of {len(population)} lineups created.") # Less verbose

    best_lineup_overall_ids = None
    best_fitness_overall = -float('inf')

    # CSV Logging Setup (if GA_LOG_FILENAME is defined and os is imported)
    # file_exists = os.path.isfile(GA_LOG_FILENAME)
    # with open(GA_LOG_FILENAME, mode='a', newline='', encoding='utf-8') as csvfile_obj:
    #    log_writer = csv.writer(csvfile_obj)
    #    if not file_exists or os.path.getsize(GA_LOG_FILENAME) == 0:
    #        header = ["Generation", "Individual_Index_In_Pop", "Fitness", "PPG"] + \
    #                   [f"Slot_{j}_Player_ID" for j in range(TOTAL_ROSTER_SPOTS)]
    #        log_writer.writerow(header)

    for generation in range(N_GENERATIONS):
        fitness_results = [calculate_fitness(ind, curr_round) for ind in population]
        
        # Log to CSV here if doing so:
        # for i, individual_ids_log in enumerate(population):
        #     fitness_score_log = fitness_results[i][0] 
        #     ppg_score_log = fitness_results[i][1]     
        #     row_to_log = [generation + 1, i, fitness_score_log, ppg_score_log] + individual_ids_log
        #     log_writer.writerow(row_to_log) # Assuming log_writer is defined

        fitness_scores_only = [res[0] for res in fitness_results]
        
        current_gen_best_idx = np.argmax(fitness_scores_only)
        current_gen_best_fitness = fitness_scores_only[current_gen_best_idx]
        
        if current_gen_best_fitness > best_fitness_overall:
            best_fitness_overall = current_gen_best_fitness
            best_lineup_overall_ids = list(population[current_gen_best_idx]) 
            best_ppg_overall = fitness_results[current_gen_best_idx][1]

            # More concise print for new best:
            # ppg_display_string = f"{best_ppg_overall:.2f}" if best_fitness_overall > -PENALTY_VIOLATION * 1.5 else "N/A"
            # print(f"Gen {generation+1}: New Global Best! Fit={best_fitness_overall:.2f}, PPG={ppg_display_string}")


        next_population = []
        elite_count = max(1, int(0.05 * POPULATION_SIZE)) 
        sorted_indices_for_elitism = np.argsort(fitness_scores_only)[::-1] 
        
        elites_added_count = 0
        for elite_idx in sorted_indices_for_elitism:
            if elites_added_count >= elite_count: break
            if fitness_scores_only[elite_idx] > -PENALTY_VIOLATION * 1.5: # Stricter elite check
                candidate_elite = list(population[elite_idx]) 
                if not any(Counter(existing_elite) == Counter(candidate_elite) for existing_elite in next_population):
                    next_population.append(candidate_elite)
                    elites_added_count +=1
        
        while len(next_population) < POPULATION_SIZE:
            p1 = tournament_selection(population, fitness_scores_only)
            p2 = tournament_selection(population, fitness_scores_only)
            c1, c2 = crossover(p1, p2)
            next_population.append(mutate(c1))
            if len(next_population) < POPULATION_SIZE:
                next_population.append(mutate(c2))
        
        population = next_population[:POPULATION_SIZE]

        if (generation + 1) % 20 == 0 or generation == N_GENERATIONS - 1: # Print less often
            gen_best_fitness_display = fitness_scores_only[current_gen_best_idx]
            gen_best_ppg_display = fitness_results[current_gen_best_idx][1]
            ppg_disp_periodic = f"{gen_best_ppg_display:.2f}" if gen_best_fitness_display > -PENALTY_VIOLATION * 1.5 else 'N/A'
            print(f"Gen {generation+1}/{N_GENERATIONS}: Pop Best Fit={gen_best_fitness_display:.2f}, PPG={ppg_disp_periodic}")

    print("\n--- Genetic Algorithm Finished ---")
    if best_lineup_overall_ids and best_fitness_overall > -PENALTY_VIOLATION * 1.5 : 
        final_fitness, final_points, final_rounds_set, final_player_data_tuples = calculate_fitness(best_lineup_overall_ids, curr_round)
        
        print(f"🏆 Best Overall Lineup Found (Fitness: {final_fitness:.2f})")
        print(f"   Projected PPG: {final_points:.2f}")
        # print(f"   ADP Rounds Used: {sorted(list(final_rounds_set))}") # Can be verbose
        
        print("\n📋 Best Lineup Details:")
        if final_player_data_tuples: # Check if data tuples were returned (i.e. lineup was somewhat valid)
            for i, player_id_in_slot in enumerate(best_lineup_overall_ids):
                p_data_final = get_player_data(player_id_in_slot) 
                slot_type_display = get_slot_type_for_index(i)
                is_user_pick_str = ""
                if player_id_in_slot in USER_PLAYER_SLOT_ASSIGNMENTS and USER_PLAYER_SLOT_ASSIGNMENTS.get(player_id_in_slot) == i:
                    is_user_pick_str = " (Your Pick)"
                
                if p_data_final:
                    print(f"  Slot {i} ({slot_type_display:<5}): {p_data_final[1]:<25} ({p_data_final[2]:<2}) PPG: {p_data_final[3]:>5.2f} ADP Rd: {p_data_final[4]:>2}{is_user_pick_str}")
                else:
                     print(f"  Slot {i} ({slot_type_display:<5}): Invalid Player ID {player_id_in_slot}")
            
            actual_ids_in_best = [pid for pid in best_lineup_overall_ids if pid is not None and pid > 0]
            if len(set(actual_ids_in_best)) != len(actual_ids_in_best):
                 print("🚨 WARNING: FINAL BEST LINEUP REPORTED CONTAINS DUPLICATE PLAYERS.")
        else: # Should not happen if best_fitness_overall was high enough
            print("   Could not retrieve full details for the best lineup (likely indicates underlying issue or highly penalized lineup).")
            
        return best_lineup_overall_ids, best_fitness_overall
    else:
        print("⚠️ No significantly valid/improved solution found by GA. Consider your current team or top available players.")
        user_team_final_ids = [None] * TOTAL_ROSTER_SPOTS
        for pid_user, slot_idx_user in USER_PLAYER_SLOT_ASSIGNMENTS.items():
            if 0 <= slot_idx_user < TOTAL_ROSTER_SPOTS:
                user_team_final_ids[slot_idx_user] = pid_user
        user_team_final_ids = [-1 if x is None else x for x in user_team_final_ids]
        
        fit_user_final, ppg_user_final, _, _ = calculate_fitness(user_team_final_ids, curr_round)
        ppg_user_display = f"{ppg_user_final:.2f}" if fit_user_final > -PENALTY_VIOLATION * 1.5 else "N/A (Invalid)"
        print(f"Returning current user team state. Fitness: {fit_user_final:.2f}, PPG: {ppg_user_display}")
        return user_team_final_ids, fit_user_final
    # --- End of Placeholder for GA loop ---


# --- Main Interactive Live Draft Loop --- ( 그대로 유지 )
# This should use the version from your previous iteration which correctly calculates
# `curr_round_est` and passes it to `genetic_algorithm_adp_lineup`.
if __name__ == "__main__":
    initial_setup(DEFAULT_CSV_FILENAME) # Uses the corrected initial_setup
    if not INITIAL_PLAYER_POOL_DATA or not POSITION_ORDER or TOTAL_ROSTER_SPOTS == 0:
        print("Exiting due to critical initialization failure.")
        exit()
    
    print("\n🏈 Welcome to the Live Fantasy Football Draft Assistant! 🏈")
    print("Roster settings:")
    for slot_def, count in ROSTER_STRUCTURE.items():
        eligibility = f" (Eligible: {', '.join(FLEX_ELIGIBILITY[slot_def])})" if slot_def in FLEX_ELIGIBILITY else ""
        print(f"  - {slot_def}: {count}{eligibility}")
    print("-" * 30)
    # print(f"GA training data will be logged to: {GA_LOG_FILENAME}") # If logging is active

    while True:
        # More standard current round calculation
        if len(GLOBALLY_DRAFTED_PLAYER_IDS) == 0:
            curr_round_est = 1
        else:
            # Assumes PICKS_PER_ROUND is the total number of picks in one full round (e.g., 12 for a 12-team league)
            curr_round_est = math.floor(len(GLOBALLY_DRAFTED_PLAYER_IDS) / PICKS_PER_ROUND) + 1
        
        # Estimate max rounds for display capping, e.g. 15-20 rounds typical
        # This is just for display context, GA uses the precise curr_round_est
        max_draft_rounds = math.ceil(TOTAL_ROSTER_SPOTS * 1.5) if TOTAL_ROSTER_SPOTS > 0 else 15 
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
            query = " ".join(args)
            target_pid = None
            target_p_data = None

            try:
                pid_candidate = int(query)
                p_data_cand = get_player_data(pid_candidate)
                if p_data_cand:
                    target_pid = pid_candidate
                    target_p_data = p_data_cand
                else:
                    print(f"⚠️ Player ID '{query}' not found in master pool.")
                    continue
            except ValueError: 
                matched_players = find_player_by_name(query)
                if not matched_players:
                    print(f"⚠️ Player '{query}' not found by name.")
                    continue
                elif len(matched_players) == 1:
                    target_p_data = matched_players[0]
                    target_pid = target_p_data[0]
                else: 
                    print(f"⚠️ Ambiguous name '{query}'. Found multiple matches:")
                    for p_id_match, name_match, pos_match, _, _ in matched_players[:5]: # Show top 5
                        print(f"  ID: {p_id_match:<4} Name: {name_match:<25} Pos: {pos_match}")
                    print(f"Please use specific ID: '{command} <player_id>'.")
                    continue
            
            if target_pid is None: continue # Should be caught by logic above

            if command == 'd': 
                if target_pid in GLOBALLY_DRAFTED_PLAYER_IDS:
                    print(f"ℹ️ Player {target_p_data[1]} (ID: {target_pid}) was already marked as drafted.")
                else:
                    GLOBALLY_DRAFTED_PLAYER_IDS.add(target_pid)
                    print(f"👍 Player {target_p_data[1]} (ID: {target_pid}) marked globally drafted.")
                
                # If this player was on user's team, remove them
                was_on_user_team = any(p[0] == target_pid for p in USER_DRAFTED_PLAYERS_DATA)
                USER_DRAFTED_PLAYERS_DATA = [p for p in USER_DRAFTED_PLAYERS_DATA if p[0] != target_pid]
                if target_pid in USER_PLAYER_SLOT_ASSIGNMENTS:
                    del USER_PLAYER_SLOT_ASSIGNMENTS[target_pid]
                    if was_on_user_team: print(f"   Also removed from your team as they were drafted by another team.")

            elif command == 'my': 
                is_already_on_my_team = any(ud_p[0] == target_pid for ud_p in USER_DRAFTED_PLAYERS_DATA)
                is_globally_drafted_by_other = target_pid in GLOBALLY_DRAFTED_PLAYER_IDS and not is_already_on_my_team

                if is_globally_drafted_by_other:
                     print(f"🚫 Player {target_p_data[1]} (ID: {target_pid}) was already drafted by another team. Cannot add to your team.")
                elif is_already_on_my_team:
                    print(f"ℹ️ Player {target_p_data[1]} (ID: {target_pid}) is already on your team.")
                else: 
                    USER_DRAFTED_PLAYERS_DATA.append(target_p_data)
                    GLOBALLY_DRAFTED_PLAYER_IDS.add(target_pid) 
                    print(f"✅ You drafted: {target_p_data[1]} ({target_p_data[2]})! Added to your team.")
                    # Auto-assignment will happen in prepare_for_ga_run or when viewing team

        elif command == 'undo' and args:
            try:
                pid_to_undo = int(args[0])
                p_data_undo = get_player_data(pid_to_undo)
                if not p_data_undo:
                    print(f"⚠️ Player ID '{pid_to_undo}' not found. Cannot undo.")
                    continue

                undone_actions = []
                if pid_to_undo in GLOBALLY_DRAFTED_PLAYER_IDS:
                    GLOBALLY_DRAFTED_PLAYER_IDS.remove(pid_to_undo)
                    undone_actions.append("removed from globally drafted list")
                
                initial_user_team_size = len(USER_DRAFTED_PLAYERS_DATA)
                USER_DRAFTED_PLAYERS_DATA = [p for p in USER_DRAFTED_PLAYERS_DATA if p[0] != pid_to_undo]
                if len(USER_DRAFTED_PLAYERS_DATA) < initial_user_team_size:
                     undone_actions.append("removed from your team")
                
                if pid_to_undo in USER_PLAYER_SLOT_ASSIGNMENTS:
                    del USER_PLAYER_SLOT_ASSIGNMENTS[pid_to_undo]
                    # This is implicitly part of "removed from your team" if player was assigned

                if undone_actions:
                    print(f"⏪ Player {p_data_undo[1]} (ID: {pid_to_undo}) was: {', '.join(undone_actions)}.")
                else:
                    print(f"ℹ️ Player {p_data_undo[1]} (ID: {pid_to_undo}) was not found in any drafted lists to undo.")

            except ValueError:
                print("⚠️ Invalid player ID for 'undo'. Use 'undo <player_id>'.")
            except Exception as e: # Catch any other unexpected errors during undo
                print(f"💥 Error during undo operation: {e}")


        elif command == 'run':
            print("\n🔄 Preparing data for GA run...")
            prepare_for_ga_run() 
            open_ga_slots = TOTAL_ROSTER_SPOTS - len(USER_PLAYER_SLOT_ASSIGNMENTS)
            if len(USER_PLAYER_SLOT_ASSIGNMENTS) >= TOTAL_ROSTER_SPOTS and open_ga_slots <=0 : # check open_ga_slots too
                print("✅ Your starting roster is full! No GA run needed. Showing current team.")
                # Manually trigger team display logic similar to 'team' command
                command = 'team' # Fall through to 'team' display logic
                # continue # Skip GA, go to next input prompt after showing team
            else: 
                print("🧠 Running Genetic Algorithm for suggestions...")
                best_lineup_ids, best_fitness_val = genetic_algorithm_adp_lineup(curr_round_est) # Pass estimated actual current round

        # This 'if' must be separate from 'elif command == run' to allow fall-through
        if command == 'team': 
            prepare_for_ga_run() # Ensure USER_PLAYER_SLOT_ASSIGNMENTS is up-to-date
            print("\n--- 📊 Your Current Team ---")
            if not USER_DRAFTED_PLAYERS_DATA:
                print("You haven't drafted any players yet.")
            else:
                display_roster = ["<EMPTY SLOT>"] * TOTAL_ROSTER_SPOTS
                assigned_players_details_list = []
                
                for p_id_user, slot_idx_user in USER_PLAYER_SLOT_ASSIGNMENTS.items():
                    p_d_user = get_player_data(p_id_user)
                    if p_d_user:
                         display_roster[slot_idx_user] = f"{p_d_user[1]:<25} ({p_d_user[2]:<2}) PPG: {p_d_user[3]:>5.2f} ADP Rd: {p_d_user[4]:>2}"
                         assigned_players_details_list.append(p_d_user)
                    else: 
                        display_roster[slot_idx_user] = f"<UNKNOWN PLAYER ID: {p_id_user}>"

                print("Starters:")
                team_current_ppg = sum(p[3] for p in assigned_players_details_list)
                for i_slot_disp in range(TOTAL_ROSTER_SPOTS):
                    slot_type_disp = POSITION_ORDER[i_slot_disp]
                    print(f"  Slot {i_slot_disp} ({slot_type_disp:<5}): {display_roster[i_slot_disp]}")
                print(f"Total PPG from Starters: {team_current_ppg:.2f}")

                unassigned_user_players = [p_un for p_un in USER_DRAFTED_PLAYERS_DATA if p_un[0] not in USER_PLAYER_SLOT_ASSIGNMENTS]
                if unassigned_user_players:
                    print("\nBench/Surplus (Your Drafted Players Not in Starting Slots):")
                    for p_d_unassigned in sorted(unassigned_user_players, key=lambda x: (-x[3], x[4])): # Sort by PPG (desc), then ADP (asc)
                        print(f"  - {p_d_unassigned[1]:<25} ({p_d_unassigned[2]:<2}) ID: {p_d_unassigned[0]:<4} PPG: {p_d_unassigned[3]:>5.2f} ADP Rd: {p_d_unassigned[4]:>2}")

        elif command == 'drafted':
            print(f"\n--- 📜 Globally Drafted Players ({len(GLOBALLY_DRAFTED_PLAYER_IDS)}) ---")
            if not GLOBALLY_DRAFTED_PLAYER_IDS: print("None yet.")
            else:
                drafted_list_details_display = []
                for pid_glob in GLOBALLY_DRAFTED_PLAYER_IDS:
                    p_d_glob = get_player_data(pid_glob)
                    is_yours = any(ud_p[0] == pid_glob for ud_p in USER_DRAFTED_PLAYERS_DATA)
                    if p_d_glob: 
                        drafted_list_details_display.append( (p_d_glob, is_yours) )
                
                drafted_list_details_display.sort(key=lambda x: (x[0][4], x[0][0])) # Sort by ADP round, then player ID

                for p_d_glob_sorted, is_yours_sorted in drafted_list_details_display:
                    is_yours_str = " (Your Pick)" if is_yours_sorted else ""
                    print(f"Rd {p_d_glob_sorted[4]:>2}: {p_d_glob_sorted[1]:<25} ({p_d_glob_sorted[2]:<2}, ID: {p_d_glob_sorted[0]:<4}){is_yours_str}")
        
        elif command == 'available':
            temp_available_pool_view_cmd = [p for p in INITIAL_PLAYER_POOL_DATA if p[0] not in GLOBALLY_DRAFTED_PLAYER_IDS]
            temp_available_pool_view_cmd.sort(key=lambda x: (-x[3], x[4])) # Sort by PPG (desc), then ADP (asc)

            filter_pos_view_cmd = args[0].upper() if args else None
            
            title_str = f"--- ⭐ Top Available Players {('(' + filter_pos_view_cmd + ' only)' if filter_pos_view_cmd else '(All Positions)')} ---"
            print(f"\n{title_str}")
            count_view_cmd = 0
            limit_cmd = 15 if filter_pos_view_cmd else 30 # Show more if not filtered by position

            for p_avail_view_data in temp_available_pool_view_cmd:
                if filter_pos_view_cmd and p_avail_view_data[2] != filter_pos_view_cmd:
                    continue 
                
                print(f"{p_avail_view_data[1]:<25} ({p_avail_view_data[2]:<2}, ID: {p_avail_view_data[0]:<4}) PPG: {p_avail_view_data[3]:>5.2f} ADP Rd: {p_avail_view_data[4]:>2}")
                count_view_cmd += 1
                if count_view_cmd >= limit_cmd : break 
            
            if count_view_cmd == 0:
                print(f"None available {'for position ' + filter_pos_view_cmd if filter_pos_view_cmd else 'at all'}.")

        elif command == 'q':
            print("👋 Exiting live draft tool. Good luck with your draft!")
            # if GA_LOG_FILENAME is used: print(f"GA training log is in '{GA_LOG_FILENAME}'.")
            break
        elif command == "": 
            continue
        else:
            print(f"❓ Unknown command: '{command}'. Check available commands above.")