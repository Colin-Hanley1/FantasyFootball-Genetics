# Fantasy Football Live Draft Assistant & GA Optimizer

## Overview

Welcome to the Fantasy Football Live Draft Assistant! This application helps you manage your fantasy football draft in real-time and uses a Genetic Algorithm (GA) to provide optimal lineup suggestions and next-pick recommendations based on your current roster, available players, and various strategic considerations like ADP and bye weeks.

The app is built with Python using the Dash framework for the user interface and Dash Bootstrap Components for styling.

## Features

* **Live Draft Tracking:** Record players drafted by yourself and opponents.
* **Dynamic Available Player Pool:** See the best available players update as the draft progresses.
* **Roster Management:** View your current team with auto-assigned positions.
* **Genetic Algorithm (GA) Powered Suggestions:**
    * Generates an optimal full lineup based on available players and your current team.
    * Provides a "Next Pick Suggestion" to guide your immediate decision.
    * Calculates team fitness and projected Points Per Game (PPG).
* **Strategic Considerations:**
    * Player PPG (Points Per Game) as a primary metric.
    * ADP (Average Draft Position) round for valuing players and penalizing unrealistic team constructions.
    * Bye week conflict penalty to avoid too many players on bye simultaneously.
    * Penalties for mismanaging high-ADP bench players if starters are weak.
    * Penalties for "reaching back" too far for players past their ADP.
* **Flexible Player Search:** Search for players by ID, parts of their name, or their CSV formatted name.
* **Configurable Roster Settings:** Define your league's specific roster structure (QB, RB, WR, TE, FLEX, BENCH).
* **User-Friendly Interface:** Styled with Dash Bootstrap Components for a clean and responsive layout.

## Prerequisites

Before you can run this application, you'll need:

* **Python:** Version 3.8 or newer recommended.
* **pip:** Python's package installer.
* **Git:** (Optional, but recommended for managing the code if you clone it from a repository).

## Setup & Installation

1.  **Get the Code:**
    * If you have the code as a `.py` file (e.g., `dash_app.py`), save it to a new directory on your computer.
    * If it's in a Git repository, clone it:
        ```bash
        git clone <repository_url>
        cd <repository_directory>
        ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file in the same directory as your app script with the following content:
    ```txt
    dash
    dash-bootstrap-components
    numpy
    gunicorn # Recommended for deployment, good to have for consistency
    ```
    Then, install the packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare the Player Data CSV (`player_pool.csv`):**
    * Create a file named `player_pool.csv` in the same directory as your app script.
    * This file **must** have the following headers in the first row:
        `ID,Name,Position,TotalPoints,ADP,ByeWeek`
    * **Data Format:**
        * `ID`: Unique integer ID for each player.
        * `Name`: Player's name, formatted as `POSITION_FIRSTINITIALLASTNAME` (e.g., `WR_JJefferson` for Justin Jefferson, `TE_TKelce` for Travis Kelce). The app's flexible search is designed around this format in the CSV.
        * `Position`: Player's primary position (e.g., `QB`, `WR`, `RB`, `TE`, `K`, `DST`).
        * `TotalPoints`: Projected total fantasy points for the season.
        * `ADP`: Player's Average Draft Position (overall rank, e.g., 1, 2, 3...).
        * `ByeWeek`: Integer representing the player's bye week (e.g., `9`). Use `0` or leave blank if no bye week (e.g., for DSTs, though consistency is good).

## Running the Application

1.  Navigate to the directory containing your app script (e.g., `dash_app.py`) and `player_pool.csv`.
2.  Ensure your virtual environment is activated (if you created one).
3.  Run the app using the following command:
    ```bash
    python dash_app.py
    ```
4.  Open your web browser and go to the URL shown in the terminal (usually `http://127.0.0.1:8050/`).

The line `server = app.server` is included in the script, making it ready for deployment with WSGI servers like Gunicorn. For local development, `app.run(debug=True)` is typically used, but for deployment, `debug=False` is essential.

## How to Use the App

The application interface is divided into several main sections:

### 1. Current Draft Status (Top Center)

* Displays the current overall draft round and the next pick number in the draft. This updates as players are marked globally drafted.

### 2. Action Messages (Below Draft Status)

* This area will show confirmation messages for your actions (e.g., "Player X drafted") or any errors encountered (e.g., "Player not found"). Alerts are usually dismissable and may auto-dismiss after a few seconds.

### 3. Left Column: Draft Management

* **Draft Actions Card:**
    * **Enter Player Name or ID:** Type a player's ID (e.g., `158`) or name to select them for drafting.
        * **Flexible Name Search:** You can type:
            * The exact CSV formatted name (e.g., `WR_JJefferson`).
            * Last name (e.g., `Jefferson`).
            * First initial and last name (e.g., `J Jefferson`).
            * First initial directly followed by last name (e.g., `JJefferson`).
            * Position and last name (e.g., `WR Jefferson`).
            * Partial last names (e.g., `Jeffer`).
        * If the search is ambiguous (multiple players match), the app will notify you and suggest using the player's unique ID.
    * **"Draft for Opponent" Button:** Marks the selected player as drafted by another team. They will be removed from the available pool and added