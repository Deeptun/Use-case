import random
import tkinter as tk
import webbrowser
import time
import threading
from tkinter import messagebox, simpledialog, ttk, font

# Game state dictionary
game_state = {
    'players': [],
    'game_started': False,
    'current_player': 0,
    'visited_tiles': set(),
    'timer_value': 0,
    'start_time': 0,
    'used_questions': {},
    'player_icons': ["circle", "triangle", "square", "star"],
    'root': None,
    'canvas': None,
    'dice_label': None,
    'score_label': None,
    'timer_label': None,
    'roll_btn': None,
    'main_frame': None
}

# Store all tiles data
tiles = []
quiz_questions = {}

def create_initial_tiles():
    """Create all tiles for the game board with hard-coded properties"""
    global tiles
    
    # Define quiz tile positions - 3 on each arm (12 total)
    quiz_positions = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23}
    
    # Hard-coded tile properties for all 24 tiles
    tile_properties = [
        # Corner tiles
        {'name': "GO", 'symbol': "🎮", 'value': 0, 'hyperlink': "https://www.google.com", 'color': "#FFB6C1", 'is_corner': True, 'has_quiz': False, 'teleport': False},
        {'name': "Property 1", 'value': -75, 'hyperlink': "https://www.amazon.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': True},
        {'name': "Property 2", 'value': -50, 'hyperlink': "https://www.youtube.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': False},
        {'name': "Property 3", 'value': -100, 'hyperlink': "https://www.netflix.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': True},
        {'name': "Property 4", 'value': -65, 'hyperlink': "https://www.twitter.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': False},
        {'name': "Property 5", 'value': -120, 'hyperlink': "https://www.reddit.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': True},
        {'name': "JAIL", 'symbol': "🔒", 'value': 0, 'hyperlink': "https://www.linkedin.com", 'color': "#FFB6C1", 'is_corner': True, 'has_quiz': False, 'teleport': True},
        {'name': "Property 7", 'value': -90, 'hyperlink': "https://www.instagram.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': True},
        {'name': "Property 8", 'value': -55, 'hyperlink': "https://www.apple.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': False},
        {'name': "Property 9", 'value': -110, 'hyperlink': "https://www.microsoft.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': True},
        {'name': "Property 10", 'value': -70, 'hyperlink': "https://www.github.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': False},
        {'name': "Property 11", 'value': -130, 'hyperlink': "https://www.stackoverflow.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': True},
        {'name': "FREE PARKING", 'symbol': "🅿️", 'value': 0, 'hyperlink': "https://www.facebook.com", 'color': "#FFB6C1", 'is_corner': True, 'has_quiz': False, 'teleport': True},
        {'name': "Property 13", 'value': -85, 'hyperlink': "https://www.wikipedia.org", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': True},
        {'name': "Property 14", 'value': -60, 'hyperlink': "https://www.spotify.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': False},
        {'name': "Property 15", 'value': -95, 'hyperlink': "https://www.nytimes.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': True},
        {'name': "Property 16", 'value': -80, 'hyperlink': "https://www.cnn.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': False},
        {'name': "Property 17", 'value': -115, 'hyperlink': "https://www.bbc.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': True},
        {'name': "GO TO JAIL", 'symbol': "👮", 'value': 0, 'hyperlink': "https://www.espn.com", 'color': "#FFB6C1", 'is_corner': True, 'has_quiz': False, 'teleport': True},
        {'name': "Property 19", 'value': -105, 'hyperlink': "https://www.weather.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': True},
        {'name': "Property 20", 'value': -65, 'hyperlink': "https://www.twitch.tv", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': False},
        {'name': "Property 21", 'value': -125, 'hyperlink': "https://www.pinterest.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': True},
        {'name': "Property 22", 'value': -75, 'hyperlink': "https://www.ebay.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': False},
        {'name': "Property 23", 'value': -140, 'hyperlink': "https://www.walmart.com", 'color': "#FFE4E1", 'is_corner': False, 'has_quiz': True},
    ]
    
    # Use the hard-coded properties
    tiles = tile_properties
    
    return tiles

def create_question_bank():
    """Create quiz questions for quiz tiles with 10 unique questions per tile"""
    global quiz_questions
    quiz_questions = {}
    
    # Hard-coded question banks for each quiz tile position - 10 questions per tile
    quiz_questions = {
        # Property 1 questions
        "Property 1": [
            {"question": "What is the capital of France?", "answer": "paris"},
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What planet is known as the Red Planet?", "answer": "mars"},
            {"question": "What is the chemical symbol for gold?", "answer": "au"},
            {"question": "Who painted the Mona Lisa?", "answer": "leonardo da vinci"},
            {"question": "How many continents are there?", "answer": "7"},
            {"question": "What is the square root of 64?", "answer": "8"},
            {"question": "What is the largest mammal on Earth?", "answer": "blue whale"},
            {"question": "What is the first element in the periodic table?", "answer": "hydrogen"},
            {"question": "What is the largest planet in our solar system?", "answer": "jupiter"}
        ],
        
        # Property 3 questions
        "Property 3": [
            {"question": "What is the capital of Japan?", "answer": "tokyo"},
            {"question": "What is 7×8?", "answer": "56"},
            {"question": "Which element has the chemical symbol 'O'?", "answer": "oxygen"},
            {"question": "Who wrote 'Romeo and Juliet'?", "answer": "shakespeare"},
            {"question": "What is the currency of United Kingdom?", "answer": "pound"},
            {"question": "What is the largest ocean on Earth?", "answer": "pacific"},
            {"question": "How many sides does a pentagon have?", "answer": "5"},
            {"question": "What is the capital of Australia?", "answer": "canberra"},
            {"question": "Which planet is closest to the Sun?", "answer": "mercury"},
            {"question": "What is the chemical formula for water?", "answer": "h2o"}
        ],
        
        # Property 5 questions
        "Property 5": [
            {"question": "Who is known as the father of computers?", "answer": "charles babbage"},
            {"question": "What is the square root of 144?", "answer": "12"},
            {"question": "What is the capital of Egypt?", "answer": "cairo"},
            {"question": "Which gas do plants absorb from the atmosphere?", "answer": "carbon dioxide"},
            {"question": "Who discovered penicillin?", "answer": "alexander fleming"},
            {"question": "How many players are in a standard soccer team?", "answer": "11"},
            {"question": "What is the hardest natural substance on Earth?", "answer": "diamond"},
            {"question": "What is the largest land animal?", "answer": "elephant"},
            {"question": "What do you call a shape with 8 sides?", "answer": "octagon"},
            {"question": "What is the main component of the Sun?", "answer": "hydrogen"}
        ],
        
        # Property 7 questions
        "Property 7": [
            {"question": "Who painted the Sistine Chapel ceiling?", "answer": "michelangelo"},
            {"question": "What is the smallest prime number?", "answer": "2"},
            {"question": "What is the capital of Brazil?", "answer": "brasilia"},
            {"question": "What is the tallest mountain in the world?", "answer": "everest"},
            {"question": "Which planet has the most moons?", "answer": "saturn"},
            {"question": "What color is chlorophyll?", "answer": "green"},
            {"question": "What is the freezing point of water in Fahrenheit?", "answer": "32"},
            {"question": "What is the powerhouse of the cell?", "answer": "mitochondria"},
            {"question": "What's the smallest unit of matter?", "answer": "atom"},
            {"question": "What is the most abundant gas in Earth's atmosphere?", "answer": "nitrogen"}
        ],
        
        # Property 9 questions
        "Property 9": [
            {"question": "Who invented the light bulb?", "answer": "thomas edison"},
            {"question": "What is the square of 13?", "answer": "169"},
            {"question": "What is the capital of China?", "answer": "beijing"},
            {"question": "Which instrument has 88 keys?", "answer": "piano"},
            {"question": "What is the largest desert in the world?", "answer": "sahara"},
            {"question": "What is the most common element in the Earth's crust?", "answer": "oxygen"},
            {"question": "How many bones are in the human body?", "answer": "206"},
            {"question": "Who discovered gravity?", "answer": "newton"},
            {"question": "What is the largest species of shark?", "answer": "whale shark"},
            {"question": "What is the boiling point of water in Celsius?", "answer": "100"}
        ],
        
        # Property 11 questions
        "Property 11": [
            {"question": "Who wrote 'The Theory of Relativity'?", "answer": "einstein"},
            {"question": "What is 15² - 10²?", "answer": "125"},
            {"question": "What is the capital of Germany?", "answer": "berlin"},
            {"question": "What is the smallest planet in our solar system?", "answer": "mercury"},
            {"question": "What is the human body's largest organ?", "answer": "skin"},
            {"question": "What is the symbol for the element silver?", "answer": "ag"},
            {"question": "How many legs does a spider have?", "answer": "8"},
            {"question": "What is the main language spoken in Brazil?", "answer": "portuguese"},
            {"question": "Who was the first woman to win a Nobel Prize?", "answer": "marie curie"},
            {"question": "What is the name for a young swan?", "answer": "cygnet"}
        ],
        
        # Property 13 questions
        "Property 13": [
            {"question": "Who painted 'Starry Night'?", "answer": "van gogh"},
            {"question": "What is the cube root of 27?", "answer": "3"},
            {"question": "What is the capital of Russia?", "answer": "moscow"},
            {"question": "What is the largest bird in the world?", "answer": "ostrich"},
            {"question": "What is the atomic number of carbon?", "answer": "6"},
            {"question": "How many teeth does an adult human have?", "answer": "32"},
            {"question": "What is the main ingredient in traditional hummus?", "answer": "chickpeas"},
            {"question": "What is the third planet from the sun?", "answer": "earth"},
            {"question": "Which blood type is the universal donor?", "answer": "o negative"},
            {"question": "Who wrote 'Hamlet'?", "answer": "shakespeare"}
        ],
        
        # Property 15 questions
        "Property 15": [
            {"question": "What is the world's largest island?", "answer": "greenland"},
            {"question": "What is 1000 divided by 8?", "answer": "125"},
            {"question": "What element has the chemical symbol 'Na'?", "answer": "sodium"},
            {"question": "What is the capital of Spain?", "answer": "madrid"},
            {"question": "What year was the first iPhone released?", "answer": "2007"},
            {"question": "What is the largest internal organ in the human body?", "answer": "liver"},
            {"question": "How many years make a century?", "answer": "100"},
            {"question": "Which planet is known as the 'Blue Planet'?", "answer": "earth"},
            {"question": "How many squares are on a chessboard?", "answer": "64"},
            {"question": "What is the main component of natural gas?", "answer": "methane"}
        ],
        
        # Property 17 questions
        "Property 17": [
            {"question": "Who wrote 'Pride and Prejudice'?", "answer": "jane austen"},
            {"question": "What is 5 factorial?", "answer": "120"},
            {"question": "What is the capital of Italy?", "answer": "rome"},
            {"question": "Which gas makes up most of the Earth's atmosphere?", "answer": "nitrogen"},
            {"question": "How many sides does a hexagon have?", "answer": "6"},
            {"question": "What is the largest species of big cat?", "answer": "tiger"},
            {"question": "What is the hardest natural mineral?", "answer": "diamond"},
            {"question": "What is the main ingredient in guacamole?", "answer": "avocado"},
            {"question": "What is the speed of light (rounded)?", "answer": "300000 km/s"},
            {"question": "Who discovered penicillin?", "answer": "alexander fleming"}
        ],
        
        # Property 19 questions
        "Property 19": [
            {"question": "What is the chemical symbol for iron?", "answer": "fe"},
            {"question": "What is 12 × 12?", "answer": "144"},
            {"question": "What is the capital of Canada?", "answer": "ottawa"},
            {"question": "Which planet has the Great Red Spot?", "answer": "jupiter"},
            {"question": "What is the most widely spoken language in the world?", "answer": "mandarin"},
            {"question": "What tissue connects muscle to bone?", "answer": "tendon"},
            {"question": "What is the world's largest flower?", "answer": "rafflesia"},
            {"question": "How many chambers does the human heart have?", "answer": "4"},
            {"question": "What is the largest country by land area?", "answer": "russia"},
            {"question": "What element has the chemical symbol 'He'?", "answer": "helium"}
        ],
        
        # Property 21 questions
        "Property 21": [
            {"question": "Who discovered America in 1492?", "answer": "columbus"},
            {"question": "What is the sum of angles in a triangle?", "answer": "180"},
            {"question": "What is the capital of South Korea?", "answer": "seoul"},
            {"question": "What planet is known as the 'Morning Star'?", "answer": "venus"},
            {"question": "What is the longest bone in the human body?", "answer": "femur"},
            {"question": "What is the chemical symbol for potassium?", "answer": "k"},
            {"question": "Which blood type is the universal recipient?", "answer": "ab positive"},
            {"question": "What is the smallest prime number greater than 10?", "answer": "11"},
            {"question": "What gas do plants release during photosynthesis?", "answer": "oxygen"},
            {"question": "What is the slowest moving land animal?", "answer": "sloth"}
        ],
        
        # Property 23 questions
        "Property 23": [
            {"question": "Who painted the 'Mona Lisa'?", "answer": "da vinci"},
            {"question": "What is (8 × 7) + 9?", "answer": "65"},
            {"question": "What is the capital of India?", "answer": "new delhi"},
            {"question": "What is the smallest bone in the human body?", "answer": "stapes"},
            {"question": "What is the closest star to Earth?", "answer": "sun"},
            {"question": "What is the chemical symbol for calcium?", "answer": "ca"},
            {"question": "How many days are in a leap year?", "answer": "366"},
            {"question": "What is the only mammal that can't jump?", "answer": "elephant"},
            {"question": "What is the largest organ in the human body?", "answer": "skin"},
            {"question": "What is the world's largest ocean?", "answer": "pacific"}
        ]
    }
    
    return quiz_questions

def calculate_positions():
    """Calculate positions of tiles on the board"""
    positions = []
    
    # Clockwise positions starting from bottom-right (GO)
    # Bottom row (right to left)
    for col in range(6, -1, -1):
        positions.append((6, col))
        
    # Left edge (bottom to top)
    for row in range(5, -1, -1):
        positions.append((row, 0))
        
    # Top row (left to right)
    for col in range(1, 7):
        positions.append((0, col))
        
    # Right edge (top to bottom)
    for row in range(1, 6):
        positions.append((row, 6))
        
    return positions

def init_game(root):
    """Initialize game window and state"""
    game_state['root'] = root
    root.title("Monopoly Game")
    root.geometry("1400x900")
    root.minsize(1200, 800)
    root.configure(bg="#2E2E2E")
    
    # Initialize game components
    create_initial_tiles()
    create_question_bank()
    
    # Create start screen
    create_start_screen()

def create_start_screen():
    """Create the initial start screen"""
    root = game_state['root']
    
    start_frame = tk.Frame(root, bg="#3E3E3E")
    start_frame.pack(expand=True, fill=tk.BOTH)
    
    # Game title
    title = tk.Label(start_frame, text="MONOPOLY", font=("Impact", 48), 
                    bg="#3E3E3E", fg="gold")
    title.pack(pady=40)
    
    # Buttons styling
    btn_style = {"font": ("Arial", 16), "width": 15, "height": 2, "bd": 3}
    
    # Start game button
    start_btn = tk.Button(start_frame, text="Start Game", command=start_game,
                         bg="#4CAF50", fg="white", **btn_style)
    start_btn.pack(pady=10)
    
    # Add player button
    add_player_btn = tk.Button(start_frame, text="Add Player", command=add_player,
                              bg="#2196F3", fg="white", **btn_style)
    add_player_btn.pack(pady=10)
    
    # Quit button
    quit_btn = tk.Button(start_frame, text="Quit Game", command=root.destroy,
                        bg="#F44336", fg="white", **btn_style)
    quit_btn.pack(pady=10)
    
    # Store reference to start frame for later removal
    game_state['start_frame'] = start_frame

def create_game_ui():
    """Create the main game interface"""
    root = game_state['root']
    
    # Remove start screen
    if 'start_frame' in game_state:
        game_state['start_frame'].destroy()
    
    # Create main frame
    main_frame = tk.Frame(root, bg="#2E2E2E")
    main_frame.pack(fill=tk.BOTH, expand=True)
    game_state['main_frame'] = main_frame
    
    # Game board (canvas)
    canvas = tk.Canvas(main_frame, bg="white", highlightthickness=0)
    canvas.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
    game_state['canvas'] = canvas
    
    # Control panel
    control_frame = tk.Frame(main_frame, bg="#3E3E3E")
    control_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
    
    # Configure grid weights for responsive layout
    main_frame.grid_columnconfigure(0, weight=4)  # Game board takes more space
    main_frame.grid_columnconfigure(1, weight=1)  # Control panel takes less space
    main_frame.grid_rowconfigure(0, weight=1)
    
    # Add controls to the panel
    create_control_panel(control_frame)
    
    # Bind resize event
    canvas.bind("<Configure>", on_resize)
    
    # Draw initial board
    draw_board()
    
    # Start timer
    start_timer()

def create_control_panel(parent):
    """Create control panel with game controls"""
    # Timer display
    timer_label = tk.Label(parent, text="Time: 00:00", font=("Arial", 16), 
                          bg="#3E3E3E", fg="white")
    timer_label.pack(pady=10)
    game_state['timer_label'] = timer_label
    
    # Dice display
    dice_label = tk.Label(parent, text="Dice: ", font=("Arial", 16), 
                         bg="#3E3E3E", fg="white")
    dice_label.pack(pady=10)
    game_state['dice_label'] = dice_label
    
    # Current player display
    current_player_label = tk.Label(parent, text="Current Player: ", font=("Arial", 14), 
                                   bg="#3E3E3E", fg="white")
    current_player_label.pack(pady=10)
    game_state['current_player_label'] = current_player_label
    
    # Scores display
    score_label = tk.Label(parent, text="Scores:\n", font=("Arial", 14), 
                          bg="#3E3E3E", fg="white")
    score_label.pack(pady=10)
    game_state['score_label'] = score_label
    
    # Roll dice button
    roll_btn = tk.Button(parent, text="🎲 Roll Dice", command=roll_dice_turn,
                        font=("Arial", 14), bg="#2196F3", fg="white", bd=3)
    roll_btn.pack(pady=20)
    game_state['roll_btn'] = roll_btn
    
    # Quit button
    quit_btn = tk.Button(parent, text="🚪 Quit Game", command=game_state['root'].destroy,
                        font=("Arial", 12), bg="#F44336", fg="white", bd=3)
    quit_btn.pack(side=tk.BOTTOM, pady=20)

def on_resize(event):
    """Handle window resize events"""
    draw_board()

def draw_board():
    """Draw the game board on canvas"""
    canvas = game_state['canvas']
    canvas.delete("all")
    
    # Get canvas dimensions
    w = canvas.winfo_width()
    h = canvas.winfo_height()
    
    # Calculate tile size based on canvas size
    tile_size = min(w//8, h//8)
    
    # Get tile positions
    positions = calculate_positions()
    
    # Draw all tiles
    for idx, (row, col) in enumerate(positions):
        if idx >= len(tiles):
            continue
            
        tile = tiles[idx]
        x = col * tile_size + tile_size//2
        y = row * tile_size + tile_size//2
        
        # Draw tile background
        draw_tile(tile, x, y, tile_size, idx)
    
    # Draw players on their positions
    draw_players(positions, tile_size)

def open_url(url):
    """Safely open a URL in the default browser"""
    try:
        webbrowser.open(url)
    except Exception as e:
        messagebox.showerror("Error", f"Could not open URL: {e}")

def draw_tile(tile, x, y, tile_size, idx):
    """Draw a single tile on the board"""
    canvas = game_state['canvas']
    
    # Determine if any player is on this tile
    is_highlighted = any(p['position'] == idx and p['started'] for p in game_state['players'])
    
    # Tile background
    color = tile['color']
    outline_color = "gold" if is_highlighted else "black"
    outline_width = 3 if is_highlighted else 1
    
    # Create rectangle for tile
    canvas.create_rectangle(
        x-tile_size//2, y-tile_size//2,
        x+tile_size//2, y+tile_size//2,
        fill=color, outline=outline_color, width=outline_width,
        tags=(f"tile_{idx}")
    )
    
    # Clickable area for hyperlink
    canvas.tag_bind(f"tile_{idx}", "<Button-1>", 
                   lambda e, url=tile['hyperlink']: open_url(url))
    
    # Tile content
    if tile.get('is_corner', False):
        # Draw corner tile with symbol
        if tile.get('teleport', False):
            teleport_text = f"{tile.get('symbol', '')}\n{tile['name']}\n⚡ TELEPORT ⚡"
        else:
            teleport_text = f"{tile.get('symbol', '')}\n{tile['name']}"
            
        canvas.create_text(
            x, y, 
            text=teleport_text, 
            font=("Arial Bold", 10), 
            fill="black",
            justify=tk.CENTER
        )
    else:
        # Draw regular tile with 3D text effect
        # Shadow text (for 3D effect)
        canvas.create_text(
            x+2, y+2, 
            text=tile['name'], 
            font=("Arial Bold", 9),
            fill="#888888",
            width=tile_size-20
        )
        
        # Main text
        canvas.create_text(
            x, y, 
            text=tile['name'], 
            font=("Arial Bold", 9),
            fill="black",
            width=tile_size-20,
            tags=f"text_{idx}"
        )
        
        # Add quiz indicator if applicable
        if tile.get('has_quiz', False):
            canvas.create_text(
                x, y+15, 
                text="❓", 
                font=("Arial", 12),
                fill="red"
            )

def draw_players(positions, tile_size):
    """Draw all players on their positions"""
    canvas = game_state['canvas']
    
    # Draw each player
    for i, player in enumerate(game_state['players']):
        if player['started']:
            # Get player position
            pos = player['position']
            
            # Skip if position is invalid
            if pos >= len(positions):
                continue
                
            # Calculate coordinates
            row, col = positions[pos]
            
            # Add offset based on player index to avoid overlap
            offset_x = (i % 2) * 20 - 10
            offset_y = (i // 2) * 20 - 10
            
            x = col * tile_size + tile_size//2 + offset_x
            y = row * tile_size + tile_size//2 + offset_y
            
            # Draw player icon based on their assigned shape
            icon_type = game_state['player_icons'][i % len(game_state['player_icons'])]
            
            if icon_type == "circle":
                canvas.create_oval(
                    x-12, y-12, x+12, y+12,
                    fill=player['color'], outline="black"
                )
            elif icon_type == "square":
                canvas.create_rectangle(
                    x-12, y-12, x+12, y+12,
                    fill=player['color'], outline="black"
                )
            elif icon_type == "triangle":
                canvas.create_polygon(
                    x, y-12, x+12, y+12, x-12, y+12,
                    fill=player['color'], outline="black"
                )
            elif icon_type == "star":
                # Simple star approximation
                canvas.create_text(
                    x, y, text="★", 
                    font=("Arial", 20), 
                    fill=player['color']
                )

def highlight_tile(pos):
    """Highlight a tile when a player lands on it"""
    canvas = game_state['canvas']
    
    # Update tile outline
    canvas.itemconfig(f"tile_{pos}", outline="gold", width=4)
    
    # Schedule reset of highlight
    game_state['root'].after(
        1000, 
        lambda: canvas.itemconfig(f"tile_{pos}", outline="black", width=1)
    )

def teleport_player(player):
    """Teleport a player to a random non-corner tile"""
    # Get list of non-corner, non-GO tile positions
    non_corner_positions = [i for i in range(len(tiles)) 
                           if i != 0 and not (tiles[i].get('is_corner', False))]
    
    if not non_corner_positions:
        return  # No valid teleport destinations
    
    # Select random destination
    new_pos = random.choice(non_corner_positions)
    old_pos = player['position']
    player['position'] = new_pos
    
    # Update visited tiles
    game_state['visited_tiles'].add(new_pos)
    
    # Highlight the destination tile
    highlight_tile(new_pos)
    
    # Show teleport animation
    flash_teleport(old_pos, new_pos)
    
    # Show teleport message with destination
    messagebox.showinfo(
        "TELEPORT!", 
        f"{player['name']} teleported from {tiles[old_pos]['name']} to {tiles[new_pos]['name']}!"
    )
    
    # Handle landing on destination tile
    handle_landing(player, new_pos)

def flash_teleport(old_pos, new_pos):
    """Create a visual flash effect for teleportation"""
    canvas = game_state['canvas']
    
    # Flash the old position
    canvas.itemconfig(f"tile_{old_pos}", fill="#FFD700")  # Gold
    
    # Schedule the flash at new position
    def flash_new():
        canvas.itemconfig(f"tile_{new_pos}", fill="#FFD700")  # Gold
        # Restore original colors
        game_state['root'].after(300, lambda: restore_colors(old_pos, new_pos))
    
    game_state['root'].after(300, flash_new)

def restore_colors(old_pos, new_pos):
    """Restore original tile colors after teleport effect"""
    canvas = game_state['canvas']
    canvas.itemconfig(f"tile_{old_pos}", fill=tiles[old_pos]['color'])
    canvas.itemconfig(f"tile_{new_pos}", fill=tiles[new_pos]['color'])

def handle_landing(player, position):
    """Handle the effects of landing on a tile"""
    tile = tiles[position]
    
    # Handle different tile effects
    if tile.get('has_quiz', False):
        # Mark that player is on quiz tile (will be handled on next turn)
        player['on_quiz_tile'] = True
        messagebox.showinfo(
            "Quiz Tile", 
            f"You landed on a quiz tile: {tile['name']}\n" +
            "You'll answer a question on your next turn!"
        )
    elif not tile.get('is_corner', False):
        # Regular property tile
        player['score'] += tile['value']
        messagebox.showinfo(
            "Property Tile", 
            f"{player['name']} landed on {tile['name']}\n" +
            f"Score changed by: {tile['value']}"
        )

def start_game():
    """Start the game"""
    # Check if we have at least one player
    if len(game_state['players']) < 1:
        messagebox.showwarning("Players Needed", "Add at least 1 player to start!")
        return
    
    # Get timer value
    game_state['timer_value'] = simpledialog.askinteger(
        "Game Timer", 
        "Enter game duration in minutes:",
        minvalue=1, maxvalue=60,
        initialvalue=5
    )
    
    # If user cancels timer dialog, use default value
    if game_state['timer_value'] is None:
        game_state['timer_value'] = 5
    
    # Set game state
    game_state['start_time'] = time.time()
    game_state['game_started'] = True
    
    # Create game UI
    create_game_ui()
    
    # Update player display
    update_current_player()
    update_scores()

def add_player():
    """Add a new player to the game"""
    # Check maximum players
    if len(game_state['players']) >= 4:
        messagebox.showinfo("Max Players", "Maximum 4 players allowed!")
        return
    
    # Get player name
    name = simpledialog.askstring("Add Player", "Enter player name:")
    if not name:
        return
    
    # Get initial score
    initial_score = simpledialog.askinteger(
        "Initial Score", 
        f"Enter initial score for {name}:",
        initialvalue=0
    )
    
    # Default to 0 if cancelled
    if initial_score is None:
        initial_score = 0
    
    # Add player to game
    game_state['players'].append({
        'name': name,
        'score': initial_score,
        'color': random.choice(["red", "blue", "green", "yellow", "purple", "orange"]),
        'position': 0,
        'started': False,
        'on_quiz_tile': False  # Track if player is on a quiz tile
    })

def roll_dice_turn():
    """Handle dice roll for current player's turn"""
    # Check if game is started
    if not game_state['game_started']:
        return
    
    # Get current player
    current_player = game_state['current_player']
    player = game_state['players'][current_player]
    
    # Roll dice
    roll = random.randint(1, 6)
    game_state['dice_label'].config(text=f"🎲 Dice: {roll}")
    
    # Handle player's move based on roll
    if not player['started']:
        # Player needs to roll 1 to start
        if roll == 1:
            player['started'] = True
            player['position'] = 0  # Start at GO (position 0)
            player['on_quiz_tile'] = False  # Initialize quiz tile flag
            messagebox.showinfo("Started!", f"{player['name']} has started at GO!")
        else:
            messagebox.showinfo("Roll Again", f"{player['name']} needs 1 to start. Rolled {roll}.")
    else:
        # Move player
        move_player(player, roll)
    
    # Update scores
    update_scores()
    
    # Redraw board
    draw_board()
    
    # Move to next player
    next_player_turn()

def move_player(player, roll):
    """Move a player according to dice roll"""
    # Store current position to check if it was a quiz tile
    current_pos = player['position']
    current_tile = tiles[current_pos]
    
    # Check if player is currently on a quiz tile
    # If so, we need to handle the quiz before moving
    if current_tile.get('has_quiz', False) and player.get('on_quiz_tile', False):
        # Handle quiz before moving
        quiz_result = handle_quiz(current_tile, player)
        # If quiz was cancelled, don't move player
        if quiz_result == False:
            return
    
    # Calculate new position
    new_pos = (player['position'] + roll) % len(tiles)
    player['position'] = new_pos
    
    # Add position to visited tiles
    game_state['visited_tiles'].add(new_pos)
    
    # Get tile at new position
    tile = tiles[new_pos]
    
    # Highlight tile
    highlight_tile(new_pos)
    
    # Check if this is a teleport corner
    if tile.get('is_corner', False) and tile.get('teleport', False):
        # Handle teleportation
        teleport_player(player)
        return
    
    # Mark if the player is now on a quiz tile (for next turn)
    player['on_quiz_tile'] = tile.get('has_quiz', False)
    
    # Handle landing on this tile
    handle_landing(player, new_pos)

def handle_quiz(tile, player):
    """Handle quiz for a player leaving a quiz tile"""
    tile_name = tile['name']
    
    # Check if there are questions for this tile
    if tile_name not in quiz_questions or not quiz_questions[tile_name]:
        messagebox.showinfo("No Questions", "No more questions for this tile!")
        return True
    
    # Initialize used questions for this tile if not already
    if tile_name not in game_state['used_questions']:
        game_state['used_questions'][tile_name] = set()
    
    # Get available questions
    available_questions = [
        q for i, q in enumerate(quiz_questions[tile_name]) 
        if i not in game_state['used_questions'][tile_name]
    ]
    
    # Check if we have any available questions
    if not available_questions:
        messagebox.showinfo("No Questions", "All questions used for this tile!")
        return True
    
    # Select a random question
    question_index = random.randint(0, len(available_questions) - 1)
    question = available_questions[question_index]
    
    # Mark question as used
    original_index = quiz_questions[tile_name].index(question)
    game_state['used_questions'][tile_name].add(original_index)
    
    # Ask question
    answer = simpledialog.askstring(
        "Quiz Time!", 
        f"Quiz from {tile_name}:\n\n{question['question']}\n\nEnter your answer:"
    )
    
    # If user cancels the dialog, don't move the player
    if answer is None:
        return False
    
    # Check answer
    if answer.strip().lower() == question['answer'].lower():
        player['score'] += 100
        messagebox.showinfo("Correct!", "Correct answer! +100 points!")
    else:
        player['score'] -= 100
        messagebox.showinfo("Wrong!", f"Wrong answer! -100 points!\nCorrect answer was: {question['answer']}")
    
    # Clear the quiz flag now that they've answered
    player['on_quiz_tile'] = False
    return True

def next_player_turn():
    """Move to next player's turn"""
    # Move to next player
    game_state['current_player'] = (game_state['current_player'] + 1) % len(game_state['players'])
    
    # Update current player display
    update_current_player()
    
    # Check if game should end
    if len(game_state['visited_tiles']) >= len(tiles):
        declare_winner()

def update_current_player():
    """Update the current player display"""
    if 'current_player_label' in game_state and len(game_state['players']) > 0:
        current_player = game_state['players'][game_state['current_player']]
        game_state['current_player_label'].config(
            text=f"Current Player: {current_player['name']}"
        )

def update_scores():
    """Update the scores display"""
    if 'score_label' in game_state:
        # Create score text
        scores = "\n".join([
            f"{p['name']}: {p['score']}" 
            for p in game_state['players']
        ])
        
        game_state['score_label'].config(text=f"💰 Scores:\n{scores}")

def start_timer():
    """Start the game timer"""
    def timer_thread():
        """Timer thread function"""
        # Convert minutes to seconds
        total_seconds = game_state['timer_value'] * 60
        end_time = game_state['start_time'] + total_seconds
        
        while game_state['game_started'] and time.time() < end_time:
            # Calculate remaining time
            remaining = int(end_time - time.time())
            if remaining <= 0:
                break
                
            # Update timer display
            mins, secs = divmod(remaining, 60)
            
            # Use after method to safely update UI from thread
            game_state['root'].after(
                0, 
                lambda m=mins, s=secs: game_state['timer_label'].config(
                    text=f"⏱️ Time Left: {m:02}:{s:02}"
                )
            )
            
            # Sleep for a second
            time.sleep(1)
        
        # End game when timer expires
        if game_state['game_started']:
            game_state['root'].after(0, declare_winner)
    
    # Start timer in a separate thread
    threading.Thread(target=timer_thread, daemon=True).start()

def declare_winner():
    """Declare the winner and end the game"""
    # Set game as finished
    game_state['game_started'] = False
    
    # Find player with highest score
    max_score = max(p['score'] for p in game_state['players'])
    winners = [p for p in game_state['players'] if p['score'] == max_score]
    
    # Create report
    report = "Final Report:\n\n"
    
    # Add player scores (sorted by score)
    for p in sorted(game_state['players'], key=lambda x: x['score'], reverse=True):
        report += f"{p['name']}: {p['score']}\n"
    
    # Add winner information
    if len(winners) == 1:
        report += f"\nWinner: {winners[0]['name']} 🏆"
    else:
        names = ", ".join([w['name'] for w in winners])
        report += f"\nTie between {names} 🤝"
    
    # Show game over message
    messagebox.showinfo("Game Over!", report)
    
    # Close game
    game_state['root'].destroy()

def main():
    """Main function to start the game"""
    # Create root window
    root = tk.Tk()
    
    # Initialize game
    init_game(root)
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()
