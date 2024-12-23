import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from simpleai.search import SearchProblem, astar, breadth_first, depth_first, greedy, backtrack, uniform_cost, iterative_limited_depth_first
import time
from queue import PriorityQueue

# Kích thước mỗi ô trên level
TILE_SIZE = 45

# Path to assets và levels
ASSET_PATH = "../assets/"
LEVEL_PATH = "../Sokoban/levels/"

# Manage assets
class AssetManager:
    def __init__(self):
        self.assets = self.load_assets()

    def load_assets(self):
        return {
            "wall": ImageTk.PhotoImage(Image.open(ASSET_PATH + "wall.png").resize((TILE_SIZE, TILE_SIZE))),
            "box": ImageTk.PhotoImage(Image.open(ASSET_PATH + "box.png").resize((TILE_SIZE, TILE_SIZE))),
            "box_on_target": ImageTk.PhotoImage(
                Image.open(ASSET_PATH + "box_on_target.png").resize((TILE_SIZE, TILE_SIZE))
            ),
            "ground": ImageTk.PhotoImage(Image.open(ASSET_PATH + "ground.png").resize((TILE_SIZE, TILE_SIZE))),
            "player": ImageTk.PhotoImage(Image.open(ASSET_PATH + "player.png").resize((TILE_SIZE, TILE_SIZE))),
            "target": ImageTk.PhotoImage(Image.open(ASSET_PATH + "target.png").resize((TILE_SIZE, TILE_SIZE))),
        }

# Manage levels
class Level:
    def __init__(self, level_number):
        self.number = level_number
        self.data = self.load_level(f"level{level_number}.txt")

    def load_level(self, level_number):
        with open(LEVEL_PATH + level_number, "r") as file:
            level_data = file.readlines()
        # Loại bỏ ký tự xuống dòng và giữ nguyên khoảng trắng
        return [list(line.rstrip('\n')) for line in level_data]

# Render interface like background and game zone
class GameCanvas:
    def __init__(self, root, width, height):
        self.canvas = tk.Canvas(root, width=width, height=height)
        self.canvas.pack(fill="both", expand=True)

    def draw_background(self, image_path):
        bg_image = Image.open(image_path).resize((self.canvas.winfo_reqwidth(), self.canvas.winfo_reqheight()), Image.LANCZOS)
        
        self.bg_image = ImageTk.PhotoImage(bg_image)
        self.canvas.create_image(0, 0, image=self.bg_image, anchor="nw")

    def draw_level(self, level, assets):
        for y, row in enumerate(level.data):
            for x, tile in enumerate(row):
                if tile == "#":     # Wall
                    self.canvas.create_image(x * TILE_SIZE, y * TILE_SIZE, image=assets["wall"], anchor=tk.NW)
                elif tile == " ":   # Ground
                    self.canvas.create_image(x * TILE_SIZE, y * TILE_SIZE, image=assets["ground"], anchor=tk.NW)
                elif tile == "@":   # Player
                    self.canvas.create_image(x * TILE_SIZE, y * TILE_SIZE, image=assets["ground"], anchor=tk.NW)
                    self.canvas.create_image(x * TILE_SIZE, y * TILE_SIZE, image=assets["player"], anchor=tk.NW)
                elif tile == "$":   # Box
                    self.canvas.create_image(x * TILE_SIZE, y * TILE_SIZE, image=assets["ground"], anchor=tk.NW)
                    self.canvas.create_image(x * TILE_SIZE, y * TILE_SIZE, image=assets["box"], anchor=tk.NW)
                elif tile == ".":   # Target
                    self.canvas.create_image(x * TILE_SIZE, y * TILE_SIZE, image=assets["ground"], anchor=tk.NW)
                    self.canvas.create_image(x * TILE_SIZE, y * TILE_SIZE, image=assets["target"], anchor=tk.NW)
                elif tile == "*":   # Box on target
                    self.canvas.create_image(x * TILE_SIZE, y * TILE_SIZE, image=assets["target"], anchor=tk.NW)
                    self.canvas.create_image(x * TILE_SIZE, y * TILE_SIZE, image=assets["box_on_target"], anchor=tk.NW)

class SokobanProblem(SearchProblem):
    def __init__(self, level):
        self.level = level
        self.initial_state = self.get_initial_state()
        self.goal = self.get_targets_positions()
        super(SokobanProblem, self).__init__(initial_state = self.initial_state)

    def get_initial_state(self):
        player = None
        boxes = []
        for x, row in enumerate(self.level.data):
            for y, cell in enumerate(row):
                if cell == "@":
                    player = (x, y)
                elif cell == "$" or cell == "*":
                    boxes.append((x, y))
        return (player, tuple(boxes))

    def get_targets_positions(self):
        targets = []
        for i, row in enumerate(self.level.data):
            for j, tile in enumerate(row):
                if tile == ".":
                    targets.append((i, j))
        return tuple(targets) 
    
    def canMove(self, x, y):
        if self.level.data[x][y] not in ["#"]:
            return True
    
    def is_deadlock(self, box_x, box_y):
        if (self.level.data[box_x-1][box_y] in ['#'] and self.level.data[box_x][box_y-1] in ['#']) or \
            (self.level.data[box_x-1][box_y] in ['#'] and self.level.data[box_x][box_y+1] in ['#']) or \
            (self.level.data[box_x+1][box_y] in ['#'] and self.level.data[box_x][box_y-1] in ['#']) or \
            (self.level.data[box_x+1][box_y] in ['#'] and self.level.data[box_x][box_y+1] in ['#']):
            return True
        return False
    
    def is_goal(self, state):
        _, boxes = state
        return set(boxes) == set(self.goal)  # Kiểm tra nếu các box trùng với các targets

    def update_valid_move(self, state, dx, dy):
        player, boxes = state
        cur_x, cur_y = player
        next_x, next_y = cur_x + dx, cur_y + dy
        # Kiểm tra nếu vị trí mới hợp lệ (không đụng wall)
        if self.level.data[next_x][next_y] not in ["#"] and (next_x, next_y) not in boxes: # Không đẩy box
            return ((next_x, next_y), boxes)
        elif (next_x, next_y) in boxes:  # Nếu vị trí là box
            new_box_x, new_box_y = next_x + dx, next_y + dy

            # Kiểm tra nếu box có thể đẩy được
            if self.level.data[new_box_x][new_box_y] not in ["#"] and (new_box_x, new_box_y) not in boxes and not self.is_deadlock(new_box_x, new_box_y):
                new_boxes = list(boxes)
                new_boxes.append((new_box_x, new_box_y))
                new_boxes.remove((next_x, next_y))  
                return ((next_x, next_y), tuple(new_boxes))
            else:
                return (player, boxes)    
        else:
            return (player, boxes)

    # Các hành động mà player có thể thực hiện
    def actions(self, state):
        old_state = state
        moves = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
        actions = []
        for action, (dx, dy) in moves.items():
            new_state = self.update_valid_move(state, dx, dy)
            if new_state != old_state:
                actions.append(action)
        return actions
    
    def result(self, state, action):
        dx, dy = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}[action]
        new_state = self.update_valid_move(state, dx, dy)
        return new_state
       
    def cost(self, state, action, state2):
        return 1

    def heuristic(self, state):
        player, boxes = state
            
        total_distance = 0
        for box in tuple(boxes):
            distances = [abs(box[0] - goal[0]) + abs(box[1] - goal[1]) for goal in self.goal]
            total_distance += min(distances)
        return total_distance
    
    
class SokobanGame:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sokoban Game")

        # Tạo canvas chính
        self.main_canvas = GameCanvas(self.root, 750, 530)
        self.main_canvas.draw_background(ASSET_PATH + "BG.jpg")

        # Quản lý assets và level
        self.assets = AssetManager()
        self.level = Level(0)
        self.targets = self.get_targets_positions()
        # Thêm các thành phần game
        self.create_menu()
        self.create_game_canvas()
    
    # Kiểm tra nếu một box bị deadlock do nằm ở góc giữa hai wall/box.
    def is_deadlock(self, box_x, box_y):
        if (self.level.data[box_x-1][box_y] in ['#'] and self.level.data[box_x][box_y-1] in ['#']) or \
            (self.level.data[box_x-1][box_y] in ['#'] and self.level.data[box_x][box_y+1] in ['#']) or \
            (self.level.data[box_x+1][box_y] in ['#'] and self.level.data[box_x][box_y-1] in ['#']) or \
            (self.level.data[box_x+1][box_y] in ['#'] and self.level.data[box_x][box_y+1] in ['#']):
            return True
        return False

    def check_all_boxes_for_deadlock(self):
        for i, row in enumerate(self.level.data):
            for j, tile in enumerate(row):
                if tile == '$':
                    if self.is_deadlock(i, j):
                        return True
        return False
    
    def canMove(self, x, y):
        return self.level.data[x][y] not in ["#", "$", "*"]

    def get_player_positions(self):
        for i, row in enumerate(self.level.data):
            for j, tile in enumerate(row):
                if tile == "@":
                    return i, j

    def get_targets_positions(self):
        targets = []
        for i, row in enumerate(self.level.data):
            for j, tile in enumerate(row):
                if tile == ".":
                    targets.append((i, j))
        return targets
          
    def is_Completed(self):
        for x, y in self.targets:
            if self.level.data[x][y] != "*":
                return False
        return True
    
    def update_position(self, old_x, old_y, new_x, new_y, symbol):
        self.level.data[old_x][old_y] = " "
        self.level.data[new_x][new_y] = symbol

        # xử lý targets
        if (old_x, old_y) in self.targets:
            self.level.data[old_x][old_y] = "."
        else:
            self.level.data[old_x][old_y] = " "
    
    def update_ui(self):
        self.game_canvas.canvas.delete(tk.ALL)
        self.game_canvas.draw_level(self.level, self.assets.assets)

    # Điều khiển player bằng phím mũi tên
    def move_player_and_box(self, dx, dy):
        cur_x, cur_y = self.get_player_positions()
        next_x, next_y = cur_x + dx, cur_y + dy

        # Biến để theo dõi xem có thay đổi không
        moved = False

        # Kiểm tra nếu vị trí mới hợp lệ (không đụng wall)
        if self.canMove(next_x, next_y):
            self.update_position(cur_x, cur_y, next_x, next_y, "@")
            moved = True
        elif self.level.data[next_x][next_y] in ["$", "*"]:  # Nếu vị trí là box
            box_x, box_y = next_x, next_y
            new_box_x, new_box_y = box_x + dx, box_y + dy

            # Kiểm tra nếu box có thể đẩy được
            if self.canMove(new_box_x, new_box_y):                
                
                # Cập nhật vị trí nhân vật và box
                self.update_position(cur_x, cur_y, box_x, box_y, "@")
                
                # Xử lý box
                if self.level.data[new_box_x][new_box_y] == " ":
                    self.level.data[new_box_x][new_box_y] = "$"
                elif self.level.data[new_box_x][new_box_y] == ".":
                    self.level.data[new_box_x][new_box_y] = "*"
                
                moved = True
            else:
                print("Cannot push box")

        # Cập nhật UI ngay lập tức sau mỗi lần di chuyển
        if moved:
            self.update_ui()
            
            if self.is_Completed():
                messagebox.showinfo("Successful", "Game solved successfully!") 

            # Kiểm tra trạng thái Deadlock sau khi di chuyển
            elif self.check_all_boxes_for_deadlock():
                messagebox.showerror("Deadlock", "Your game is now deadlock!")        

    # Điều khiển nhân vật di chuyển theo hướng.
    # direction: U (lên), D (xuống), L (trái), R (phải).        
    def move(self, action):
        dx, dy = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}[action]
        self.move_player_and_box(dx, dy)

    # Navigation and Algorithms buttons
    def create_menu(self):
        menu_x = 40
        menu_y = 70
        menu_spacing = 60

        a_star_button = tk.Button(self.root, text="A*", bg="#aaffaa", font=("Arial", 12), width=12, command=self.btn_Astar_click)
        self.main_canvas.canvas.create_window(menu_x, menu_y, window=a_star_button, anchor="nw")
        menu_y += menu_spacing

        bfs_button = tk.Button(self.root, text="BFS", bg="#aaffaa", font=("Arial", 12), width=12, command=self.btn_BFS_click)
        self.main_canvas.canvas.create_window(menu_x, menu_y, window=bfs_button, anchor="nw")
        menu_y += menu_spacing

        dfs_button = tk.Button(self.root, text="DFS", bg="#aaffaa", font=("Arial", 12), width=12, command=self.btn_DFS_click)
        self.main_canvas.canvas.create_window(menu_x, menu_y, window=dfs_button, anchor="nw")
        menu_y += menu_spacing

        greedy_button = tk.Button(self.root, text="Greedy", bg="#aaffaa", font=("Arial", 12), width=12, command=self.btn_greedy_click)
        self.main_canvas.canvas.create_window(menu_x, menu_y, window=greedy_button, anchor="nw")
        menu_y += menu_spacing

        uniform_cost_button = tk.Button(self.root, text="Uniform cost", bg="#aaffaa", font=("Arial", 12), width=12, command=self.btn_uniform_cost_click)
        self.main_canvas.canvas.create_window(menu_x, menu_y, window=uniform_cost_button, anchor="nw")
        menu_y += menu_spacing

        ids_button = tk.Button(self.root, text="Ids", bg="#aaffaa", font=("Arial", 12), width=12, command=self.btn_ids_click)
        self.main_canvas.canvas.create_window(menu_x, menu_y, window=ids_button, anchor="nw")


    # Game zone
    def create_game_canvas(self):
        game_x = 220
        game_y = 50
        game_canvas_width = len(self.level.data[0]) * TILE_SIZE
        game_canvas_height = len(self.level.data) * TILE_SIZE

        self.game_canvas = GameCanvas(self.main_canvas.canvas, game_canvas_width, game_canvas_height)
        self.game_canvas.canvas.place(x=game_x, y=game_y)
        self.game_canvas.draw_level(self.level, self.assets.assets)

        # Level navigation buttons
        nav_x = game_x
        nav_y = game_y + game_canvas_height + 40

        prev_button = tk.Button(self.root, text="Previous", bg="#d9d9d9", font=("Arial", 10), width=10, command=self.btn_pre_click)
        self.main_canvas.canvas.create_window(nav_x, nav_y, window=prev_button, anchor="nw")

        reset_button = tk.Button(self.root, text="Reset", bg="#d9d9d9", font=("Arial", 10), width=10, command=self.btn_reset_click)
        self.main_canvas.canvas.create_window(nav_x + 150, nav_y, window=reset_button, anchor="nw")

        next_button = tk.Button(self.root, text="Next", bg="#d9d9d9", font=("Arial", 10), width=10, command=self.btn_next_click)
        self.main_canvas.canvas.create_window(nav_x + 300, nav_y, window=next_button, anchor="nw")

    def btn_reset_click(self):
        level_cur_number = self.level.number
        self.game_canvas.canvas.delete(tk.ALL)
        self.level = Level(level_cur_number)
        self.create_game_canvas()
    
    def btn_pre_click(self):
        self.game_canvas.canvas.destroy()
        self.main_canvas.canvas.destroy()
        level_next_number = (self.level.number - 1) % 9
        self.main_canvas = GameCanvas(self.root, 750, 530)
        self.main_canvas.draw_background(ASSET_PATH + "BG.jpg")
        self.level = Level(level_next_number)
        self.targets = self.get_targets_positions()
        self.create_menu()
        self.create_game_canvas()

    def btn_next_click(self):
        self.game_canvas.canvas.destroy()
        self.main_canvas.canvas.destroy()
        level_next_number = (self.level.number + 1) % 9
        self.main_canvas = GameCanvas(self.root, 750, 530)
        self.main_canvas.draw_background(ASSET_PATH + "BG.jpg")
        self.level = Level(level_next_number)
        self.targets = self.get_targets_positions()
        self.create_menu()
        self.create_game_canvas()

    def btn_Astar_click(self):
        try:
            problem = SokobanProblem(self.level)
            
            # Use astar search
            result = astar(problem, graph_search=True)
            if not result:
                messagebox.showwarning("Warning", "No solution found!")
            path = [x[0] for x in result.path()]
            for action in path[1:]:
                print(action + ", ", end=" ")
                self.move(action)
                self.update_ui()  # Update interface
                self.root.update()  # Ép buộc Tkinter vẽ lại
                time.sleep(0.08)  # Pause to show movement
            print()
        
        except Exception as e:
            print(f"Error solving Sokoban: {e}")
            messagebox.showerror("Error", str(e))

    def btn_BFS_click(self):
        try:
            problem = SokobanProblem(self.level)
            
            # Use BFS search
            result = breadth_first(problem, graph_search=True)
            if not result:
                messagebox.showwarning("Warning", "No solution found!")
            path = [x[0] for x in result.path()]
            for action in path[1:]:
                print(action + ", ", end=" ")
                self.move(action)
                self.update_ui()  # Update interface
                self.root.update()  # Ép buộc Tkinter vẽ lại
                time.sleep(0.08)  # Pause to show movement
            print()
        
        except Exception as e:
            print(f"Error solving Sokoban: {e}")
            messagebox.showerror("Error", str(e))

    def btn_DFS_click(self):
        try:
            problem = SokobanProblem(self.level)
            
            # Use BFS search
            result = depth_first(problem, graph_search=True)
            if not result:
                messagebox.showwarning("Warning", "No solution found!")
            path = [x[0] for x in result.path()]
            for action in path[1:]:
                print(action + ", ", end=" ")
                self.move(action)
                self.update_ui()  # Update interface
                self.root.update()  # Ép buộc Tkinter vẽ lại
                time.sleep(0.08)  # Pause to show movement
            print()
        
        except Exception as e:
            print(f"Error solving Sokoban: {e}")
            messagebox.showerror("Error", str(e))
            
    def btn_greedy_click(self):
        try:
            problem = SokobanProblem(self.level)
            
            # Use BFS search
            result = greedy(problem, graph_search=True)
            if not result:
                messagebox.showwarning("Warning", "No solution found!")
            print (result)
            path = [x[0] for x in result.path()]
            for action in path[1:]:
                print(action + ", ", end=" ")
                self.move(action)
                self.update_ui()  # Update interface
                self.root.update()  # Ép buộc Tkinter vẽ lại
                time.sleep(0.08)  # Pause to show movement
            print()
        
        except Exception as e:
            print(f"Error solving Sokoban: {e}")
            messagebox.showerror("Error", str(e))

    def btn_uniform_cost_click(self):
        try:
            problem = SokobanProblem(self.level)
            
            # Use BFS search
            result = uniform_cost(problem, graph_search=True)
            if not result:
                messagebox.showwarning("Warning", "No solution found!")
            path = [x[0] for x in result.path()]
            for action in path[1:]:
                print(action + ", ", end=" ")
                self.move(action)
                self.update_ui()  # Update interface
                self.root.update()  # Ép buộc Tkinter vẽ lại
                time.sleep(0.08)  # Pause to show movement
            print()
        
        except Exception as e:
            print(f"Error solving Sokoban: {e}")
            messagebox.showerror("Error", str(e))

    def btn_ids_click(self):
        try:
            problem = SokobanProblem(self.level)
            
            # Use BFS search
            result = iterative_limited_depth_first(problem, graph_search=True)
            if not result:
                messagebox.showwarning("Warning", "No solution found!")
            path = [x[0] for x in result.path()]
            for action in path[1:]:
                print(action + ", ", end=" ")
                self.move(action)
                self.update_ui()  # Update interface
                self.root.update()  # Ép buộc Tkinter vẽ lại
                time.sleep(0.08)  # Pause to show movement
            print()
        
        except Exception as e:
            print(f"Error solving Sokoban: {e}")
            messagebox.showerror("Error", str(e))


    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    game = SokobanGame()
    game.root.bind("<Up>", lambda event: game.move('U'))
    game.root.bind("<Down>", lambda event: game.move('D'))
    game.root.bind("<Left>", lambda event: game.move('L'))
    game.root.bind("<Right>", lambda event: game.move('R'))
    game.run()
