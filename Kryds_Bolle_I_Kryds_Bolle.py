import pygame as pg
from pygame import mouse
from random import choice as choose


# pygame setup
if __name__ == "__main__":
    pg.init()

    screen = pg.display.set_mode((1280, 720))
    clock = pg.time.Clock()
    dt = 0
running = True

r = 40
mouseDown = False

boardstart = (310,32)
_space_size = 220

turn = choose(["X", "O"])

last_space_index = None

class Board():
    def __init__(self, boardstart, space_size, color, is_clickable = True, has_children = False, parent = None):
        self.boardstart = boardstart
        self.space_size = space_size
        self.is_clickable = is_clickable
        self.children = []
        self.color = color
        self.spaces = [None] * 9
        self.resolved = "not"
        self.parent = parent

        if has_children:
            for x in range(3):
                for y in range(3):
                    self.children.append(Board((self.boardstart[0] + x*self.space_size + 7, self.boardstart[1] + y*self.space_size+7), int(space_size/3)-5, "red", parent=self))
    
    def draw(self):
        for x in range(3):
            for y in range(3):
                pg.draw.rect(screen, self.color, (self.boardstart[0] + x*self.space_size, self.boardstart[1] + y*self.space_size, self.space_size, self.space_size), width=1)
                if self.spaces[3*x+y] == "X":
                    pg.draw.line(screen, "purple", (self.boardstart[0] + x*self.space_size + 5, self.boardstart[1] + y*self.space_size + 5), 
                                 (self.boardstart[0] + x*self.space_size - 5 + self.space_size, self.boardstart[1] + y*self.space_size - 5 + self.space_size))
                    pg.draw.line(screen, "purple", (self.boardstart[0] + x*self.space_size + 5, self.boardstart[1] + y*self.space_size - 5 + self.space_size), 
                                 (self.boardstart[0] + x*self.space_size - 5 + self.space_size, self.boardstart[1] + y*self.space_size + 5))

                elif self.spaces[3*x+y] == "O":
                    pg.draw.circle(screen, "purple", 
                                   ((self.boardstart[0] + x*self.space_size - 5 + self.space_size - (self.boardstart[0] + x*self.space_size + 5))/2
                                    + self.boardstart[0] + x*self.space_size + 5, 
                                    (self.boardstart[1] + y*self.space_size - 5 + self.space_size - (self.boardstart[1] + y*self.space_size + 5))/2 
                                    + self.boardstart[1] + y*self.space_size + 5), 
                                    (self.boardstart[0] + x*self.space_size - 5 + self.space_size - (self.boardstart[0] + x*self.space_size + 5))/2 - 2, width=1)


        for child in self.children:
            child.draw()
    
    def check_if_resolved(self):
        #check for vertical win
        for x in range(3):
            piece = ""
            for y in range(3):
                if piece == "":
                    if self.spaces[3*x+y] != None:
                        piece = self.spaces[3*x+y]
                    else: break
                elif piece != self.spaces[3*x+y]: break
            else:
                self.resolved = piece
                break
        
        #check for horizontal win
        for y in range(3):
            piece = ""
            for x in range(3):
                if piece == "":
                    if self.spaces[3*x+y] != None:
                        piece = self.spaces[3*x+y]
                    else: break
                elif piece != self.spaces[3*x+y]:
                    break
            else:
                self.resolved = piece
                break
        
        #check for up-left to down-right win
        for piece in ["X", "O"]:
            for i in range(3):
                if self.spaces[i*4] != piece:
                    break
            else:
                self.resolved = piece
                break

        #check for down-left to up-right win
        for piece in ["X", "O"]:
            for i in range(3):
                if self.spaces[i*3+(2-i)] != piece:
                    break
            else:
                self.resolved = piece
                break
        
        if self.resolved != "not": 
            if self.parent != None:
                self.parent.child_resolved(self)
            else:
                print(self.resolved, "won")
                global running
                running = False
                return self.resolved

    def child_resolved(self, _child):
        for i in range(len(self.children)):
            if self.children[i] == _child:
                self.spaces[i] = _child.resolved

        self.check_if_resolved()

    def check_if_clicked(self, mouse_pos):
        global last_space_index
        if self.is_clickable:
            if self.resolved == "not":
                for x in range(3):
                    if mouse_pos[0] > self.boardstart[0] + x*self.space_size and mouse_pos[0] < self.boardstart[0] + x*self.space_size + self.space_size:
                        for y in range(3):
                            if mouse_pos[1] > self.boardstart[1] + y*self.space_size and mouse_pos[1] < self.boardstart[1] + y*self.space_size + self.space_size:
                                self.place_piece_on_board(3*x+y)
                                

        if self.children:
            for i, child in enumerate(self.children):
                if i == last_space_index or last_space_index is None or self.children[last_space_index].resolved != "not":
                    child.check_if_clicked(mouse_pos)
    
    def place_piece_on_board(self, i):
        if self.children != []:
            return self.children[i//9].place_piece_on_board(i%9)
            
        else:
            global turn
            if self.spaces[i] is None:
                self.spaces[i] = turn
                self.check_if_resolved()
                turn = "O" if turn == "X" else "X"
                last_space_index = i
                return True
            
            return False

    def get_board_stat(self):
        global turn
        spaces = []
        if self.children != []:
            for child in self.children:
                spaces.extend(child.get_board_stat())
            
            return spaces

        else:
            for space in self.spaces:
                if space == turn: spaces.append(1)
                elif space != turn: spaces.append(-1)
                else: spaces.append(0)
            return spaces



board = Board(boardstart, _space_size, "black", has_children=True, is_clickable=False)

if __name__ == "__main__":

    print("hej")
    while running:
        try:
            # poll for events
            # pygame.QUIT event means the user clicked X to close your window
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

            # fill the screen with a color to wipe away anything from last frame
            screen.fill("white")

            #pg.draw.circle(screen, "red", player_pos, r)

            #pg.draw.rect(screen, "red", (10,10,200,200), width=1)

            board.draw()
            if turn == "X":
                #draw X
                pg.draw.line(screen, "purple", (30, 30), (100, 100))
                pg.draw.line(screen, "purple", (30, 100), (100, 30))
                
            else: 
                #draw O
                pg.draw.circle(screen, "purple", (65, 65), 35, width=1)

            if mouse.get_pressed()[0]: 
                if not mouseDown:  
                    mouseDown = True
                    mouse_pos = mouse.get_pos()
                    board.check_if_clicked(mouse_pos)
            else: 
                mouseDown = False

            # flip() the display to put your work on screen
            pg.display.flip()

            # limits FPS to 60
            # dt is delta time in seconds since last frame, used for framerate-
            # independent physics.
            dt = clock.tick(60) / 1000
        except KeyboardInterrupt:
            break


    pg.quit()