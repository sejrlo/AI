import pygame as pg
from pygame import mouse


# pygame setup
pg.init()
screen = pg.display.set_mode((1280, 720))
clock = pg.time.Clock()
running = True
dt = 0

player_pos = pg.Vector2(screen.get_width() / 2, screen.get_height() / 2)
r = 40
mouseDown = False

boardstart = (310,32)
space_size = 220

class Board():
    def __init__(self, boardstart, space_size, is_clickable = True, has_children = False):
        self.boardstart = boardstart
        self.space_size = space_size
        self.is_clickable = is_clickable
        self.children = []
        if has_children:
            for x in range(self.boardstart[0], self.boardstart[0] + 3*self.space_size, space_size):
                for y in range(self.boardstart[1], self.boardstart[1] + 3*self.space_size, space_size):
                    self.children.append(Board((x+5, y+5), int(self.space_size/3)-10))
    
    def draw(self):
        for x in range(self.boardstart[0], self.boardstart[0] + 3*self.space_size, space_size):
            for y in range(self.boardstart[1], self.boardstart[1] + 3*self.space_size, space_size):
                pg.draw.rect(screen, "black", (x, y, space_size, space_size), width=1)
        
        for child in self.children:
            child.draw()
    
board = Board(boardstart, space_size, has_children=True, is_clickable=False)


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

    
        

        
        # if mouse.get_pressed()[0]: 
        #     if not mouseDown:  
        #         mouseDown = True
        #         mouse_pos = mouse.get_pos()
        #         if ((player_pos[0] - mouse_pos[0])**2 + (player_pos[1] - mouse_pos[1])**2)** (1/2) <= r:
        #             print(player_pos)
        #             print(mouse.get_pos())
        # else: 
        #     mouseDown = False



        # flip() the display to put your work on screen
        pg.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000
    except KeyboardInterrupt:
        break



pg.quit()