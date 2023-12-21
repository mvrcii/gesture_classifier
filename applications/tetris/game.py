from copy import deepcopy
from random import choice, randrange

import pygame
import pygame.mixer

from Prediction_Mode.live_video_feed import LiveLoopInference


class Tetris:
    def __init__(self, bpm):
        self.live_video_loop = LiveLoopInference(inference_mode=1)
        self.deltaMovement = 0
        self.use_gestures = True
        self.game_paused = False

        # tetris Board Variables
        self.screen_flipped = False
        self.width, self.height = 10, 20
        self.tile_size = 45
        self.game_resolution = self.width * self.tile_size, self.height * self.tile_size
        self.resolution = 1920, 1080
        self.FPS = 60

        pygame.init()
        self.window = pygame.display.set_mode(self.resolution)
        self.surface = pygame.Surface(self.game_resolution)
        self.clock = pygame.time.Clock()

        # Center the game surface
        self.surface_rect = self.surface.get_rect()
        window_rect = self.window.get_rect()
        self.surface_rect.center = window_rect.center

        # Board and tile Definitions
        self.grid = [pygame.Rect(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size) for x in
                     range(self.width) for y in
                     range(self.height)]
        self.tiles_pos = [[(-1, 0), (-2, 0), (0, 0), (1, 0)],
                          [(0, -1), (-1, -1), (-1, 0), (0, 0)],
                          [(-1, 0), (-1, 1), (0, 0), (0, -1)],
                          [(0, 0), (-1, 0), (0, 1), (-1, -1)],
                          [(0, 0), (0, -1), (0, 1), (-1, -1)],
                          [(0, 0), (0, -1), (0, 1), (1, -1)],
                          [(0, 0), (0, -1), (0, 1), (-1, 0)]]
        self.tiles = [[pygame.Rect(x + self.width // 2, y + 1, 1, 1) for x, y in fig_pos] for fig_pos in self.tiles_pos]
        self.tile_rect = pygame.Rect(0, 0, self.tile_size - 2, self.tile_size - 2)
        self.field = [[0 for i in range(self.width)] for j in range(self.height)]

        # Add music to the game
        pygame.mixer.init()
        pygame.mixer.music.load('Tetris_160bpm.wav')
        pygame.mixer.music.set_volume(0.01)
        pygame.mixer.music.play(-1)  # Play the music in a loop

        # Set the BPM and calculate the time interval for the tile movement
        self.bpm = bpm
        self.beat_interval = self.reset_beat_interval()  # Convert BPM to milliseconds

        # Creation of Surface
        self.bg = self.window.convert()
        self.game_bg = self.surface.convert()

        # Font & Text Definition
        self.game_font = pygame.font.SysFont('heightelvetica', 45)

        self.title_score = self.game_font.render('Score:', True, pygame.Color('green'))
        self.title_pause = self.game_font.render('Pause', True, pygame.Color('red'))
        self.title_play = self.game_font.render('Play', True, pygame.Color('green'))
        self.title = self.title_play

        # tile Choice Selector
        self.tile, self.next_tile = deepcopy(choice(self.tiles)), deepcopy(choice(self.tiles))
        self.tile_color, self.next_tile_color = self.get_tile_color(), self.get_tile_color()

        # Score Definition
        self.score, self.lines = 0, 0
        self.scores = {0: 0, 1: 100, 2: 300, 3: 700, 4: 1500}

        self.last_command_tick = 0
        self.last_movement_y_tick = 375 * 2
        self.game_loop()

    def game_loop(self):

        while True:
            self.deltaMovement = 0
            self.clearWindow()

            self.process_input()

            self.update()

            self.render()

            self.game_over()
            pygame.display.flip()
            self.clock.tick(self.FPS)

    def process_input(self):
        # delay for full lines
        for i in range(self.lines):
            pygame.time.wait(200)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and \
                    (event.key == pygame.K_q or event.key == pygame.K_ESCAPE):
                pygame.quit()
                self.live_video_loop.running = False
                raise SystemExit

        command = None
        if (pygame.time.get_ticks() - self.last_command_tick) > 200:
            command = self.live_video_loop.get_command()
            self.last_command_tick = pygame.time.get_ticks()

        if command is not None:
            print(command)
            if command == 'swipe_left':  # moves Tile to the right
                self.deltaMovement = -1
            elif command == 'swipe_right':  # moves Tile to the left
                self.deltaMovement = 1
            elif command == 'rotate_right':  # rotates Tile to the right
                center = self.tile[0]
                tile_old = deepcopy(self.tile)
                for title_index in range(4):
                    x = self.tile[title_index].y - center.y
                    y = self.tile[title_index].x - center.x
                    self.tile[title_index].x = center.x - x
                    self.tile[title_index].y = center.y + y
                    if not self.is_tile_inside_boarders(title_index):
                        self.tile = deepcopy(tile_old)
                        break
            elif command == 'rotate_left':
                center = self.tile[0]
                tile_old = deepcopy(self.tile)
                for title_index in range(4):
                    x = self.tile[title_index].y - center.y
                    y = self.tile[title_index].x - center.x
                    self.tile[title_index].x = center.x + x
                    self.tile[title_index].y = center.y - y
                    if not self.is_tile_inside_boarders(title_index):
                        self.tile = deepcopy(tile_old)
                        break
            elif command == 'flip_table':
                self.game_paused = not self.game_paused
            elif command == 'swipe_down':
                self.reduce_beat_interval_by_factor(bar_interval=4)
            elif command == 'spin':
                print("triggered spin")
                self.screen_flipped = not self.screen_flipped

    def reduce_beat_interval_by_factor(self, bar_interval):
        self.beat_interval = (60 / self.bpm) * 2 * 1000 / bar_interval

    def reset_beat_interval(self):
        return (60 / self.bpm) * 2 * 1000

    def update(self):
        self.title = self.title_pause if self.game_paused else self.title_play

        if self.game_paused:
            return

        # ======== MOVEMENT ========
        # === Move X
        tile_old = deepcopy(self.tile)
        for i in range(4):
            # TODO: Check why game breaks if moving Tile outwards the right border
            self.tile[i].x += self.deltaMovement
            if not self.is_tile_inside_boarders(i):
                self.tile = deepcopy(tile_old)
                break

        # === Move Y
        if (pygame.time.get_ticks() - self.last_movement_y_tick) > self.beat_interval:
            self.last_movement_y_tick = pygame.time.get_ticks()
            tile_old = deepcopy(self.tile)
            for i in range(4):
                self.tile[i].y += 1
                if not self.is_tile_inside_boarders(i):
                    for i in range(4):
                        self.field[tile_old[i].y][tile_old[i].x] = self.tile_color
                    self.tile, self.tile_color = self.next_tile, self.next_tile_color
                    self.next_tile, self.next_tile_color = deepcopy(choice(self.tiles)), self.get_tile_color()
                    self.beat_interval = self.reset_beat_interval()
                    break

        # ======== COLLISION CHECK ========
        line, self.lines = self.height - 1, 0
        for row in range(self.height - 1, -1, -1):
            count = 0
            for i in range(self.width):
                if self.field[row][i]:
                    count += 1
                self.field[line][i] = self.field[row][i]
            if count < self.width:
                line -= 1
            else:
                # If row is cleared, falldown speed is increased
                self.lines += 1
        # compute score
        self.score += self.scores[self.lines]

    def render(self):
        self.draw()
        if self.screen_flipped:
            self.flip_screen()

    def flip_screen(self):
        flipped_surface = pygame.transform.rotate(self.window, 180)
        self.window.blit(flipped_surface, (0, 0))

    # Random Range tile_color Selector
    def get_tile_color(self):
        tile_color = [randrange(30, 256), randrange(30, 256), randrange(30, 256)]
        return tile_color

    # Draw Functions
    def draw_grid(self):
        [pygame.draw.rect(self.surface, (40, 40, 40), i_rect, 1) for i_rect in self.grid]

    def draw_tile(self):
        for i in range(4):
            self.tile_rect.x = self.tile[i].x * self.tile_size
            self.tile_rect.y = self.tile[i].y * self.tile_size
            pygame.draw.rect(self.surface, self.tile_color, self.tile_rect)

    def draw_field(self):
        for y, raw in enumerate(self.field):
            for x, col in enumerate(raw):
                if col:
                    self.tile_rect.x, self.tile_rect.y = x * self.tile_size, y * self.tile_size
                    pygame.draw.rect(self.surface, col, self.tile_rect)

    def draw_next_tile(self):
        x_offset = self.surface_rect.x + self.game_bg.get_rect().width - 85
        for i in range(4):
            self.tile_rect.x = self.next_tile[i].x * self.tile_size + x_offset
            self.tile_rect.y = self.next_tile[i].y * self.tile_size + 185
            pygame.draw.rect(self.window, self.next_tile_color, self.tile_rect)

    def draw_tiles(self):
        x_offset = self.surface_rect.x + self.game_bg.get_rect().width + 50
        self.window.blit(self.title_score, (x_offset, 780))
        self.window.blit(self.title, (x_offset, 540))
        self.window.blit(self.game_font.render(str(self.score), True, pygame.Color('white')), (x_offset, 840))

    # Check that Tiles do not extend boarder
    def is_tile_inside_boarders(self, tile_index):
        if self.tile[tile_index].x < 0 or self.tile[tile_index].x > self.width - 1:
            return False
        elif self.tile[tile_index].y > self.height - 1 or self.field[self.tile[tile_index].y][self.tile[tile_index].x]:
            return False
        return True

    def draw(self):
        self.draw_grid()
        self.draw_tile()
        self.draw_field()
        self.draw_next_tile()
        self.draw_tiles()

    def game_over(self):
        # global width, field, falldown_count, falldown_speed, falldown_max, score
        for rows in range(self.width):
            if self.field[0][rows]:
                self.field = [[0 for i in range(self.width)] for i in range(self.height)]
                # self.falldown_count, self.falldown_speed, self.falldown_max = 0, 20, 2000
                self.score = 0

    def clearWindow(self):
        self.window.blit(self.bg, (0, 0))
        self.window.blit(self.surface, self.surface_rect)
        self.surface.blit(self.game_bg, (0, 0))


if __name__ == "__main__":
    game_bpm = 160  # Set the BPM of the music
    Tetris(game_bpm)
