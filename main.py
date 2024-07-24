from collections import deque
from time import sleep

import numpy as np
import colorama
from colorama import Fore, Back
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


colorama.init(autoreset=True)
np.random.seed(42)

color_to_foreground = {
    'red': Fore.RED,
    'green': Fore.GREEN,
    'blue': Fore.BLUE,
    'yellow': Fore.YELLOW,
    'magenta': Fore.MAGENTA,
    'cyan': Fore.CYAN,
    }
color_to_background = {
    'red': Back.LIGHTRED_EX,
    'green': Back.LIGHTGREEN_EX,
    'blue': Back.LIGHTBLUE_EX,
    'yellow': Back.LIGHTYELLOW_EX,
    'magenta': Back.LIGHTMAGENTA_EX,
    'cyan': Back.LIGHTCYAN_EX,
}


class BrainPond():

    min_int = -128
    max_int = 127

    min_int_seed = -16
    max_int_seed = 16

    copy_cost = 256  # steps

    char_color_tuples = [
        ('@', 'green'),  # execution start (random direction)
        ('<', 'yellow'),  # move current head left
        ('>', 'yellow'),  # move current head right
        ('^', 'yellow'),  # move current head up
        ('v', 'yellow'),  # move current head down
        ('+', 'magenta'),  # increment number at tape head by one
        ('-', 'magenta'),  # decrement number at tape head by one
        ('i', 'blue'),  # select instruction head. movements change direction of instruction execution.
        ('t', 'blue'),  # select tape head
        ('a', 'blue'),  # select head a
        ('b', 'blue'),  # select head b
        ('c', 'blue'),  # select head c
        ('.', 'cyan'),  # copy from current to previous head in stack. copying to tape is free.
        (',', 'cyan'),  # copy from previous to current head in stack. copying to tape is free.
        ('[', 'red'),  # if the number at the current head is negative, jump to after the matching ]
        (']', 'red'),  # if the number at the current head is positive, jump to after the matching [
    ]
    num_to_char = {np.int8(num): char for num, (char, color) in enumerate(char_color_tuples, 1)}
    char_to_num = {char: np.int8(num) for num, (char, color) in enumerate(char_color_tuples, 1)}
    char_to_color = {char: color for num, (char, color) in enumerate(char_color_tuples, 1)}
    num_to_color = {np.int8(num): color for num, (char, color) in enumerate(char_color_tuples, 1)}

    head_to_color = {
        'i': 'yellow',
        't': 'magenta',
        'a': 'red',
        'b': 'green',
        'c': 'blue',
    }

    inv_direction = {
        '<': '>',
        '>': '<',
        '^': 'v',
        'v': '^',
    }
    inv_bracket = {
        '[': ']',
        ']': '[',
    }

    def __init__(self, width=1024, height=1024, tape_width=32, tape_height=32):
        self.height = height
        self.width = width
        self.tape_width = tape_width
        self.tape_height = tape_height
        # self.grid = np.random.randint(self.min_int_seed, self.max_int_seed,
        #                               size=(self.height, self.width), dtype=np.int8)
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)

        cmap_list = [self.num_to_color.get(i, 'white') for i in range(self.min_int, self.max_int + 1)]
        cmap = ListedColormap(cmap_list)
        self.fig, ax = plt.subplots()
        self.pixel_map = ax.imshow(self.grid, cmap=cmap, interpolation='nearest', vmin=self.min_int, vmax=self.max_int)
        self.fig.canvas.draw()
        plt.show(block=False)

    def print(self, x0, y0, x1, y1, head_coords={}):
        for i in range(x0, x1):
            for j in range(y0, y1):
                num = self.grid[i][j]
                char = self.num_to_char.get(num)

                background_color = ''
                for head, coord in head_coords.items():
                    if head == 't':
                        continue
                    if (i, j) == coord:
                        head_color = self.head_to_color[head]
                        background_color = color_to_background[head_color]

                if char:
                    char_color = self.char_to_color[char]
                    foreground_color = color_to_foreground[char_color]
                    string = f"{background_color}{foreground_color}   {char} "
                else:
                    string = f"{background_color}{num:5d}"
                print(string, end='')
            print()
        print()

    def show(self):
        self.pixel_map.set_data(self.grid)
        plt.pause(0.01)  # update plot

    def seed(self, seed, coord):
        x, y = coord
        for i, row in enumerate(seed):
            i_wrapped = (i + x) % self.height
            for j, char in enumerate(row):
                number = self.char_to_num.get(char)
                if number is None:
                    assert number <= self.max_int and number >= self.min_int
                    number = np.int8(number)

                j_wrapped = (j + y) % self.width
                self.grid[i_wrapped, j_wrapped] = number

    def execute_random(self, steps, print_=False):
        entrypoin_coordinates = self.get_entypoints()
        if len(entrypoin_coordinates) == 0:  # wait to indel mutate more entrypoints
            return
        random_index = np.random.choice(len(entrypoin_coordinates))
        entrypoin_coord = entrypoin_coordinates[random_index]
        entrypoin_coord = tuple(entrypoin_coord)

        # direction = np.random.choice(list('<>^v'))
        direction = '>'

        self.execute(entrypoin_coord, direction, steps, print_=print_)

    def get_entypoints(self):
        entrypoin_number = self.char_to_num['@']
        entrypoin_coordinate_tuple = np.where(self.grid == entrypoin_number)
        entrypoin_coordinates = np.array(list(zip(entrypoin_coordinate_tuple[0],
                                                  entrypoin_coordinate_tuple[1])), dtype=int)
        return entrypoin_coordinates

    def execute(self, start_coord, instruction_direction, steps, print_=False):
        coords = {
            'i': start_coord,
            't': (0, 0),  # tape always starts at 0,0
            'a': start_coord,
            'b': start_coord,
            'c': start_coord,
        }
        head_stack = deque('ti', maxlen=2)  # start with tape and instruction headers
        tape = np.zeros((self.tape_height, self.tape_width), dtype=np.int8)

        step = steps
        while step > 0:
            step -= 1
            if print_:
                print(step)
                self.print(0, 0, 40, 40, coords)
                sleep(0.1)

            instruction_coord = coords['i']
            num = self.grid[instruction_coord]
            char = self.num_to_char.get(num)
            current_head = head_stack[0]
            previous_head = head_stack[1]

            match char:
                case None | '@':  # NoOp
                    pass
                case '<' | '>' | '^' | 'v':
                    direction = char
                    if current_head == 'i':  # instruction head changes direction
                        instruction_direction = direction
                    else:  # all other heads change position
                        head_coord = coords[current_head]
                        head_coord = self._update_coordinates(head_coord, direction, current_head)
                        coords[current_head] = head_coord
                case '+' | '-':
                    tape_coord = coords['t']
                    if char == '+':
                        tape[tape_coord] += 1
                    else:
                        tape[tape_coord] -= 1
                    if print_:
                        print(tape)
                case 'i' | 't' | 'a' | 'b' | 'c':
                    head_stack.appendleft(char)
                case '.' | ',':
                    current_head_coord = coords[current_head]
                    previous_head_coord = coords[previous_head]
                    if char == '.':
                        from_coord = current_head_coord
                        to_coord = previous_head_coord
                    else:
                        from_coord = previous_head_coord
                        to_coord = current_head_coord
                    if self.grid[to_coord] != self.grid[from_coord]:
                        self.grid[to_coord] = self.grid[from_coord]
                        step -= self.copy_cost  # only add cost if something is actually copied
                case '[' | ']':
                    current_head_coord = coords[current_head]
                    if current_head == 't':
                        current_number = tape[current_head_coord]
                    else:
                        current_number = self.grid[current_head_coord]

                    if char == '[' and current_number < 0 or char == ']' and current_number > 0:
                        new_instruction_coord = self._get_matching_bracket(
                            instruction_coord, instruction_direction, char)
                        coords['i'] = new_instruction_coord
                case _:
                    raise Exception

            coords['i'] = self._update_coordinates(coords['i'], instruction_direction)

    def mutate(self, prob):
        num_replacements = int(prob * self.width * self.height)
        assert num_replacements > 0
        rows = np.random.randint(0, self.height, num_replacements)
        cols = np.random.randint(0, self.width, num_replacements)
        rand_numbers = np.random.randint(self.min_int_seed, self.max_int_seed, size=num_replacements, dtype=np.int8)
        self.grid[rows, cols] = rand_numbers

    def _update_coordinates(self, coord, direction=None, head='i'):
        x, y = coord
        match direction:
            case '<':
                y -= 1
            case '>':
                y += 1
            case '^':
                x -= 1
            case 'v':
                x += 1
        if head == 't':
            x = x % self.tape_height
            y = y % self.tape_width
        else:
            x = x % self.height
            y = y % self.width
        coord = (x, y)
        return coord

    def _get_matching_bracket(self, starting_coord, direction, bracket):
        if bracket == ']':
            direction = self.inv_direction[direction]
        opposite_bracket = self.inv_bracket[bracket]

        bracket_counter = 1
        coord = starting_coord
        while True:
            coord = self._update_coordinates(coord, direction, 'i')
            if coord == starting_coord:  # prevent infinite loops
                return coord

            number = self.grid[coord]
            char = self.num_to_char.get(number)
            if char == bracket:
                bracket_counter += 1
            elif char == opposite_bracket:
                bracket_counter -= 1

            if bracket_counter == 0:
                return coord


if __name__ == '__main__':
    # pond = BrainPond(1024, 1024)
    pond = BrainPond(265, 265)

    seed = ['>i@avt[ab.a>b>]']
    # for i in range(pond.height):
    #     for j in range(0, pond.width, len(seed[0])+1):
    #         pond.seed(seed, (i, j))
    # for i in range(pond.height):
    #     pond.seed(seed, (i, 0))
    pond.seed(seed, (0, 0))
    # pond.print(0, 0, 40, 40)

    i = 0
    while True:
        i += 1
        pond.execute_random(512)
        if i % 1000 == 0:
            pond.mutate(0.001)
        if i % 1000 == 0:
            n_entries = len(pond.get_entypoints())
            print(f"Step: {i}, entrypoints: {n_entries}")
            pond.print(0, 0, 40, 40)
            pond.show()
