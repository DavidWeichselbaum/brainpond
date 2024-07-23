from collections import deque
from time import sleep

import numpy as np
import colorama
from colorama import Fore, Back


colorama.init(autoreset=True)
np.random.seed(42)


class BrainPond():

    max_int = np.int8(127)
    min_int = np.int8(-128)

    char_color_tuples = [
        ('@', Fore.GREEN),  # execution start (random direction)
        ('<', Fore.YELLOW),  # move current head left
        ('>', Fore.YELLOW),  # move current head right
        ('^', Fore.YELLOW),  # move current head up
        ('v', Fore.YELLOW),  # move current head down
        ('+', Fore.MAGENTA),  # increment number at tape head by one
        ('-', Fore.MAGENTA),  # decrement number at tape head by one
        ('i', Fore.BLUE),  # select instruction head. movements change direction of instruction execution.
        ('t', Fore.BLUE),  # select tape head
        ('a', Fore.BLUE),  # select head a
        ('b', Fore.BLUE),  # select head b
        ('c', Fore.BLUE),  # select head c
        ('.', Fore.CYAN),  # copy from current to previous head in stack. copying to tape is free.
        (',', Fore.CYAN),  # copy from previous to current head in stack. copying to tape is free.
        ('[', Fore.RED),  # if the number at the current head is negative, jump to after the matching ]
        (']', Fore.RED),  # if the number at the current head is positive, jump to after the matching [
    ]
    num_to_char = {np.int8(num): char for num, (char, color) in enumerate(char_color_tuples)}
    char_to_num = {char: np.int8(num) for num, (char, color) in enumerate(char_color_tuples)}
    char_to_color = {char: color for num, (char, color) in enumerate(char_color_tuples)}

    head_to_color = {
        'i': Back.LIGHTYELLOW_EX,
        't': Back.LIGHTMAGENTA_EX,
        'a': Back.LIGHTRED_EX,
        'b': Back.LIGHTGREEN_EX,
        'c': Back.LIGHTBLUE_EX,
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
        self.grid = np.random.randint(-16, 16, size=(self.height, self.width), dtype=np.int8)

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
                        background_color = self.head_to_color[head]

                if char:
                    color = self.char_to_color[char]
                    string = f"{background_color}{color}   {char} "
                else:
                    string = f"{background_color}{num:5d}"
                print(string, end='')
            print()
        print()

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
        entrypoin_number = self.char_to_num['@']
        entrypoin_coordinate_tuple = np.where(self.grid == entrypoin_number)
        entrypoin_coordinates = np.array(list(zip(entrypoin_coordinate_tuple[0],
                                                  entrypoin_coordinate_tuple[1])), dtype=int)
        random_index = np.random.choice(len(entrypoin_coordinates))
        entrypoin_coord = entrypoin_coordinates[random_index]
        entrypoin_coord = tuple(entrypoin_coord)

        direction = np.random.choice(list('<>^v'))

        self.execute(entrypoin_coord, direction, steps, print_=print_)

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

        for step in range(steps):
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
                    self.grid[to_coord] = self.grid[from_coord]
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
    pond = BrainPond()

    seed = ['@avt[ab.a>b>]']
    # pond.seed(seed, (0, 0))

    for i in range(100):
        pond.seed(seed, (i, 0))

    # pond.execute((0, 0), '>', 300, print_=True)
    # for i in range(100000):
    #     coord = (i, 0)
    #     coord = pond._update_coordinates(coord)
    #     pond.execute(coord, '>', 200)
    #     if i % 10 == 0:
    #         print(i)
    #         pond.print(0, 0, 40, 40)

    for i in range(100000):
        pond.execute_random(300)
        if i % 100 == 0:
            print(i)
            pond.print(0, 0, 40, 40)
