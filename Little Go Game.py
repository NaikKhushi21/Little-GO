import numpy as np
from skimage.measure import label
import numpy.typing as npt
from typing import Iterator, Union, List, Tuple, Optional, Set
from scipy import ndimage

class MyPlayer:
    winning_utility = 1000

    def __init__(self) -> None:
        self.piece: int
        self.opponent_piece: int

        self.board: MyGO
        self.search_depth = 4

    def ReadInput(self) -> None:
        with open('input.txt', "r") as f:
            lines = f.read().splitlines()

        self.piece = self.move_no = int(lines[0])
        self.opponent_piece = 3 - self.piece

        prev_board = [list(map(int, line)) for line in lines[1:6]]
        board = [list(map(int, line)) for line in lines[6:]]

        prev_board = np.array(prev_board)
        board = np.array(board)

        if board.sum() > 1:
            self.ReadMove()

        self.board = MyGO(self.piece, self.move_no, prev_board, board)

    def WriteNextOutput(self, move: Optional[Tuple[int, int]]) -> None:
        self.WriteMove()

        with open('output.txt', "w") as f:
            if move:
                f.write(f"{move[0]},{move[1]}")
            else:
                f.write("PASS")

    def ReadMove(self) -> None:
        with open('moves.txt', "r") as f:
            m = int(f.read())
            if m < 25:
                self.move_no = m

    def WriteMove(self) -> None:
        with open('moves.txt', "w") as f:
            f.write(f"{self.board.moves + 2}")

    def calculate_move(self) -> Optional[Tuple[int, int]]:
        if self.board.moves <= 4:
            initial_moves = [(2, 2), (2, 3), (3, 2), (2, 1), (1, 2)]
            for i, j in initial_moves:
                if self.board.valid_place_check(i, j):
                    return (i, j)

        return self.alpha_beta(4)

    def evaluate_board(self) -> Union[int, float]:
        evaluation_score = 0
        piece_multiplier = 1 if self.piece == 1 else -1

        black_counts, white_counts = self.board.total_pieces()
        evaluation_score += (black_counts - white_counts) * piece_multiplier * 10

        evaluation_score += self.board.cluster_liberty(self.piece) * 6
        evaluation_score -= self.board.cluster_liberty(self.opponent_piece) * 6

        evaluation_score -= self.board.find_cluster(self.piece) * 4
        evaluation_score += self.board.find_cluster(self.opponent_piece) * 4

        for i, j in np.argwhere(self.board.board == self.piece):
            if i == 0 or i == 4:
                evaluation_score -= 2
            if j == 0 or j == 4:
                evaluation_score -= 2
        return evaluation_score

    def alpha_beta(self, depth: int) -> Optional[Tuple[int, int]]:
        best_move, _ = self.alpha_beta_recursive(
            depth, float("-inf"), float("inf"), maximizing=True
        )
        return best_move

    def alpha_beta_recursive(
        self, depth: int, alpha: float, beta: float, maximizing: bool
    ) -> Tuple[Optional[Tuple[int, int]], Union[float, int]]:
        if self.board.game_end() or depth == 0:
            return None, self.evaluate_board()

        best_move: Optional[Tuple[int, int]] = None
        best_value = float("-inf") if maximizing else float("inf")

        for move in self.board.find_valid_moves():
            if move:
                self.board.place_chess(move[0], move[1])
            else:
                self.board.pass_chess()

            _, value = self.alpha_beta_recursive(
                depth - 1, alpha, beta, not maximizing
            )

            self.board.unplace_chess()

            if maximizing:
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, value)
                if value >= beta:
                    return best_move, best_value
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, value)
                if value <= alpha:
                    return best_move, best_value

        return best_move, best_value

class MyGO:
    def __init__(
        self,
        piece: int,
        moves: int,
        prev_board: npt.NDArray[np.int_],
        board: npt.NDArray[np.int_],
    ) -> None:
        self.size: int = 5
        self.komi: float = 2.5
        self.max_moves: int = 25

        self.piece: int = piece
        self.opponent_piece: int = 3 - self.piece
        self.moves: int = moves

        self.prev_board: npt.NDArray[np.int_] = prev_board
        self.board: npt.NDArray[np.int_] = board

        self.each_pos = [(i, j) for i in range(5) for j in range(5)]
        self.mid_pos = [(i, j) for i in range(1, 4) for j in range(1, 4)]

        self.neigh_pos = {
            (i, j): [pos for pos in self.detect_neighbours(i, j)]
            for i, j in self.each_pos
        }

        self.neigh_pos_dia = {
            (i, j): [pos for pos in self.detect_neighbour_diag(i, j)]
            for i, j in self.each_pos
        }

        self.move_hist: List[Optional[Tuple[int, int]]] = []
        self.board_history: List[npt.NDArray[np.int_]] = []

        self.allies: npt.NDArray[np.int_] = label(self.board, connectivity=1)
        self.ally_hist: List[npt.NDArray[np.int_]] = []

    def place_chess(self, i: int, j: int) -> None:
        self.board_history.append(self.prev_board)
        self.prev_board = np.copy(self.board)
        self.ally_hist.append(self.allies)
        self.move_hist.append((i, j))
        self.board[i, j] = self.piece
        self.remove_died_pieces()
        self.detect_allies()
        self.moves += 1
        self.piece, self.opponent_piece = self.opponent_piece, self.piece

    def pass_chess(
        self,
    ) -> None:
        self.move_hist.append(None)
        self.moves += 1
        self.piece, self.opponent_piece = self.opponent_piece, self.piece

    def unplace_chess(self) -> None:
        if self.move_hist.pop():
            self.board = self.prev_board
            self.prev_board = self.board_history.pop()
            self.allies = self.ally_hist.pop()
        self.moves -= 1
        self.piece, self.opponent_piece = self.opponent_piece, self.piece

    def detect_allies(self) -> None:
        self.allies = label(self.board, connectivity=1)

    def compare_board(self) -> bool:
        return np.array_equal(self.prev_board, self.board)

    def valid_place_check(self, i: int, j: int) -> bool:
        if self.board[i, j]:
            return False

        temp_allies = self.allies
        temp_board = np.copy(self.board)
        temp_board[i, j] = self.piece
        self.allies = label(temp_board, connectivity=1)
        
        if self.find_liberty(i, j):
            self.board[i, j] = 0
            self.allies = temp_allies
            return True

        removed_pieces = self.remove_died_pieces()
        self.detect_allies()
        
        if not self.find_liberty(i, j) or (removed_pieces and self.compare_board()):
            for x, y in removed_pieces:
                self.board[x, y] = self.opponent_piece
            self.board[i, j] = 0
            self.allies = temp_allies
            return False

        for x, y in removed_pieces:
            self.board[x, y] = self.opponent_piece
        self.board[i, j] = 0
        self.allies = temp_allies
        return True
    
    def detect_if_captured(self, i: int, j: int) -> bool:
        self.board[i, j] = self.piece
        is_captured = self.find_died_pieces()
        self.board[i, j] = 0
        return is_captured

    def find_liberty(self, i: int, j: int) -> bool:
        allies_positions = np.argwhere(self.allies == self.allies[i, j])
        for position in allies_positions:
            neighbors = self.neigh_pos[tuple(position)]
            for neighbor in neighbors:
                if not self.board[tuple(neighbor)]:
                    return True
        return False

    def game_end(self) -> bool:
        if self.moves >= self.max_moves:
            return True
        elif self.move_hist and not self.move_hist[-1] and self.compare_board():
            return True
        else:
            return False

    def find_died_pieces(self) -> bool:
        opponent_positions = np.argwhere(self.board == self.opponent_piece)
        for position in opponent_positions:
            if not self.find_liberty(*position):
                return True
        return False

    def remove_died_pieces(self) -> Set[Tuple[int, int]]:
        dead_pieces = set()

        opponent_positions = np.argwhere(self.board == self.opponent_piece)
        for position in opponent_positions:
            position_tuple = tuple(position)
            if position_tuple not in dead_pieces and not self.find_liberty(*position_tuple):
                allies_positions = np.argwhere(self.allies == self.allies[position_tuple])
                dead_pieces.update(set(tuple(pos) for pos in allies_positions))

        for position in dead_pieces:
            self.board[position] = 0

        return dead_pieces

    def find_valid_moves(self) -> Iterator[Optional[Tuple[int, int]]]:
        discovered = set()
        last_move = self.final_move()
        valid_moves = []

        if self.moves <= 8:
            for move in self.mid_pos:
                if self.valid_place_check(*move):
                    discovered.add(move)
                    yield move
            if self.moves <= 6:
                return

        if last_move:
            for move in self.neigh_pos[last_move] + self.neigh_pos_dia[last_move]:
                if self.valid_place_check(*move):
                    if self.detect_if_captured(*move):
                        discovered.add(move)
                        yield move
                    else:
                        valid_moves.append(move)

        for move in self.each_pos:
            if move not in discovered and self.valid_place_check(*move):
                if self.detect_if_captured(*move):
                    yield move
                else:
                    valid_moves.append(move)

        yield from valid_moves

        if self.moves > 15:
            yield None

    def final_move(self) -> Optional[Tuple[int, int]]:
        if len(self.move_hist) > 0:
            return self.move_hist[-1]

        diff_positions = np.argwhere(self.board != self.prev_board)
        for i, j in diff_positions:
            if self.board[i, j] == self.opponent_piece:
                return i, j

        return None

    def find_cluster(self, piece: int) -> int:
        cluster_count = 0
        visited = set()

        def dfs(i, j):
            nonlocal visited
            stack = [(i, j)]
            cluster = set()

            while stack:
                x, y = stack.pop()
                cluster.add((x, y))
                visited.add((x, y))

                for a, b in self.neigh_pos[(x, y)]:
                    if (a, b) not in visited and self.board[a, b] == piece:
                        stack.append((a, b))

            return cluster

        for i, j in np.argwhere(self.board == piece):
            if (i, j) not in visited:
                cluster = dfs(i, j)
                cluster_count += 1

        return cluster_count

    def cluster_liberty(self, piece: int) -> int:
        liberties = 0
        visited = set()

        def dfs(i, j):
            nonlocal visited
            stack = [(i, j)]
            cluster = set()
            liberties = 0

            while stack:
                x, y = stack.pop()
                cluster.add((x, y))
                visited.add((x, y))

                for a, b in self.neigh_pos[(x, y)]:
                    if (a, b) not in visited and self.board[a, b] == piece:
                        stack.append((a, b))

                for x, y in self.neigh_pos[(i, j)]:
                    if not self.board[x, y]:
                        liberties += 1

            return cluster

        for i, j in np.argwhere(self.board == piece):
            if (i, j) not in visited:
                cluster = dfs(i, j)
                liberties += len(cluster)

        return liberties

    def total_pieces(self) -> Tuple[int, int]:
        black_count = (self.board == 1).sum()
        white_count = (self.board == 2).sum()
        return black_count, white_count

    def winner(self) -> int:
        black_count, white_count = self.total_pieces()
        diff = black_count - white_count - self.komi

        if diff > 0:
            return 1
        elif diff < 0:
            return 2
        else:
            return 0

    def detect_neighbours(self, i: int, j: int) -> List[Tuple[int, int]]:
        size = self.size
        positions = [(x, y) for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] if 0 <= x < size and 0 <= y < size]
        return positions
    
    def detect_neighbour_diag(self, i: int, j: int) -> List[Tuple[int, int]]:
        size = self.size
        positions = [(x, y) for x, y in [(i - 1, j - 1), (i + 1, j + 1), (i + 1, j - 1), (i - 1, j + 1)] if 0 <= x < size and 0 <= y < size]
        return positions

if __name__ == "__main__":
    player = MyPlayer()
    player.ReadInput()
    move = player.calculate_move()
    player.WriteNextOutput(move)