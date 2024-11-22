import chess
import chess.svg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM


def fen_to_image(fen):
    board = chess.Board(fen)
    current_board = chess.svg.board(board=board)

    output_file = open("current_board.svg", "w")
    output_file.write(current_board)
    output_file.close()

    svg = svg2rlg("current_board.svg")
    renderPM.drawToFile(svg, "current_board.png", fmt="PNG")
    return board