from tools import LETTERS

assert __name__ != '__main__', 'Module startup error.'


def placeholder(index: int) -> int: return index
def swapcase(index: int) -> int: return LETTERS.index(LETTERS[index].swapcase())
def caesar_cipher(index: int, key: int = 3) -> int: return (index + key) % 52


...  # any options
