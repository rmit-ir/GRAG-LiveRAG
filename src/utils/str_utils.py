def take_words(input_string: str, max_words: int) -> str:
    words = input_string.split(' ')[:max_words]
    return ' '.join(words)
