from typing import TextIO, List


def log_train_progress(epoch: int, loss: float, accuracy: float, accuracy_tags: float, learning_rate: float, progress: int) -> None:
    print('\r' + 140 * ' ', end='')  # clear line
    d, r = progress // 5, progress % 5
    loading_bar = d*'█' + (('░' if r < 2 else '▒' if r < 4 else '▓') + max(0, 19 - d)*'░' if progress < 100 else '')
    print(f'\repoch: {epoch:3d} ║ train loss: {loss:1.6f} │ acc lemma: {accuracy:2.3f} % │ tag: {accuracy_tags:2.3f} % │ lr: {learning_rate:1.6f} ║ {loading_bar} {progress:2d} %', end='', flush=True)

def log_train(epoch: int, loss: float, accuracy: float, accuracy_tags: float, out_file: TextIO) -> None:
    print('\r' + 140 * ' ', end='')  # clear line
    print(f'\repoch: {epoch:3d} ║ train loss: {loss:1.6f} │ acc lemma: {accuracy:2.3f} % │ tag: {accuracy_tags:2.3f} % ║ ', end='', flush=True)
    print(f'epoch: {epoch:3d} ║ train loss: {loss:1.6f} │ acc lemma: {accuracy:2.3f} % │ tag: {accuracy_tags:2.3f} % ║ ', end='', flush=True, file=out_file)

def log_skipped_dev(out_file: TextIO) -> None:
    print(flush=True)
    print(flush=True, file=out_file)

def log_dev(accuracy: float, accuracy_tags: float, learning_rate: float, out_file: TextIO) -> None:
    print(f'dev acc lemma: {accuracy:2.3f} % │ tag: {accuracy_tags:2.3f} % ║ lr: {learning_rate:1.6f}', flush=True)
    print(f'dev acc lemma: {accuracy:2.3f} % │ tag: {accuracy_tags:2.3f} % ║ lr: {learning_rate:1.6f}', flush=True, file=out_file)

def log_mistakes(mistakes: List[str], out_file: TextIO) -> None:
    max_before_len = max(len(mistake[0]) for mistake in mistakes)
    max_after_len = max(len(mistake[4]) for mistake in mistakes)
    max_original_len = max(len(mistake[1]) for mistake in mistakes)
    max_lemma_len = max(len(mistake[2]) for mistake in mistakes)

    for mistake in mistakes:
        print(f'{mistake[0].rjust(max_before_len)}\t║\t{mistake[1].ljust(max_original_len)}\t║\t{mistake[2].ljust(max_lemma_len)}\t║\t{mistake[3].ljust(max_lemma_len)}\t║\t{mistake[4].ljust(max_after_len)}', file=out_file)
