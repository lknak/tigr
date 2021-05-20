import multiprocessing
import os, sys
import numpy as np
import runner
import time


def main(folder_path: str, gpus: str, parallel_runs=3, num_reruns=3):
    gpus = gpus.strip().split(',' if ',' in gpus else ';')
    parallel_runs = int(parallel_runs)
    num_reruns = int(num_reruns)

    assert parallel_runs % len(gpus) == 0, f'Num parallel runs {parallel_runs} must be multiple of num gpus {len(gpus)}'

    if os.path.isfile(os.path.join(folder_path, 'schedule')):
        d = {}
        with open(os.path.join(folder_path, 'schedule'), 'r') as f:
            for l in f.read().splitlines():
                if l.split(':')[0] in d.keys():
                    d[l.split(':')[0]].append(l.split(':')[1])
                else:
                    d[l.split(':')[0]] = [l.split(':')[1]]
        configs = {int(k): np.array(v) for k, v in d.items() if k.isnumeric()}
    else:
        configs = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if '.json' in f])

        print(f'Found {len(configs)} configs:')
        for c in configs: print(f'\t{c}')
        print('')

        # Order num_reruns times configs for processing
        configs = np.array(configs * num_reruns)
        configs = np.array_split(configs, parallel_runs)
        configs = {process_nr: c_list for process_nr, c_list in enumerate(configs)}

        with open(os.path.join(folder_path, 'schedule'), 'w') as f:
            for process_nr, c_list in configs.items():
                for c in c_list:
                    f.write(f'{process_nr}:{c}\n')

    process_list = []

    for process_nr, c_list in configs.items():
        p = multiprocessing.Process(target=run_process, args=(c_list, gpus[process_nr % len(gpus)], process_nr))
        p.daemon = True
        p.start()
        process_list.append(p)
        # Wait 5s before next process is started so there is no interference
        time.sleep(5)

    for p in process_list:
        p.join()

    print('Finished')


def run_process(c_list: list, gpu_nr: str, process_nr=0):
    print(f'[{process_nr}: {time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())}]\tStarting process nr. {process_nr} with {len(c_list)} total runs')
    t_list = []
    t = time.time()

    for i, conf in enumerate(c_list):
        # Mark processing in schedule
        write_file(os.path.join(*os.path.split(conf)[:-1]), f'{process_nr}', conf, 'p')

        with HiddenPrints():
            runner.main(config=conf, name='', ti_option='', gpu=gpu_nr, use_mp=None, num_workers=None)

        t_list.append(time.time() - t)
        t = time.time()
        eta = np.array(t_list).mean() * (len(c_list) - (i + 1)) + t
        eta = ')' if (len(c_list) - (i + 1)) == 0 else f' -> ETA: {time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(eta))})'

        print(f'[{process_nr}: {time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())}] Finished run {i + 1}/{len(c_list)} [{(i + 1)/len(c_list)*100:.2f}%] '
              f'({pretty_time(t_list[-1])}{eta}')

        # Mark finished in schedule
        write_file(os.path.join(*os.path.split(conf)[:-1]), f'{process_nr}p', conf, 'f')


def write_file(path, queryf, queryb, mark):
    done = False
    while not done:
        if not os.path.isfile(os.path.join(path, 'temp')):
            # Create temp file
            with open(os.path.join(path, 'temp'), 'w'): pass

            # Write new schedule
            with open(os.path.join(path, 'schedule'), 'r') as f:
                t = f.read().splitlines()
            with open(os.path.join(path, 'schedule'), 'w') as f:
                f.truncate()
                found = False
                for i, c in enumerate(t):
                    if c.split(':')[0] == str(queryf) and c.split(':')[1] == str(queryb) and not found:
                        f.write(f'{queryf}{mark}:{queryb}\n')
                        found = True
                    else:
                        f.write(c + '\n')

            # Remove temp file
            os.remove(os.path.join(path, 'temp'))

            done = True
        else:
            time.sleep(1)

def pretty_time(secs):
    ints = [86400, 3600, 60, 1]
    # Find largest non-zero element
    start_ = [i for i in range(4) if secs > ints[i]][0]
    divs = [int(((secs % ints[i - 1]) if i > 0 else secs) / ints[i]) for i in range(4)]
    divs = [f'{a}{b}' for a, b in zip(divs, 'dhms')]
    return ' '.join(divs[start_:])


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        self._original_stderr = sys.stderr
        sys.stderr = sys.stdout

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


if __name__ == '__main__':
    main(*sys.argv[1:])