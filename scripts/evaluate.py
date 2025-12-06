from concurrent.futures import ProcessPoolExecutor
import shutil
import time

from arena import run
from config import CONFIGS, NUM_WORKERS, Config

def run_session(config: Config):
    arguments = ["--num-episodes", str(config.num_episodes), "--seed", str(config.seed)]
    run.main(arguments=arguments, output_dir=config.output_directory)

if __name__ == "__main__":
    for config in CONFIGS:
        shutil.rmtree(config.output_directory, ignore_errors=True)
    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(run_session, CONFIGS)
    duration = time.perf_counter() - start_time
    print("Finished evaluation in", duration, "seconds")