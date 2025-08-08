#!/usr/bin/env python3
import os
import random
import time
import copy
import uuid
import json
import io
import argparse

from statistics import mean, pstdev
import matplotlib.pyplot as plt
from deap import base, creator, tools
from azure.storage.blob import BlobServiceClient


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-tasks",        type=int,   default=int(os.getenv("NUM_TASKS", 1000)))
    p.add_argument("--num-cores",        type=int,   default=int(os.getenv("NUM_CORES", 16)))
    p.add_argument("--num-population",   type=int,   default=int(os.getenv("NUM_POPULATION", 100)))
    p.add_argument("--num-generations",  type=int,   default=int(os.getenv("NUM_GENERATIONS", 500)))
    p.add_argument("--crossover-rate",   type=float, default=float(os.getenv("CROSSOVER_RATE", 0.8)))
    p.add_argument("--mutation-rate",    type=float, default=float(os.getenv("MUTATION_RATE", 0.2)))
    p.add_argument("--base-energy",      type=float, default=float(os.getenv("BASE_ENERGY", 0.01)))
    p.add_argument("--idle-energy",      type=float, default=float(os.getenv("IDLE_ENERGY", 0.002)))
    p.add_argument("--stagnation-limit", type=int,   default=int(os.getenv("STAGNATION_LIMIT", 20)))
    p.add_argument("--seed",             type=int,   default=int(os.getenv("SEED", 42)))
    return p.parse_args()


def setup_deap():
    try:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
    except RuntimeError:
        pass


def make_toolbox(args, exec_times):
    toolbox = base.Toolbox()
    toolbox.register("attr_core", random.randint, 0, args.num_cores - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_core, n=args.num_tasks)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=args.num_population)

    TOTAL_EXEC   = sum(exec_times)
    MEAN_LOAD    = TOTAL_EXEC / args.num_cores if args.num_cores > 0 else 1e-9
    MAX_MAKESPAN = TOTAL_EXEC
    MAX_ENERGY   = TOTAL_EXEC * args.base_energy + (args.num_cores - 1) * TOTAL_EXEC * args.idle_energy

    W_MAKESPAN  = 0.4
    W_ENERGY    = 0.2
    W_IMBALANCE = 0.4

    def evaluate(individual):
        core_times = [0.0] * args.num_cores
        for idx, c in enumerate(individual):
            core_times[c] += exec_times[idx]

        makespan = max(core_times)
        active_e = sum(ct * args.base_energy for ct in core_times)
        idle_e   = sum((makespan - ct) * args.idle_energy for ct in core_times)
        total_e  = active_e + idle_e
        imbalance = pstdev(core_times) / MEAN_LOAD if TOTAL_EXEC > 0 else 0.0

        nm = min(makespan / MAX_MAKESPAN, 1.0)
        ne = min(total_e / MAX_ENERGY, 1.0)
        ni = min(imbalance, 1.0)

        score = W_MAKESPAN * nm + W_ENERGY * ne + W_IMBALANCE * ni

        individual.core_times    = core_times
        individual.total_energy  = total_e
        individual.makespan      = makespan
        individual.imbalance     = imbalance
        individual.fitness_value = score
        return (score,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate",    tools.cxUniform,   indpb=args.crossover_rate)
    toolbox.register("mutate",  tools.mutUniformInt, low=0, up=args.num_cores-1, indpb=args.mutation_rate)
    toolbox.register("select",  tools.selTournament, tournsize=3)
    toolbox.register("clone",   copy.deepcopy)
    return toolbox


def main(args, toolbox):
    run_id = str(uuid.uuid4())

    # Prime the population
    pop = toolbox.population()
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    hall = tools.HallOfFame(1)
    best_per_gen = []
    avg_per_gen  = []
    std_per_gen  = []
    stagnation   = 0
    best_so_far  = min(ind.fitness.values[0] for ind in pop)

    start = time.time()
    LOG_INTERVAL = 10

    for gen in range(1, args.num_generations + 1):
        hall.update(pop)
        curr_best = hall[0].fitness.values[0]
        if curr_best < best_so_far:
            best_so_far = curr_best
            stagnation = 0
        else:
            stagnation += 1

        if stagnation >= args.stagnation_limit:
            print(f"Early stopping at generation {gen}.")
            break

        # Standard GA loop
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < args.crossover_rate:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for mut in offspring:
            if random.random() < args.mutation_rate:
                toolbox.mutate(mut)
                del mut.fitness.values

        invalids = [ch for ch in offspring if not ch.fitness.valid]
        for ch in invalids:
            ch.fitness.values = toolbox.evaluate(ch)

        offspring[0] = hall[0]
        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]
        best_per_gen.append(min(fits))
        avg_per_gen.append(mean(fits))
        std_per_gen.append(pstdev(fits))

        if gen % LOG_INTERVAL == 0 or gen == args.num_generations:
            print(f"Gen {gen:4d} → Best={best_per_gen[-1]:.3f} | Avg={avg_per_gen[-1]:.3f} | Std={std_per_gen[-1]:.3f}")

    elapsed = time.time() - start
    best = hall[0]

    ###############Build & Upload JSON ############################
    result = {
        "run_id":               run_id,
        "generations_executed": gen,
        "elapsed_time_s":       elapsed,
        "best_individual":      list(best),
        "best_fitness":         best.fitness.values[0],
        "best_per_generation":  best_per_gen,
        "avg_per_generation":   avg_per_gen,
        "std_per_generation":   std_per_gen
    }
    json_output = json.dumps(result, indent=2)

    print("\n=== RUN SUMMARY JSON ===")
    print(json_output)

    conn_str  = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = os.getenv("BLOB_CONTAINER")
    if not conn_str or not container:
        raise RuntimeError("Set AZURE_STORAGE_CONNECTION_STRING & BLOB_CONTAINER")

    svc = BlobServiceClient.from_connection_string(conn_str)
    try:
        svc.create_container(container)
    except Exception:
        pass

    blob = svc.get_blob_client(container=container, blob=f"{run_id}.txt")
    blob.upload_blob(json_output, overwrite=True)

    ############### Plot & Upload PNG #############################
    plt.plot(best_per_gen, label="Best")
    plt.plot(avg_per_gen,  label="Average")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)

    plot_blob = svc.get_blob_client(container=container, blob=f"{run_id}.png")
    plot_blob.upload_blob(buf.read(), overwrite=True)
    print(f"Uploaded plot to blob: {run_id}.png")

    plt.show()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    exec_times = [random.randint(10, 20) for _ in range(args.num_tasks)]
    setup_deap()
    toolbox = make_toolbox(args, exec_times)
    main(args, toolbox)
