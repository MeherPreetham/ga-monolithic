#!/usr/bin/env python3
import os
import uuid
import json
import io
import time
import random
import copy
import asyncio

from typing import Dict
from statistics import mean, pstdev

import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from deap import base, creator, tools
from azure.storage.blob import BlobServiceClient

app = FastAPI()

# In‐memory stores for simplicity
job_states: Dict[str, Dict] = {}
job_results: Dict[str, Dict] = {}

############# Request model #############
class RunRequest(BaseModel):
    num_tasks: int        = Field(default=int(os.getenv("NUM_TASKS", 1000)), gt=0)
    num_cores: int        = Field(default=int(os.getenv("NUM_CORES", 16)), gt=1)
    num_population: int   = Field(default=int(os.getenv("NUM_POPULATION", 100)), gt=1)
    num_generations: int  = Field(default=int(os.getenv("NUM_GENERATIONS", 500)), gt=1)
    crossover_rate: float = Field(default=float(os.getenv("CROSSOVER_RATE", 0.8)), ge=0, le=1)
    mutation_rate: float  = Field(default=float(os.getenv("MUTATION_RATE", 0.2)), ge=0, le=1)
    base_energy: float    = Field(default=float(os.getenv("BASE_ENERGY", 0.01)), ge=0)
    idle_energy: float    = Field(default=float(os.getenv("IDLE_ENERGY", 0.002)), ge=0)
    stagnation_limit: int = Field(default=int(os.getenv("STAGNATION_LIMIT", 20)), gt=0)
    seed: int             = Field(default=int(os.getenv("SEED", 42)))

# === DEAP setup helpers ===
def setup_deap():
    try:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
    except RuntimeError:
        pass

def make_toolbox(req: RunRequest, exec_times):
    toolbox = base.Toolbox()
    toolbox.register("attr_core", random.randint, 0, req.num_cores - 1)
    toolbox.register("individual",
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.attr_core,
                     n=req.num_tasks)
    toolbox.register("population",
                     tools.initRepeat,
                     list,
                     toolbox.individual,
                     n=req.num_population)

    TOTAL_EXEC   = sum(exec_times)
    MEAN_LOAD    = TOTAL_EXEC / req.num_cores if req.num_cores > 0 else 1e-9
    MAX_MAKESPAN = TOTAL_EXEC
    MAX_ENERGY   = TOTAL_EXEC * req.base_energy \
                   + (req.num_cores - 1) * TOTAL_EXEC * req.idle_energy
    W_MAKESPAN, W_ENERGY, W_IMBALANCE = 0.4, 0.2, 0.4

    def evaluate(individual):
        core_times = [0.0] * req.num_cores
        for idx, c in enumerate(individual):
            core_times[c] += exec_times[idx]
        makespan = max(core_times)
        active_e = sum(ct * req.base_energy for ct in core_times)
        idle_e   = sum((makespan - ct) * req.idle_energy for ct in core_times)
        total_e  = active_e + idle_e
        imbalance = pstdev(core_times) / MEAN_LOAD if TOTAL_EXEC > 0 else 0.0

        nm = min(makespan / MAX_MAKESPAN, 1.0)
        ne = min(total_e / MAX_ENERGY,   1.0)
        ni = min(imbalance,              1.0)

        score = W_MAKESPAN * nm + W_ENERGY * ne + W_IMBALANCE * ni

        individual.core_times    = core_times
        individual.total_energy  = total_e
        individual.makespan      = makespan
        individual.imbalance     = imbalance
        individual.fitness_value = score
        return (score,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate",    tools.cxUniform,   indpb=req.crossover_rate)
    toolbox.register("mutate",  tools.mutUniformInt,
                     low=0, up=req.num_cores - 1, indpb=req.mutation_rate)
    toolbox.register("select",  tools.selTournament, tournsize=3)
    toolbox.register("clone",   copy.deepcopy)
    return toolbox

########## API Endpoints ##############

@app.post("/run")
async def run(req: RunRequest):
    job_id = str(uuid.uuid4())
    job_states[job_id] = {
        "status": "running",
        "generation": 0,
        "best_fitness": None,
        "best_individual": None
    }
    # launch background GA task
    asyncio.create_task(_run_ga(job_id, req))
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def status(job_id: str):
    state = job_states.get(job_id)
    if not state:
        raise HTTPException(404, "Job not found")
    return state

@app.get("/result/{job_id}")
async def result(job_id: str):
    if job_states.get(job_id, {}).get("status") == "running":
        raise HTTPException(400, "Job still running")
    res = job_results.get(job_id)
    if not res:
        raise HTTPException(404, "Job not found or failed")
    return res

######## Internal GA runner ################

async def _run_ga(job_id: str, req: RunRequest):
    # Prepare Azure Blob client
    conn_str  = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = os.getenv("BLOB_CONTAINER")
    if not conn_str or not container:
        job_states[job_id]["status"] = "error"
        job_states[job_id]["error"]  = "Missing AZURE_STORAGE_CONNECTION_STRING or BLOB_CONTAINER"
        return

    # Generate execution times
    rng_exec = random.Random(req.seed)
    exec_times = [rng_exec.randint(10, 20) for _ in range(req.num_tasks)]

    # Setup GA
    setup_deap()
    toolbox = make_toolbox(req, exec_times)

    # Initialize
    pop = toolbox.population()
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    hall = tools.HallOfFame(1)

    best_per_gen, avg_per_gen, std_per_gen = [], [], []
    stagnation = 0
    best_so_far = min(ind.fitness.values[0] for ind in pop)

    start_time = time.time()
    for gen in range(1, req.num_generations + 1):
        hall.update(pop)
        curr_best = hall[0].fitness.values[0]

        if curr_best < best_so_far:
            best_so_far = curr_best
            stagnation = 0
        else:
            stagnation += 1

        # Record intermediate state
        job_states[job_id].update({
            "generation":     gen,
            "best_fitness":   curr_best,
            "best_individual": list(hall[0])
        })

        # Early stop
        if stagnation >= req.stagnation_limit:
            break

        # Produce next generation
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < req.crossover_rate:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for mut in offspring:
            if random.random() < req.mutation_rate:
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

    # Build final result
    elapsed = time.time() - start_time
    final = {
        "run_id":               job_id,
        "generations_executed": gen,
        "elapsed_time_s":       elapsed,
        "best_individual":      list(hall[0]),
        "best_fitness":         hall[0].fitness.values[0],
        "best_per_generation":  best_per_gen,
        "avg_per_generation":   avg_per_gen,
        "std_per_generation":   std_per_gen
    }

    # Upload JSON
    svc = BlobServiceClient.from_connection_string(conn_str)
    try:
        svc.create_container(container)
    except:
        pass
    txt_blob = svc.get_blob_client(container=container, blob=f"{job_id}.txt")
    txt_blob.upload_blob(json.dumps(final), overwrite=True)

    # Plot & upload PNG
    plt.figure()
    plt.plot(final["best_per_generation"], label="Best")
    plt.plot(final["avg_per_generation"],  label="Average")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    png_blob = svc.get_blob_client(container=container, blob=f"{job_id}.png")
    png_blob.upload_blob(buf.read(), overwrite=True)

    # Mark done
    job_states[job_id]["status"] = "done"
    job_results[job_id] = final
