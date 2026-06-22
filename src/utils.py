import numpy as np
import os

from overcooked_ai_py.mdp.actions import Direction, Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.agents.agent import GreedyHumanModel, StayAgent, RandomAgent
from overcooked_ai_py.agents.agent import AgentFromPolicy, AgentPair
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.utils import load_dict_from_file, load_pickle


from collab.collab import LLMAgents

from collections import defaultdict


def make_agent(alg: str, mdp, layout, **gptargs):

    if alg == "Stay":
        agent = StayAgent()

    elif alg == "Random":
        agent = RandomAgent()

    elif alg == "LLMPair" or alg == "Greedy":
        MLAM_PARAMS = {
            "start_orientations": False,
            "wait_allowed": True,
            "counter_goals": [],
            "counter_drop": [],
            "counter_pickup": [],
            "same_motion_goals": True,
        }
        counter_locations = mdp.get_counter_locations()
        MLAM_PARAMS["counter_goals"] = counter_locations
        MLAM_PARAMS["counter_drop"] = counter_locations
        MLAM_PARAMS["counter_pickup"] = counter_locations

        if alg == "LLMPair":
            mlam = MediumLevelPlanner.from_pickle_or_compute(
                mdp, MLAM_PARAMS, force_compute=True
            ).ml_action_manager
            agent = LLMAgents(mlam, layout, **gptargs)

        elif alg == "Greedy":
            mlam = MediumLevelPlanner.from_pickle_or_compute(
                mdp, MLAM_PARAMS, force_compute=True
            )
            agent = GreedyHumanModel(mlam)

    else:
        raise ValueError("Unsupported algorithm.")

    agent.set_mdp(mdp)

    return agent


# Example-retrieval embedding precompute is disabled in the OpenAI-only fork
# (no embeddings endpoint dependency). Kept as a stub for import compatibility.
def get_example_embedding(example_path, save_path=""):
    raise NotImplementedError("example retrieval is disabled in this fork")


def combine_statistic_dict(dict1, dict2, map, score):
    rs = dict1
    rs["actions"].append(dict2["actions"][0])
    rs["map"] = map
    rs["statistical_data"]["score"] = score
    rs["statistical_data"]["communication"][1] = dict2["statistical_data"]["communication"][1]
    rs["statistical_data"]["error"][1] = dict2["statistical_data"]["error"][1]
    rs["statistical_data"]["error_correction"][1] = dict2["statistical_data"]["error_correction"][1]

    rs["content"]["observation"][1] = dict2["content"]["observation"][1]
    rs["content"]["reflection"][1] = dict2["content"]["reflection"][1]
    rs["content"]["content"][1] = dict2["content"]["content"][1]
    rs["content"]["action_list"][1] = dict2["content"]["action_list"][1]

    return rs
