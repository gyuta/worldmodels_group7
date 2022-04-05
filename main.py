# 環境チーム
# from env_team import Evaluator
from wm_team import WorldModel
from algo_team.DIAYN import DIAYN
from algo_team.Trainer import Trainer
from algo_team.config.LL import config
from util import *
import gym
env = gym.make("LunarLanderContinuous-v2")

USE_WORLDMODEL = False # 適宜変更

if USE_WORLDMODEL:
    # 世界モデルチーム
    worldmodel = WorldModel(env)
    worldmodel.train()
    lowenv = worldmodel
else:
    lowenv = env

config.low_env = lowenv
config.high_env = gym.make("LunarLanderContinuous-v2")
config.save_path = get_outputdir()

config.num_episodes_to_run = 10
config.hyperparameters["DIAYN"]["num_unsupservised_episodes"] = 5

diayn_trainer = Trainer(config, DIAYN)
diayn_trainer.train()
diayn = diayn_trainer.agents[0]

# 環境チーム
# from env_team.eval import Evaluator
# ev = Evaluator(env, diayn)  # diayn の性能を評価する。元の環境に依存する？状態の網羅性や報酬などを見るべきか
# ev.evaluate(diayn)